import time
from functools import wraps

import torch
from torch import nn
from tqdm import tqdm

from dot.utils.io import read_config
from dot.utils.torch import get_grid, sample_mask_points, sample_points

from .optical_flow import OpticalFlow
from .shelf import CoTracker, CoTracker2, Tapir


def performance_measure(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Reset CUDA memory stats
        torch.cuda.reset_peak_memory_stats()

        # Measure start time
        start_time = time.time()

        # Execute the function
        result = func(*args, **kwargs)

        # Measure end time
        end_time = time.time()

        # Calculate peak memory usage
        peak_memory = torch.cuda.max_memory_allocated() / 1e9

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        print(f"Function '{func.__name__}':")
        print(f"  - Peak memory usage: {peak_memory:.2f} GB")
        print(f"  - Execution time: {elapsed_time:.2f} seconds")

        return result

    return wrapper


class PointTracker(nn.Module):
    def __init__(
        self,
        height,
        width,
        tracker_config,
        tracker_path,
        estimator_config,
        estimator_path,
    ):
        super().__init__()
        model_args = read_config(tracker_config)
        model_dict = {
            "cotracker": CoTracker,
            "cotracker2": CoTracker2,
            "tapir": Tapir,
            "bootstapir": Tapir,
        }
        # print arguments
        # print(f"Using model {model_args.name}")
        # print(f"Height: {height}, Width: {width}")
        # print(f"Model arguments: {model_args}")
        # print(f"Estimator config: {estimator_config}, Estimator path: {estimator_path}")
        # print(f"Tracker path: {tracker_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = model_args.name
        self.model = model_dict[model_args.name](model_args)
        self.model = self.model.to(self.device)

        if tracker_path is not None:
            # Load the state dict to the same device as the model
            state_dict = torch.load(tracker_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.optical_flow_estimator = OpticalFlow(
            height, width, estimator_config, estimator_path
        )

    @performance_measure
    def forward(self, data, mode, **kwargs):
        if mode == "tracks_at_motion_boundaries":
            return self.get_tracks_at_motion_boundaries(data, **kwargs)
        elif mode == "flow_from_last_to_first_frame":
            return self.get_flow_from_last_to_first_frame(data, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def get_tracks_at_motion_boundaries(
        self,
        data,
        num_tracks=8192,
        sim_tracks=2048,
        sample_mode="all",
        random_sampling=True,
        boundary_sampling_ratio=0.5,
        debug=False,
        **kwargs,
    ):
        video = data["video"]
        N, S = num_tracks, sim_tracks
        B, T, _, H, W = video.shape

        print(f"Tracker Video shape: {video.shape}")
        assert N % S == 0

        # Define sampling strategy

        # This is the bottleneck, sample points per frame takes 0.3 second for one iteration
        # But if so, why is this slow then single segment?
        # TODO: improved sampling
        # if this sees all frames in the video,
        # number of samples can be decided via the amplitude of the motion

        if sample_mode == "all":
            samples_per_step = [S // T for _ in range(T)]
            samples_per_step[0] += S - sum(samples_per_step)
            backward_tracking = True
            flip = False
        elif sample_mode == "all_distributed":
            # Distribute samples to all frames (early frames first)
            samples_per_step = [(S + T - (i + 1)) // T for i in range(T)]
            assert sum(samples_per_step) == S
            backward_tracking = True
            flip = False
        elif sample_mode == "first":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = False
        elif sample_mode == "last":
            samples_per_step = [0 for _ in range(T)]
            samples_per_step[0] += S
            backward_tracking = False
            flip = True
        else:
            raise ValueError(f"Unknown sample mode {sample_mode}")

        if flip:
            video = video.flip(dims=[1])

        # Track batches of points
        # This may be possible to parallelize
        tracks = []
        motion_boundaries = {}
        cache_features = True

        print(
            f"Sample mode: {sample_mode}, Number of samples per step: {samples_per_step}"
        )
        
        motion_boundaries_time = time.time()
        for _ in tqdm(range(N // S), desc="Track batch of points", leave=False):
            src_points = []
            for src_step, src_samples in enumerate(samples_per_step):
                if src_samples == 0:
                    continue

                if not src_step in motion_boundaries:
                    tgt_step = src_step - 1 if src_step > 0 else src_step + 1
                    data = {
                        "src_frame": video[:, src_step],
                        "tgt_frame": video[:, tgt_step],
                    }
                    pred = self.optical_flow_estimator(
                        data, mode="motion_boundaries", **kwargs
                    )
                    motion_boundaries[src_step] = pred["motion_boundaries"]

                src_boundaries = motion_boundaries[src_step]

                point_sampling_time = time.time()

                src_points.append(
                    sample_points(
                        src_step,
                        src_boundaries,
                        src_samples,
                        random_sampling=random_sampling,
                        boundary_sampling_ratio=boundary_sampling_ratio,
                    ),
                )

            src_points = torch.cat(src_points, dim=1)
            if debug:
                print(f"Motion boundaries time: {time.time() - motion_boundaries_time}")

            # src_points are diff
            with torch.no_grad():
                traj, vis = self.model(
                    video, src_points, backward_tracking, cache_features
                )

            tracks.append(torch.cat([traj, vis[..., None]], dim=-1))
            cache_features = False
        tracks = torch.cat(tracks, dim=2)

        if flip:
            tracks = tracks.flip(dims=[1])

        return {"tracks": tracks}

    def get_flow_from_last_to_first_frame(self, data, sim_tracks=2048, **kwargs):
        video = data["video"]
        video = video.flip(dims=[1])
        src_step = 0  # We have flipped video over temporal axis so src_step is 0
        B, T, C, H, W = video.shape
        S = sim_tracks
        backward_tracking = False
        cache_features = True
        flow = get_grid(H, W, shape=[B]).cuda()
        flow[..., 0] = flow[..., 0] * (W - 1)
        flow[..., 1] = flow[..., 1] * (H - 1)
        alpha = torch.zeros(B, H, W).cuda()
        mask = torch.ones(H, W)
        pbar = tqdm(total=H * W // S, desc="Track batch of points", leave=False)
        while torch.any(mask):
            points, (i, j) = sample_mask_points(src_step, mask, S)
            idx = i * W + j
            points = points.cuda()[None].expand(B, -1, -1)

            traj, vis = self.model(video, points, backward_tracking, cache_features)
            traj = traj[:, -1]
            vis = vis[:, -1].float()

            # Update mask
            mask = mask.view(-1)
            mask[idx] = 0
            mask = mask.view(H, W)

            # Update flow
            flow = flow.view(B, -1, 2)
            flow[:, idx] = traj - flow[:, idx]
            flow = flow.view(B, H, W, 2)

            # Update alpha
            alpha = alpha.view(B, -1)
            alpha[:, idx] = vis
            alpha = alpha.view(B, H, W)

            cache_features = False
            pbar.update(1)
        pbar.close()
        return {"flow": flow, "alpha": alpha}
