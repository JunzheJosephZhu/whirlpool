import torch
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer
# from cotracker.models.core.model_utils import get_points_on_a_grid
from scipy.interpolate import griddata
from flow_vis import flow_to_color
import numpy as np
import os
import cv2

video_path = "data/Recognition test_Easy.mp4"

frames_all = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav" # T, H, W, 3
num_chunks = frames_all.shape[0] // 24

def compute_dx(grid, spacing, mask):
    '''
    grid: np.array, [H, W]
    '''
    grid_padded = np.pad(grid, ((0, 0), (0, 1)), mode='edge')
    delta_x = grid_padded[:, 1:] - grid_padded[:, :-1]
    dx = delta_x / spacing
    dx[~mask] = 0
    return dx

def compute_dy(grid, spacing, mask):
    '''
    grid: np.array, [H, W]
    Should be in pillow's y-down system, where top-left coordinate is (0, 0)
    '''
    grid_padded = np.pad(grid, ((0, 1), (0, 0)), mode='edge')
    delta_y = grid_padded[1:] - grid_padded[:-1]
    dy = delta_y / spacing
    dy[~mask] = 0
    return dy


device = 'cuda'
grid_size = 40
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
pred_tracks_all = []
pred_visibility_all = []
flow_frames = []
vorticity_frames = []
ow_frames = []

for chunk_idx in range(num_chunks):
    frames = frames_all[chunk_idx * 24 : (chunk_idx + 1) * 24]
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].half().to(device)  # B T C H W
    with torch.no_grad():
        # with torch.amp.autocast(device_type=device, dtype=torch.half):
        #     cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)  
        #     # Process the video
        #     for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
        #         pred_tracks, pred_visibility = cotracker(
        #             video_chunk=video[:, ind : ind + cotracker.step * 2]
        #         )  # B T N 2,  B T N 1
        with torch.amp.autocast(device_type=device, dtype=torch.half):
            pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size)

        pred_tracks_all.append(pred_tracks)
        pred_visibility_all.append(pred_visibility)
                
        # dont visualize stuff that are not visible the entire time
        visibility_mask = pred_visibility.all(dim=1).squeeze() # [N]
        pred_tracks = pred_tracks.transpose(0, 2)[visibility_mask].transpose(0, 2) # [B, T, N, 2]
        pred_visibility = pred_visibility.transpose(0, 2)[visibility_mask].transpose(0, 2) # [B, T, N]

        tracks_np = pred_tracks.cpu().numpy()[0] # [T, N, 2]
        
        # interpolate points
        flow_grid_spacing = 3
        # sample_points = get_points_on_a_grid((frames.shape[1]), (frames.shape[1], frames.shape[2]))[0].cpu().numpy() # can use center if windowing
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, frames.shape[1], frames.shape[1] // flow_grid_spacing, device=device),
            torch.linspace(0, frames.shape[2], frames.shape[2] // flow_grid_spacing, device=device),
            indexing="ij",
        )
        grid_shape = grid_y.shape
        sample_points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).cpu().numpy()

        # Create a list to store flow color frames
        lpf_factor = 0.3

        for frame_idx in range(frames.shape[0] - 1): 
            frame = frames[frame_idx]
            flow = tracks_np[frame_idx + 1] - tracks_np[frame_idx] # [N, 2] # maybe lpf this
            if frame_idx == 0:
                flow_ema = flow
            else:
                flow_ema = lpf_factor * flow + (1 - lpf_factor) * flow_ema
            points = tracks_np[frame_idx] # [N, 2]
            
            flow_x_grid = griddata(points, flow_ema[..., 0], sample_points, method='cubic', fill_value=np.nan)
            flow_y_grid = griddata(points, flow_ema[..., 1], sample_points, method='cubic', fill_value=np.nan)
            flow_x_grid = flow_x_grid.reshape(grid_shape[0], grid_shape[1])
            flow_y_grid = flow_y_grid.reshape(grid_shape[0], grid_shape[1])
            mask = ~(np.isnan(flow_x_grid) | np.isnan(flow_y_grid))
            flow_x_grid[~mask] = flow_x_grid[mask].mean()
            flow_y_grid[~mask] = flow_y_grid[mask].mean()
            flow_color = flow_to_color(np.stack([flow_x_grid, flow_y_grid], axis=-1))
            flow_frames.append(flow_color)
            # expand mask
            mask_eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=5).astype(np.bool)
            # convert to vorticity
            dudx = compute_dx(flow_x_grid, flow_grid_spacing, mask_eroded)
            dudy = compute_dy(flow_x_grid, flow_grid_spacing, mask_eroded)
            dvdx = compute_dx(flow_y_grid, flow_grid_spacing, mask_eroded)
            dvdy = compute_dy(flow_y_grid, flow_grid_spacing, mask_eroded)
            vorticity = dvdx - dudy
            normal_strain = dudx - dvdy
            shear_strain = dudy + dvdx
            ow_criterion = normal_strain ** 2 + shear_strain ** 2 - vorticity ** 2
            # sort, clip, and save vorticity
            sorted_vorticity = np.sort(vorticity.flatten())
            p10 = np.percentile(sorted_vorticity, 10)
            p90 = np.percentile(sorted_vorticity, 90)
            vorticity_clipped = np.clip(vorticity, p10, p90)
            vorticity_frames.append(vorticity_clipped)
            # Sort and clip ow_criterion
            sorted_ow = np.sort(ow_criterion.flatten())
            p10 = np.percentile(sorted_ow, 10)
            p90 = np.percentile(sorted_ow, 90)
            ow_criterion_clipped = np.clip(ow_criterion, p10, p90)
            ow_frames.append(ow_criterion_clipped)

        


# Convert flow_frames to numpy array and ensure uint8 type
flow_frames = np.array(flow_frames).astype(np.uint8)
save_path = "./saved_videos/flow_visualization.mp4"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
iio.imwrite(save_path, flow_frames, plugin='FFMPEG', fps=30)
print(f"Flow visualization saved to {save_path}")

# vorticity visualization
vorticity_frames = np.array(vorticity_frames)
vorticity_frames = (vorticity_frames - vorticity_frames.min()) / (vorticity_frames.max() - vorticity_frames.min())
vorticity_frames = (vorticity_frames * 255).astype(np.uint8)
vorticity_save_path = "./saved_videos/vorticity_visualization.mp4"
iio.imwrite(vorticity_save_path, vorticity_frames, plugin='FFMPEG', fps=30)
print(f"Vorticity visualization saved to {vorticity_save_path}")

# ow visualization
ow_frames = np.array(ow_frames)
ow_frames = (ow_frames - ow_frames.min()) / (ow_frames.max() - ow_frames.min())
ow_frames = (ow_frames * 255).astype(np.uint8)
ow_save_path = "./saved_videos/ow_visualization.mp4"
iio.imwrite(ow_save_path, ow_frames, plugin='FFMPEG', fps=30)
print(f"OW visualization saved to {ow_save_path}")

video_all = torch.tensor(frames_all).permute(0, 3, 1, 2)[None].half().to(device)[:, :num_chunks * 24, ...]  # B T C H W
pred_tracks_all = torch.cat(pred_tracks_all, axis=1)
pred_visibility_all = torch.cat(pred_visibility_all, axis=1)
# Original visualization
vis = Visualizer(save_dir="./saved_videos", mode="optical_flow", pad_value=120, linewidth=3, tracks_leave_trace=-1)
vis.visualize(video_all, pred_tracks_all, pred_visibility_all)