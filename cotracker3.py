import torch
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer
# from cotracker.models.core.model_utils import get_points_on_a_grid
from scipy.interpolate import griddata
from flow_vis import flow_to_color
import numpy as np
import os
import cv2
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN

video_path = "data/Recognition test_Easy.mp4"

frames_all = iio.imread(video_path, plugin="FFMPEG")  # plugin="pyav" # T, H, W, 3
num_chunks = frames_all.shape[0] // 24


class LPF:
    '''
    low pass filter 2d position
    '''
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.prev_pos = None
        
    def update(self, pos):
        if isinstance(pos, list):
            pos = np.array(pos)
        if self.prev_pos is None:
            self.prev_pos = pos
        else:
            self.prev_pos = self.alpha * self.prev_pos + (1 - self.alpha) * pos
        return self.prev_pos

def find_whirlpool_dbscan(vorticity, intensity_threshold=0.5, eps=5, min_samples=5):
    """
    Locate the whirlpool's center and radius using DBSCAN clustering.
    
    Parameters:
      vorticity (np.ndarray): 2D array of vorticity values.
      intensity_threshold (float): Fraction of max vorticity for thresholding.
      eps (float): The maximum distance between two samples for DBSCAN.
      min_samples (int): Minimum number of samples in a neighborhood for DBSCAN.
    
    Returns:
      center (np.ndarray): Estimated center [row, col] of the whirlpool.
      radius (float): Estimated radius (maximum distance of points in the cluster from the center).
      cluster_coords (np.ndarray): Array of coordinates belonging to the cluster.
    """
    # Smooth the vorticity field.
    vorticity_smoothed = cv2.GaussianBlur(vorticity, (5, 5), 0)
    
    # Create a binary mask for high vorticity regions.
    binary = vorticity_smoothed > (intensity_threshold * vorticity_smoothed.max())
    
    # Extract coordinates of high-vorticity pixels.
    coords = np.column_stack(np.nonzero(binary))
    if coords.shape[0] == 0:
        raise ValueError("No significant vorticity pixels found. Adjust the intensity_threshold.")
    
    # Cluster the coordinates with DBSCAN.
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    
    # Get unique cluster labels and their counts (ignore noise, labeled as -1).
    labels, counts = np.unique(clustering.labels_, return_counts=True)
    valid_mask = labels != -1
    labels = labels[valid_mask]
    counts = counts[valid_mask]
    
    if len(labels) == 0:
        raise ValueError("No clusters found. Adjust DBSCAN parameters.")
    
    # Choose the largest cluster as the whirlpool.
    largest_cluster_label = labels[np.argmax(counts)]
    cluster_coords = coords[clustering.labels_ == largest_cluster_label]
    
    # Compute the centroid of the cluster.
    center = np.mean(cluster_coords, axis=0)
    
    # Estimate the radius as the maximum distance from the center.
    distances = np.sqrt(np.sum((cluster_coords - center)**2, axis=1))
    radius = distances.max()
    
    return center, radius, cluster_coords


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


# TODO: huanzhe, toggle the following numbers to see the effect
clip = False
use_ow = True
filter_circle = False
dbscan_thres_vorticity = 0.3 # only affects results if use_ow is False
dbscan_thres_ow = 0.1 # only affects results if use_ow is True
grid_size = 40 # this number decides the accuracy, bigger = slower but better 
# end of TODO



device = 'cuda'
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
pred_tracks_all = []
pred_visibility_all = []
flow_frames = []
vorticity_frames = []
ow_frames = []
lps_center = LPF()
lps_radius = LPF()

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
            if clip:
                sorted_vorticity = np.sort(vorticity.flatten())
                p10 = np.percentile(sorted_vorticity, 10)
                p90 = np.percentile(sorted_vorticity, 90)
                vorticity_clipped = np.clip(vorticity, p10, p90)
                # Sort and clip ow_criterion
                sorted_ow = np.sort(ow_criterion.flatten())
                p10 = np.percentile(sorted_ow, 10)
                p90 = np.percentile(sorted_ow, 90)
                ow_criterion_clipped = np.clip(ow_criterion, p10, p90)
            else:
                vorticity_clipped = vorticity
                ow_criterion_clipped = ow_criterion
            
            # find whirlpool
            # center, radius, cluster_coords = find_whirlpool_dbscan(vorticity_clipped)
            if use_ow:
                center, radius, cluster_coords = find_whirlpool_dbscan(-ow_criterion_clipped, intensity_threshold=dbscan_thres_ow)
            else:
                center, radius, cluster_coords = find_whirlpool_dbscan(vorticity_clipped, intensity_threshold=dbscan_thres_vorticity)
            if filter_circle:
                center = lps_center.update(center)
                radius = lps_radius.update(radius)
                
            # compute average flow speed in circle
            flow_speed = np.sqrt(flow_x_grid ** 2 + flow_y_grid ** 2)
            circle_mask = np.zeros_like(flow_speed)
            cv2.circle(circle_mask, (int(center[1]), int(center[0])), int(radius), (1), -1)
            circle_mask = circle_mask.astype(np.bool)
            flow_speed_in_circle = np.mean(flow_speed[circle_mask])
            print(f"flow speed in circle: {flow_speed_in_circle}")
            
            # draw empty circle on vorticity_clipped
            vorticity_clipped = vorticity_clipped.astype(np.float32)
            vorticity_clipped = (vorticity_clipped - vorticity_clipped.min()) / (vorticity_clipped.max() - vorticity_clipped.min())
            vorticity_clipped = (vorticity_clipped * 255).astype(np.uint8)
            cv2.circle(vorticity_clipped, (int(center[1]), int(center[0])), int(radius), (0, 0, 255), 2)
            vorticity_frames.append(vorticity_clipped)
            
            # visualize mask on ow_frames
            ow_criterion_clipped = ow_criterion_clipped.astype(np.float32)
            ow_criterion_clipped = (ow_criterion_clipped - ow_criterion_clipped.min()) / (ow_criterion_clipped.max() - ow_criterion_clipped.min())
            ow_criterion_clipped = (ow_criterion_clipped * 255).astype(np.uint8)
            ow_criterion_clipped[circle_mask] = 1
            ow_frames.append(ow_criterion_clipped)
            

            
            center = center * flow_grid_spacing
            radius = radius * flow_grid_spacing
            frame_idx_all = chunk_idx * 24 + frame_idx
            frames_all[frame_idx_all] = cv2.circle(frames_all[frame_idx_all], (int(center[1]), int(center[0])), int(radius), (0, 0, 255), 2)

        


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
vis = Visualizer(save_dir="./saved_videos", mode="optical_flow", pad_value=120, linewidth=3, tracks_leave_trace=0) # change the last input to N to see trace for N steps, or to -1 for many steps
vis.visualize(video_all, pred_tracks_all, pred_visibility_all)