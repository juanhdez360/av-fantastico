import torch
import numpy as np

def rotate_90cc(xy):
    """Rotates (x, y) by 90 deg CCW -> (-y, x). Expects shape (2, N)"""
    return np.stack([-xy[1], xy[0]], axis=0)

def compute_pure_pursuit_from_rotated(rotated_xy, lookahead_distance=5.0, wheelbase=2.8):
    """Computes Pure Pursuit steering angle from a (2, N) rotated trajectory."""
    distances = np.linalg.norm(rotated_xy, axis=0)
    valid_indices = np.where(distances >= lookahead_distance)[0]
    
    target_idx = valid_indices[0] if len(valid_indices) > 0 else -1
        
    target_x = rotated_xy[0, target_idx]
    target_y = rotated_xy[1, target_idx]
    
    ld_sq = target_x**2 + target_y**2
    if ld_sq == 0.0:
        return 0.0
        
    curvature = (2.0 * target_x) / ld_sq
    return float(np.arctan(curvature * wheelbase))

def compute_target_velocity(rotated_xy, time_horizon=6.4):
    """
    Computes average target velocity along the predicted trajectory.
    
    Args:
        rotated_xy (np.ndarray): Array of shape (2, N) containing [X, Y].
        time_horizon (float): Total time in seconds for the future trajectory.
        
    Returns:
        float: Target velocity in meters per second (m/s).
    """
    # 1. Prepend ego current position (0,0) to calculate initial movement
    ego_pos = np.zeros((2, 1))
    path_points = np.hstack((ego_pos, rotated_xy))
    
    # 2. Calculate differences between consecutive points (dx, dy)
    diffs = np.diff(path_points, axis=1)
    
    # 3. Calculate Euclidean distance for each step and sum them up
    step_distances = np.linalg.norm(diffs, axis=0)
    total_distance = np.sum(step_distances)
    
    # 4. Velocity = total distance / total time
    target_velocity = total_distance / time_horizon
    
    return float(target_velocity)


if __name__ == "__main__":
    
 
    pred_xyz = torch.load("controller/inputs/pred_xyz.pt", weights_only=False, map_location='cpu')
    data = torch.load("controller/inputs/clip_data.pt", weights_only=False, map_location='cpu')


    # Evaluation parameters
    lookahead_dist = 6.0  
    car_wheelbase = 2.8   
    future_time_s = 6.4  

    try:
        for i in range(pred_xyz.shape[2]):
            # Evaluate prediction
            pred_xy = pred_xyz.cpu()[0, 0, i, :, :2].T.numpy()
            pred_xy_rot = rotate_90cc(pred_xy)
            
            pred_steer_rad = compute_pure_pursuit_from_rotated(pred_xy_rot, lookahead_dist, car_wheelbase)
            pred_speed_ms = compute_target_velocity(pred_xy_rot, future_time_s)
            
            # Evaluate ground truth
            gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
            gt_xy_rot = rotate_90cc(gt_xy)
            
            gt_steer_rad = compute_pure_pursuit_from_rotated(gt_xy_rot, lookahead_dist, car_wheelbase)
            gt_speed_ms = compute_target_velocity(gt_xy_rot, future_time_s)
            
            # Print metrics
            print(f"Mode {i}:")
            print(f"  GT Control:   Steer {np.degrees(gt_steer_rad):.2f} deg, Speed {gt_speed_ms:.2f} m/s ({gt_speed_ms*3.6:.1f} km/h)")
            print(f"  Pred Control: Steer {np.degrees(pred_steer_rad):.2f} deg, Speed {pred_speed_ms:.2f} m/s ({pred_speed_ms*3.6:.1f} km/h)")
            print(f"  Errors:       Steer {abs(np.degrees(gt_steer_rad) - np.degrees(pred_steer_rad)):.2f} deg, Speed {abs(gt_speed_ms - pred_speed_ms):.2f} m/s\n")
            
    except NameError:
        print("Error: Please assign your loaded tensors to 'pred_xyz' and 'data' before running the loop.")