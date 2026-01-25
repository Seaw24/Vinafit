import numpy as np

# --- CONFIGURATION ---
# Flexion Logic: 0 = Standing Straight, 120 = Deep Squat
SQUAT_RESET_THRESHOLD_DESCENDING = 15   # Start counting when knee bends > 15 deg
SQUAT_RESET_THRESHOLD_ASCENDING = 75   # Reset when knee returns to < 75 deg

# Skeleton Connections for Drawing
POSE_CONNECTIONS = frozenset([
    (11, 12), (11, 23), (12, 24), (23, 24), 
    (23, 25), (24, 26), (25, 27), (26, 28), 
    (27, 29), (28, 30), (29, 31), (30, 32), 
    (15, 21), (15, 17), (15, 19), (17, 19), 
    (16, 22), (16, 18), (16, 20), (18, 20), 
    (11, 13), (13, 15), (12, 14), (14, 16)
])

MODEL_PATH = r"pose_landmarker_full.task"

def calculate_3_point_angle(a, b, c):
    """ 
    Returns Geometric Angle (0-180).
    180 = Straight Line. 90 = Right Angle. 
    """
    radians_a = np.arctan2(a.y - b.y, a.x - b.x)
    radians_c = np.arctan2(c.y - b.y, c.x - b.x)
    angle_degrees = np.abs((radians_c - radians_a) * 180.0 / np.pi)
    
    if angle_degrees > 180.0:
        angle_degrees = 360.0 - angle_degrees
        
    return angle_degrees 

def calculate_vertical_angle(pivot, point):
    """ 
    Returns deviation from Vertical Line. 
    0 = Perfectly Vertical. 
    """
    virtual_vertical_point = [pivot.x, pivot.y - 0.5] 
    p_vertical = np.array(virtual_vertical_point)
    p_pivot = np.array([pivot.x, pivot.y])
    p_point = np.array([point.x, point.y])
    
    radians_vertical = np.arctan2(p_vertical[1] - p_pivot[1], p_vertical[0] - p_pivot[0])
    radians_point = np.arctan2(p_point[1] - p_pivot[1], p_point[0] - p_pivot[0])
    
    angle_degrees = np.abs((radians_vertical - radians_point) * 180.0 / np.pi)
    if angle_degrees > 180.0:
        angle_degrees = 360.0 - angle_degrees
        
    return angle_degrees