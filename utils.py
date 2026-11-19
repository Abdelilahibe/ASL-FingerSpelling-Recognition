import numpy as np

def normalize(landmarks):
    pts = np.array(landmarks, dtype=np.float32).reshape(21, 3)
    pts = pts - pts[0]                                   # make the wrist the origin
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))   # hand size
    if scale > 0:
        pts = pts / scale                                # same size at any distance
    return pts.flatten()