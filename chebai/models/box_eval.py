import numpy as np

n = len(boxes)
containment_matrix = np.zeros((n, n), dtype=bool)
threshold = 0.99

for i in range(n):
    for j in range(n):
        if i != j:
            box1 = boxes[i]
            box2 = boxes[j]

            min_corners_box_1 = np.min(box1, axis=-2)
            max_corners_box_1 = np.max(box1, axis=-2)

            vol_box_1 = np.prod(max_corners_box_1 - min_corners_box_1)

            min_corners_box_2 = np.min(box2, axis=-2)
            max_corners_box_2 = np.max(box2, axis=-2)

            a = np.maximum(min_corners_box_1, min_corners_box_2) # right face of intersection
            b = np.minimum(max_corners_box_1, max_corners_box_2) # left face of intersection

            intersection_per_dim = (a <= b) * (b - a)
            vol_intersection = np.prod(intersection_per_dim)
            box_1_is_contained_in_box_2 = vol_intersection / vol_box_1

            containment_matrix[i][j] = box_1_is_contained_in_box_2 > threshold
