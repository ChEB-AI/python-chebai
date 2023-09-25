import numpy as np

n = len(boxes)
containment_matrix = np.zeros((n, n), dtype=bool)

for i in range(n):
    for j in range(n):
        if i != j:
            box1 = boxes[i]
            box2 = boxes[j]

            min_corners_box_1 = np.minimum(box1[0], box1[1])
            max_corners_box_1 = np.maximum(box1[0], box1[1])

            min_corners_box_2 = np.minimum(box2[0], box2[1])
            max_corners_box_2 = np.maximum(box2[0], box2[1])

            dim = len(min_corners_box_1)

            membership_per_dim = []
            for d in range(dim):
                  a = max(min_corners_box_1[d], min_corners_box_2[d])
                  b = min(max_corners_box_1[d], max_corners_box_2[d])
                  intersection = (a <= b) * (b - a)
                  size_of_a = min_corners_box_1[d] + max_corners_box_1[d]

                  # if box_1 is not contained in box_2, then is_contained is zero

                  is_contained = abs(intersection / size_of_a)
                  membership_per_dim.append(1 if is_contained else 0)


            count = sum(1 for item in membership_per_dim if item == 1)
            box_1_is_contained_in_box_2 = (count == 10)
            containment_matrix[i][j] = box_1_is_contained_in_box_2