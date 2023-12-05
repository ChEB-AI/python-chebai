"""
Some utilities for evaluations. The code is not integrated to Chebai library yet.
"""
import torch
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
#-----------------------------------------------------------------


val_f1 = pd.read_csv("./outputs/metrics.csv.csv")
pd.DataFrame([list(val_f1[["val_micro-f1"]].dropna()), list(val_f1[["train_micro-f1"]].dropna())])
df1 = val_f1["val_micro-f1"].dropna().tolist()
df2 = val_f1["train_micro-f1"].dropna().tolist()
pd.DataFrame(zip(df1, df2), columns=["val_micro-f1", "train_micro-f1"]).plot()


#-----------------------------------------------------------------

checkpoint = torch.load("./saved_models/best_epoch=199_val_loss=0.0399.ckpt", map_location=torch.device('cpu'))
mboxes = checkpoint["state_dict"]["boxes"]
corner_1 = mboxes[:,:,0]
corner_2 = mboxes[:,:,1]
boxes = [[corner_1[i].cpu().detach().numpy(), corner_2[i].cpu().detach().numpy()] for i in range(854)]


#-----------------------------------------------------------------
# Extract top most and least presented classes in the training dataset
with open('./train_data/train.pkl', 'rb') as f:
  train_data = pickle.load(f)

chebi_labels = list(train_data.columns[3:])
ds_dict = train_data.iloc[:, 3:].sum().to_dict()

N = 200
top_represented_classes = [ i[0] for i in sorted( ds_dict.items(), key=lambda pair: pair[1], reverse=True )[:N] ]
lowest_represented_classes = [ i[0] for i in sorted( ds_dict.items(), key=lambda pair: -pair[1], reverse=True )[:N] ]

#visualize_boxes_3d(boxes, chebi_labels, limits=lowest_represented_classes )

def visualize_boxes_3d(boxes, labels, limits):

    plt.figure(figsize=(40,40))
    ax = plt.axes(projection='3d')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])

    for idx_i, box_i in enumerate(boxes):
        min_corner, max_corner = box_i
        vertices = [
            [min_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]]
        ]
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[4], vertices[7], vertices[3], vertices[0]]
        ]
        if labels[idx_i] in limits:
          poly3d = Poly3DCollection(faces, color='blue', linewidths=1, edgecolors='r', alpha=0.02)
          ax.add_collection3d(poly3d)
          ax.text(min_corner[0], min_corner[1], min_corner[2], labels[idx_i], color='green', fontsize=24)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Visualization of Chebi classes in 3D')
    plt.show()

# Example usage:
# visualize_boxes_3d(boxes, chebi_labels, limits=top_represented_classes )
# visualize_boxes_3d(boxes, chebi_labels, limits=lowest_represented_classes )

#-----------------------------------------------------------------
# Calculate containments based on boxes:

n = len(boxes)
containment_matrix = np.zeros((n, n), dtype=float)

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
            product_of_b = 1
            product_of_intersection = 1

            for d in range(dim):
                left_most_corner_of_intersection = max(min_corners_box_1[d], min_corners_box_2[d])
                right_most_corner_of_intersection = min(max_corners_box_1[d], max_corners_box_2[d])

                intersection = (left_most_corner_of_intersection <= right_most_corner_of_intersection) * (right_most_corner_of_intersection - left_most_corner_of_intersection)
                product_of_intersection *= intersection

                size_of_a =  max_corners_box_1[d] - min_corners_box_1[d]
                size_of_b =  max_corners_box_2[d] - min_corners_box_2[d]
                product_of_b *= size_of_b

            if product_of_b:
                containment_matrix[j][i] = ( product_of_intersection /product_of_b)

# A heatmap for containments:

binary_data = containment_matrix.astype(int)
fig, ax = plt.subplots(figsize=(20, 8))
im = ax.imshow(binary_data, cmap='coolwarm')
plt.colorbar(im, ticks=[0, 1], label='True/False')
plt.grid(False)
plt.show()



#--------------------------------


with open('./chebi_class_ancesstors/class_ancesstors.pkl', 'rb') as f:
  train_data = pickle.load(f)

n = 854
chebi_containment_matrix = np.zeros((n, n), dtype=bool)

for idx_i, chebi_id_i in enumerate(chebi_labels):
  for idx_j, chebi_id_j in enumerate(chebi_labels):
    if idx_i != idx_j:

      CHEBI_ID_i = "CHEBI:{id}".format(id=chebi_id_i)
      CHEBI_ID_j = "CHEBI:{id}".format(id=chebi_id_j)
      all_superclasses = class_ansectors[CHEBI_ID_j]

      if CHEBI_ID_i in all_superclasses:
        chebi_containment_matrix[idx_j][idx_i] = True


#------------------------------------
#F1 score for hierarchy

y_true = chebi_containment_matrix
y_pred = containment_matrix.astype(bool)

print(precision_score(y_true, y_pred, average="micro"))
print(recall_score(y_true, y_pred, average="micro"))
print(f1_score(y_true, y_pred, average="micro"))