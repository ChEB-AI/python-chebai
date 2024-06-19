import torch
import pickle
import pandas as pd
from collections import defaultdict
import numpy as np
import networkx as nx

raw_chebi_50_file_path = './data/ChEBI50/chebi_v200/raw/train.pkl'
chebi_50_file_path = 'all.pt'
chebi_100_file_path = './chebi100/test.pkl'

embedded_points_file_path =  'model_outputs/w1_embedded_points_chebi50_all.pkl' # './model_outputs/w2_embedded_points_chebi50_all.pkl'
model_learned_boxes_file_path = 'model_outputs/w1_boxes_chebi50_all.pkl' # './model_outputs/w2_boxes.pkl' #

with open(raw_chebi_50_file_path, 'rb') as raw_chebi_50_file_path_file:
    train_50_raw = pickle.load(raw_chebi_50_file_path_file)
raw_chebi_50_file_path_file.close()

with open(model_learned_boxes_file_path, 'rb') as model_learned_file:
    model_learned_boxes = pickle.load(model_learned_file)[0]
model_learned_file.close()

raw_chebi_50_file_path_file.close()
chebi_50_labels = list(train_50_raw.columns[3:])

with open(embedded_points_file_path, 'rb') as embedded_points_chebi50_all_file:
    embedded_points_chebi50_all = pickle.load(embedded_points_chebi50_all_file)
embedded_points_chebi50_all_file.close()

test_100 = pd.read_pickle(chebi_100_file_path)
chebi_100_labels = list(test_100.columns[3:])

chebi_50_all = torch.load(chebi_50_file_path)

chebi_50_ground_truth = []
for item in chebi_50_all:
    chebi_50_ground_truth.append(item['labels'])

points_with_labels_df = pd.DataFrame(chebi_50_ground_truth, columns=chebi_50_labels)
points_with_labels_df['Embedded_SMILES'] = embedded_points_chebi50_all

points_with_only_extra_labels_df = points_with_labels_df[points_with_labels_df.columns.difference(chebi_100_labels)]

with open('./points_with_only_extra_labels_df.pkl', 'wb') as points_with_only_extra_labels_df_file:
    pickle.dump(points_with_only_extra_labels_df, points_with_only_extra_labels_df_file)
points_with_only_extra_labels_df_file.close()

#--------------------------------------------------------------------------------------------------------------------

labels = list(points_with_only_extra_labels_df.columns)
class_labels = labels[:-1]

def remove_outliers(points, threshold=1.0):
    q1 = np.percentile(points, 25, axis=0)
    q3 = np.percentile(points, 75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    mask = np.all((points >= lower_bound) & (points <= upper_bound), axis=1)
    return points[mask]

instances_of_labels = defaultdict(list)
for class_label in class_labels:
    instances = list(points_with_only_extra_labels_df.loc[points_with_only_extra_labels_df[class_label] == True]['Embedded_SMILES'])
    instances_of_labels[class_label] = remove_outliers(np.array(instances)) #instances #

with open('./instances_of_labels.pkl', 'wb') as instances_of_labels_file:
    pickle.dump(instances_of_labels, instances_of_labels_file)
    instances_of_labels_file.close()

test_100 = pd.read_pickle('./chebi100/test.pkl')
main_labels = list(test_100.columns[3:])

corners_1 = model_learned_boxes[:, :, 0]
corners_2 = model_learned_boxes[:, :, 1]

learned_boxes = [[corners_1[i].cpu().detach().numpy(), corners_2[i].cpu().detach().numpy()] for i in range(854)]

boxes = defaultdict(list)
for label, instances in instances_of_labels.items():
    min_coordinates = np.min(instances, axis=0)
    max_coordinates = np.max(instances, axis=0)
    boxes[label] = [min_coordinates, max_coordinates]

boxes_labels = []
corners_1 = []
corners_2 = []

for label, box in boxes.items():
    boxes_labels.append(label)

    corner_1 = box[0]
    corner_2 = box[1]

    corners_1.append(corner_1)
    corners_2.append(corner_2)

chebi_50_boxes = [[corners_1[i], corners_2[i]] for i in range(len(corners_1))]
concatenated_boxes = learned_boxes + chebi_50_boxes

combined_labels = main_labels + boxes_labels

inferred_class_ansectors = defaultdict(list)
inferred_class_descendants = defaultdict(list)

boxes_to_evaluate = chebi_50_boxes #concatenated_boxes #chebi_50_boxes #concatenated_boxes #chebi_50_boxes #learned_boxes #concat_boxes
label_of_boxes_to_evaluate = boxes_labels #combined_labels # boxes_labels #boxes_labels #combined_labels #main_labels #combined_labels

new_boxes_inside_learned_boxes = defaultdict(list)
new_classes_subclass_of_base_classes = defaultdict(list)

with open('./chebi_200_transitive.pkl', 'rb') as f:
    transitive_closure = pickle.load(f)
closed_transitive_closure = transitive_closure.subgraph(combined_labels)

for k, v in boxes.items():
    new_box_class = "CHEBI:{id}".format(id=k)
    ground_truth_ancestors = [item for item in list(nx.ancestors(closed_transitive_closure, k)) if item in main_labels]
    new_classes_subclass_of_base_classes[new_box_class] = ["CHEBI:{id}".format(id=item) for item in ground_truth_ancestors]

    min_corner_new_box = v[0]
    max_corner_new_box = v[1]
    dim = len(min_corner_new_box)

    for learned_box_index, learned_box in enumerate(learned_boxes):
        learned_box_class = "CHEBI:{id}".format(id=main_labels[learned_box_index])

        min_corner_learned_box = np.minimum(learned_box[0], learned_box[1])
        max_corner_learned_box = np.maximum(learned_box[0], learned_box[1])

        is_inside_per_dim = []
        for d in range(dim):
            numerical_error = 0.03
            error_margin_new = ((max_corner_new_box[d] - min_corner_new_box[d]) * numerical_error)

            # With Epsilon
            is_inside = (min_corner_learned_box[d] <= (min_corner_new_box[d] + error_margin_new)) and (max_corner_learned_box[d] >= (max_corner_new_box[d] - error_margin_new))
            # Without Epsilon
            #is_inside = (min_corner_learned_box[d] <= min_corner_new_box[d]) and (max_corner_learned_box[d] >= max_corner_new_box[d])

            is_inside_per_dim.append(is_inside)

        if all(is_inside_per_dim) == True:
            new_boxes_inside_learned_boxes[new_box_class].append(learned_box_class)

        # Another way to calculate containment (based on ratio of 'product of intersection').
        """
        product_of_a = 1
        product_of_b = 1
        product_of_intersection = 1
        for d in range(dim):
            left_most_corner_of_intersection = max(min_corner_learned_box[d], min_corner_new_box[d])
            right_most_corner_of_intersection = min(max_corner_learned_box[d], max_corner_new_box[d])

            intersection = (left_most_corner_of_intersection <= right_most_corner_of_intersection) * (
                    right_most_corner_of_intersection - left_most_corner_of_intersection)
            product_of_intersection *= intersection

            size_of_b = max_corner_new_box[d] - min_corner_new_box[d]
            product_of_b *= (size_of_b)  # j

        if product_of_b and product_of_intersection:
            intersection_ratio = (product_of_intersection / product_of_b)
            if (product_of_intersection / product_of_b) == 1:  # > 0.80: #== 1: # j is inside i, and i contains j > 0.90: #
                new_boxes_inside_learned_boxes[new_box_class].append(learned_box_class)
        """
main_classes = ["CHEBI:{id}".format(id=item) for item in main_labels]

ancesstors_result = []
for k, true_class_ancestors in new_classes_subclass_of_base_classes.items():
    inferred_ancesstors = new_boxes_inside_learned_boxes[k]

    true_positive_count = 0
    false_positive_count = 0
    for item in inferred_ancesstors:
        if item in [x for x in true_class_ancestors if x in main_classes]:
            true_positive_count += 1
        else:
            false_positive_count += 1

    false_negative_count = len([x for x in true_class_ancestors if x not in inferred_ancesstors])#len(true_class_ancestors) - true_positive_count

    ground_truth_negatives = [x for x in main_classes if x not in true_class_ancestors]
    inferred_negatives = [x for x in main_classes if x not in inferred_ancesstors]

    true_negatives = [x for x in ground_truth_negatives if x in inferred_negatives]
    true_negative_count = len(true_negatives)

    combined = inferred_negatives + inferred_ancesstors

    all_inferred = true_positive_count + false_positive_count + false_negative_count + true_negative_count

    recall = 0.0
    if (true_positive_count) != 0: # Prevent division by zero
        recall = true_positive_count / (true_positive_count + false_negative_count)

    precision = 0.0
    if (true_positive_count) != 0: # Prevent division by zero
        precision = true_positive_count / (true_positive_count + false_positive_count)

    f1 = 0.0
    if (true_positive_count != 0):
        f1 = (2 * precision * recall) / (precision + recall)

    include = True
    if ((len(true_class_ancestors) == 0) and (len(inferred_ancesstors) == 0)):
        include = False


    if include:
        ancesstors_result.append(
            [k, len(true_class_ancestors), true_positive_count, false_positive_count, false_negative_count,
             true_negative_count, recall, precision, f1])

ansectors_results_df = pd.DataFrame(ancesstors_result,
                                 columns=['Label', 'Number_of_Ancesstors', 'TP', 'FP','FN', 'TN',
                                          'Recall', 'Precision', 'f1'
                                          ])

#print(ansectors_results_df.to_string())
#print(ansectors_results_df['Number_of_Ancesstors'].sum())

with open('./ansectors_results_df_new.pkl', 'wb') as ignore_no_parents_df_file:
    pickle.dump(ansectors_results_df, ignore_no_parents_df_file)
ignore_no_parents_df_file.close()

print('Ansectors Precision: ', ansectors_results_df['Precision'].mean())
print('Ansectors Recall: ', ansectors_results_df['Recall'].mean())
print('Ansectors F1: ', ansectors_results_df['f1'].mean())

total_ancesstors = ansectors_results_df['Number_of_Ancesstors'].sum()
weighted_ansectors_precision_sum = 0
weighted_ansectors_recall_sum = 0
weighted_ansectors_f1_sum = 0

for index, row in ansectors_results_df.iterrows():
    w_i = row['Number_of_Ancesstors']
    weighted_ansectors_precision_sum += (row['Precision'] * w_i)
    weighted_ansectors_recall_sum += (row['Recall'] * w_i)

weighted_precision_ansectors = weighted_ansectors_precision_sum / total_ancesstors
weighted_recall_ansectors = weighted_ansectors_recall_sum / total_ancesstors
Weighted_ansectors_F1 = (2 * weighted_precision_ansectors * weighted_recall_ansectors) / (
            weighted_precision_ansectors + weighted_recall_ansectors)

print('weighted precision ansectors: ', weighted_precision_ansectors)
print('weighted recall ansectors: ', weighted_recall_ansectors)
print('weighted F1 ansectors: ', Weighted_ansectors_F1)

a_tp = ansectors_results_df["TP"].sum()
a_fp = ansectors_results_df["FP"].sum()
a_fn = ansectors_results_df["FN"].sum()
a_tn = ansectors_results_df["TN"].sum()

print("TP: ", a_tp, "FP: ", a_fp, "FN: ", a_fn)
a_precision = a_tp / (a_tp + a_fp)
print("Micro Precision: ", a_precision)
a_recall = a_tp / (a_tp + a_fn)
print("Micro Recall: ", a_recall)
f1 = (a_tp) / (a_tp + (0.5 * (a_fp + a_fn)))
print("Micro F1: ", f1)