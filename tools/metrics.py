from __future__ import division

import numpy as np

def compute_precision_recall(scores, labels, num_gt):
    """

    :param scores:  A float numpy array representing detection score
    :param labels:  A boolean numpy array representing true/false positive labels
    :param num_gt: number of ground truth instances
    :return:  precision and recall
    """

    sorted_indices = np.argsort(scores)
    sorted_indices = sorted_indices[::-1]
    labels = labels.astype(int)
    true_positive_labels = labels[sorted_indices]
    false_positive_labels = 1 - true_positive_labels
    cum_true_positives = np.cumsum(true_positive_labels)
    cum_false_positives = np.cumsum(false_positive_labels)
    precision = cum_true_positives.astype(float) / (cum_true_positives + cum_false_positives)
    recall = cum_true_positives.astype(float) / num_gt
    return precision, recall


def compute_average_precision(precision, recall):
    """
    Precision is modified to ensure that it does not decrease as recall
    decrease.
    :param precision: A float [N, 1] numpy array of precisions
    :param recall: A float [N, 1] numpy array of recalls
    :return: The area under the precision recall curve. NaN if
      precision and recall are None.
    """

    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")

    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0,1]")
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("precision must be in the range of [0,1]")

    recall = np.concatenate([[0], recall, [1]])

    precision = np.concatenate([[0], precision, [0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
    """

    :param num_gt_imgs_per_class: 1D array, representing number of images containing
        at least one object instance of a particular class
    :param num_images_correctly_detected_per_class: 1D array, representing number of
        images that are correctly detected at least one object instance of a
        particular class
    :return:
    """
    return np.where(
        num_gt_imgs_per_class == 0,
        np.nan,
        num_images_correctly_detected_per_class / num_gt_imgs_per_class)
















