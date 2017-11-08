from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
def calc_iou(box_a, box_b):
    """
    Calculate the Intersection Over Union of two boxes

    :param box_a: (x1, y1,x2,y2) denotes upper left corner, lower right corner
    :param box_b:
    :return: IOU value
    """

    x_overlap = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    y_overlap = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    intersection = x_overlap * y_overlap

    # Calculate union
    area_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_box_a + area_box_b - intersection

    iou = intersection / union
    return iou

