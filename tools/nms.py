from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tools.utils import calc_iou
import numpy as np
import tensorflow as tf
from conf.ssd_config import *

def nms(y_pred_conf, y_pred_loc, prob):
    """
    Non-Maximum Suppression(NMS)

    :param y_pred_conf:  Class predictions, numpy array of shape( num_feature_map_cell * num_default_boxes
    :param y_pred_loc:  Bounding box coordinates, numpy array of shape(num_feature_map_cell * num_default_boxes*4)
    :param prob: class probabilities, numpy array of shape( num_feature_map_cell * num_default_boxes)
    :return: a list of box coordinates post-NMS, numpy array of boxes, with shape(num_boxes, 6 )  [x1, y1, x2, y2, class,probability]
    """

    # Keep track of boxes for each class
    class_boxes = {}  # class -> [(x1, y1, x2, y2, prob), (...), ...]
    with open('signnames.csv', 'r') as f:
        for line in f:
            cls, _ = line.split(',')
            class_boxes[float(cls)] = []

    # Go through all possible boxes and perform class-based greedy NMS (greedy based on class prediction confidence)
    y_idx = 0

    for fm_size in FM_SIZES:
        fm_h, fm_w = fm_size   # get feature map height and width
        for row in range(fm_h):
            for col in range(fm_w):
                for db in DEFAULT_BOXES:
                    if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0:
                        xc, yc = col + 0.5, row + 0.5
                        center_coords = np.array([xc, yc, xc, yc])
                        abs_box_coords = center_coords + y_pred_loc[y_idx * 4: y_idx * 4 + 4]

                        # Calculate predicted box coordinates in actual image
                        scale = np.array([IMG_W / fm_w, IMG_H / fm_h, IMG_W / fm_w, IMG_H / fm_h])
                        box_coords = abs_box_coords * scale
                        box_coords = [int(round(x)) for x in box_coords]

                        # Compare this box to all previous boxes of this class
                        cls = y_pred_conf[y_idx]
                        cls_prob = prob[y_idx]
                        box = (*box_coords, cls, cls_prob)
                        if len(class_boxes[cls]) == 0:
                            class_boxes[cls].append(box)
                        else:
                            suppressed = False  # did this box suppress other box(es)?
                            overlapped = False  # did this box overlap with other box(es)?
                            for other_box in class_boxes[cls]:
                                iou = calc_iou(box[:4], other_box[:4])
                                if iou > NMS_IOU_THRESH:
                                    overlapped = True
                                    # If current box has higher confidence than other box
                                    if box[5] > other_box[5]:
                                        class_boxes[cls].remove(other_box)
                                        suppressed = True
                            if suppressed or not overlapped:
                                class_boxes[cls].append(box)

                    y_idx += 1
                    # Gather all the pruned boxes and return them
        boxes = []
        for cls in class_boxes.keys():
            for class_box in class_boxes[cls]:
                boxes.append(class_box)
        boxes = np.array(boxes)

        return boxes


