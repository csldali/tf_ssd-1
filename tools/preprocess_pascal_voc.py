from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tools.data_utils import *
import numpy as np
import pickle
import os
from PIL import Image
from lxml import etree
import logging
import io
from conf.ssd_config import *
from tools.utils import calc_iou
import tensorflow as tf
# Script config
RESIZE_IMAGE = True  # resize the images and write to 'resized_images/'
GRAYSCALE = False  # convert image to grayscale? this option is only valid if RESIZE_IMAGE==True (FIXME)
TARGET_W, TARGET_H = 400, 260


flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('set', 'test', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('year', 'VOC2007', 'Desired challenge year.')
flags.DEFINE_string('output_path_image', '', 'Path to resize image')
flags.DEFINE_string('output_path_pickle','', 'Path to pickled data')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test']
YEARS = ['VOC2007', 'VOC2012', 'merged']

def create_gt_boxes(data, dataset_directory, label_map_dict, output_dir, image_subdirectory='JPEGImages'):
    img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    print(full_path)

    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    orig_w = int(data['size']['width'])
    orig_h = int(data['size']['height'])

    image_file = data['filename']

    if GRAYSCALE:
        image = image.convert('L')

    image = image.resize((TARGET_W, TARGET_H), Image.LANCZOS)  # high-quality downsampling filter
    resized_dir = output_dir+'resized_images_%dx%d/' % (TARGET_W, TARGET_H)
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)

    image.save(os.path.join(output_dir, resized_dir, image_file))
    # Rescale box coordinates
    x_scale = TARGET_W / orig_w
    y_scale = TARGET_H / orig_h

    box_coords_list = []
    class_list = []
    ## left down  (x_min, y_max) right up(x_max, y_min) pascal

    ## (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner  (xmin, ymin)(xmax,ymax)


    for obj in data['object']:
        x_1 = float(obj['bndbox']['xmin'])
        y_1 = float(obj['bndbox']['ymin'])
        x_2 = float(obj['bndbox']['xmax'])
        y_2 = float(obj['bndbox']['ymax'])
        box_coords = np.array([x_1, y_1, x_2, y_2])

        ulc_x, ulc_y, lrc_x, lrc_y = box_coords
        new_box_coords = (ulc_x * x_scale, ulc_y * y_scale, lrc_x * x_scale, lrc_y * y_scale)
        new_box_coords = [round(x) for x in new_box_coords]
        box_coords = np.array(new_box_coords)
        box_coords_list.append(box_coords)
        class_label = label_map_dict[obj['name']]
        class_list.append(class_label)
    the_list = []
    for i in range(len(box_coords_list)):
        d = {
            'class': class_list[i],
            'box_coords': box_coords_list[i]
             }
        the_list.append(d)

    return the_list


def get_label_map_dict(classLabel):
    class_map = {}
    with open(classLabel, 'r') as f:
        for line in f:
            line = line.strip()
            integer_label, class_name = line.split(',')
            class_map[class_name] = int(integer_label)

    return class_map

def find_gt_boxes(data_raw, image_file):
    """
    Given (global) feature map sizes, and single training example, find all default boxes that exceed Jaccard overlap threshold
    :param data_raw:
    :param image_file:
    :return: y_true array that flags the matching default boxes with class ID (-1 means nothing there)
    """

    # pre-process ground true data
    data = data_raw[image_file]


    class_labels = []
    box_coords = []   # relative coordinates
    for obj in data:
        class_label = obj['class']
        class_labels.append(class_label)

        # calculate relative coordinates
        # (x1, y1, x2, y2), where 1 denotes upper left corner, 2 denotes lower right corner

        abs_box_coords = obj['box_coords']
        scale = np.array([IMG_W, IMG_H, IMG_W, IMG_H])
        box_coord = np.array(abs_box_coords) / scale
        box_coords.append(box_coord)

    y_true_len = 0
    for fm_size in FM_SIZES:
        y_true_len += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
    y_true_conf = np.zeros(y_true_len)
    y_true_loc = np.zeros(y_true_len * 4)



    # For each GT box, for each feature map, for each feature map cell, for each default box:
    # 1) Calculate the Jaccard overlap (IOU) and annotate the class label
    # 2) Count how many box matches we got
    # 3) If we got a match, calculate normalized box coordinates and updte y_true_loc

    match_counter = 0
    for i, gt_box_coords in enumerate(box_coords):
        y_true_idx = 0
        # for fm_idx, fm_size in enumerate(FM_SIZES):
        for fm_size in FM_SIZES:
            fm_h, fm_w = fm_size  # feature map height and width
            for row in range(fm_h):
                for col in range(fm_w):
                    for db in DEFAULT_BOXES:
                        # Calculate relative box coordinates for this default box
                        x1_offset, y1_offset, x2_offset, y2_offset = db
                        abs_db_box_coords = np.array([
                            max(0, col + x1_offset),
                            max(0, row + y1_offset),
                            min(fm_w, col + 1 + x2_offset),
                            min(fm_h, row + 1 + y2_offset)
                        ])
                        scale = np.array([fm_w, fm_h, fm_w, fm_h])
                        db_box_coords = abs_db_box_coords / scale

                        # Calculate Jaccard overlap (i.e. Intersection Over Union, IOU) of GT box and default box
                        iou = calc_iou(gt_box_coords, db_box_coords)

                        # If box matches, i.e. IOU threshold met
                        if iou >= IOU_THRESH:
                            # Update y_true_conf to reflect we found a match, and increment match_counter
                            y_true_conf[y_true_idx] = class_labels[i]
                            match_counter += 1

                            # Calculate normalized box coordinates and update y_true_loc
                            abs_box_center = np.array(
                                [col + 0.5, row + 0.5])  # absolute coordinates of center of feature map cell
                            abs_gt_box_coords = gt_box_coords * scale  # absolute ground truth box coordinates (in feature map grid)
                            norm_box_coords = abs_gt_box_coords - np.concatenate((abs_box_center, abs_box_center))
                            y_true_loc[y_true_idx * 4: y_true_idx * 4 + 4] = norm_box_coords

                        y_true_idx += 1
    print(y_true_conf)
    return y_true_conf, y_true_loc, match_counter


def main(_):

    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))
    if FLAGS.year not in YEARS:
        raise ValueError('year must be in : {}'.format(YEARS))

    if FLAGS.year != 'merged':
        years = [FLAGS.year]
    assert FLAGS.label_map_path, '`label_map_path` is missing.'
    assert FLAGS.output_path_image, '`output_path_image` is missing.'
    assert FLAGS.output_path_pickle, '`output_path_pickle` is missing.'

    label_map_dict = get_label_map_dict(FLAGS.label_map_path)
    data_dir = FLAGS.data_dir
    data_raw = {}
    for year in years:
        logging.info('Reading from PASCAL %s dataset.', year)
        examples_path = os.path.join(data_dir, year, 'ImageSets', 'Main',
                                     FLAGS.set + '.txt')
        annotations_dir = os.path.join(data_dir, year, FLAGS.annotations_dir)
        examples_list = read_examples_list(examples_path)
        for idx, example in enumerate(examples_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(examples_list))

            path = os.path.join(annotations_dir, example + '.xml')
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()

            xml = etree.fromstring(xml_str)
            data = recursive_parse_xml_to_dict(xml)['annotation']
            image_file = data['filename']
            data_dict = create_gt_boxes(data, FLAGS.data_dir, label_map_dict, FLAGS.output_path_image)
            data_raw[image_file] = data_dict

        data_prep ={}
        i = 0
        for image_file in data_raw.keys():
            i += 1
            if (i % 100 == 0):
                print (i)
            y_true_conf, y_true_loc, match_counter = find_gt_boxes(data_raw, image_file)

            if match_counter > 0:
                data_prep[image_file] = {'y_true_conf': y_true_conf, 'y_true_loc': y_true_loc}

        with open(FLAGS.output_path_pickle + 'data_raw_%dx%d.p' % (TARGET_W, TARGET_H), 'wb') as f:
            pickle.dump(data_prep, f)

if __name__ == '__main__':
    tf.app.run()

















