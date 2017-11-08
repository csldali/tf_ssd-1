from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from PIL import Image
from conf.ssd_config import *
class DataGenerate(object):
    """
        X : List of image file names
        y_conf: List of ground-truth vectors for class labels
        y_loc: List of ground-truth vectors for localization
        batch_size: Batch size
    """
    def __init__(self, X, y_conf, y_loc, batch_size, image_path):
        self.X = X
        self.y_conf = y_conf
        self.y_loc = y_loc
        self.batch_size = batch_size
        self.image_path = image_path


    def next_batch(self):
        """

        :return: yeilds
        images : Batch numpy array representation of batch of images
        y_true_conf: Batch numpy array of ground-truth class labels
        y_true_loc: Batch numpy array of ground-truth localization
        conf_loss_mask: Loss mask for confidence loss
        """

        start_idx = 0
        while True:
            image_files = self.X[start_idx:start_idx + self.batch_size]
            y_true_conf = np.array(self.y_conf[start_idx:start_idx+self.batch_size])
            y_true_loc = np.array(self.y_loc[start_idx:start_idx+self.batch_size])

            images = []
            for image_file in image_files:
                image = Image.open(self.image_path + '/resized_images_%sx%s/%s' % (IMG_W, IMG_H, image_file))
                image = np.asarray(image)
                images.append(image)
            images = np.array(images, dtype='float32')

            # Grayscale images have array shape (H, W), but we want shape (H, W, 1)
            if NUM_CHANNELS == 1:
                images = np.expand_dims(images, axis=-1)

            # Normalize pixel values (scale them between -1 and 1)
            images = images / 127.5 - 1.

            # For y_true_conf, calculate how many negative examples we need to satisfy NEG_POS_RATIO
            num_pos = np.where(y_true_conf > 0)[0].shape[0]
            num_neg = NEG_POS_RATIO * num_pos
            y_true_conf_size = np.sum(y_true_conf.shape)

            # Create confidence loss mask to satisfy NEG_POS_RATIO
            if num_pos + num_neg < y_true_conf_size:
                conf_loss_mask = np.copy(y_true_conf)
                conf_loss_mask[np.where(conf_loss_mask > 0)] = 1.

                # Find all (i,j) tuples where y_true_conf[i][j] ==0

                zero_indices = np.where(conf_loss_mask == 0.)  # ([i1, i2, ...], [j1, j2, ...])
                zero_indices = np.transpose(zero_indices)  # [[i1, j1], [i2, j2], ...]

                # Randomly choose num_neg rows from zero_indices, w/o replacement
                chosen_zero_indices = zero_indices[np.random.choice(zero_indices.shape[0], int(num_neg), False)]

                for zero_idx in chosen_zero_indices:
                    i, j = zero_idx
                    conf_loss_mask[i][j] = 1.
            else:
                conf_loss_mask = np.ones_like(y_true_conf)

            yield (images, y_true_conf, y_true_loc, conf_loss_mask)

            start_idx += self.batch_size
            if start_idx >= len(self.X):
                start_idx = 0