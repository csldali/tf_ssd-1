# --*- coding:utf-8 -*--
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.contrib.slim as slim
from conf.ssd_config import *



class SSDNet(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, NUM_CHANNELS], name='x')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.preds_conf = []
        self.preds_loc = []

    def SSDHook(self, feature_map, hook_id):
        """

        :param feature_map:
        :param hook_id:  variable_scope unque string id
        :return:
        """

        with tf.variable_scope('ssd_hook_' + hook_id):
            net_conf = slim.conv2d(feature_map, NUM_PRED_CONF, [3, 3], activation_fn=None, scope='conv_conf')
            net_conf = tf.contrib.layers.flatten(net_conf)

            net_loc = slim.conv2d(feature_map, NUM_PRED_LOC, [3, 3], activation_fn=None, scope='conv_loc')
            net_loc = tf.contrib.layers.flatten(net_loc)

        return net_conf, net_loc


    def inference(self):
        """
        Alexnet as base Net
        :return:a dictionary of {tensor_name, tensor_reference}
        """
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': True}, \
                            weights_regularizer=slim.l2_regularizer(scale=REG_SCALE)):
            net = slim.conv2d(self.x, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')

            net_conf, net_loc = self.SSDHook(net, 'conv2')
            self.preds_conf.append(net_conf)
            self.preds_loc.append(net_loc)

            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')

            # The following layers added for SSD
            net = slim.conv2d(net, 1024, [3, 3], scope='conv6')
            net = slim.conv2d(net, 1024, [1, 1], scope='conv7')

            net_conf, net_loc = self.SSDHook(net, 'conv7')
            self.preds_conf.append(net_conf)
            self.preds_loc.append(net_loc)

            net = slim.conv2d(net, 256, [1, 1], scope='conv8')
            net = slim.conv2d(net, 512, [3, 3], 2, scope='conv8_2')

            net_conf, net_loc = self.SSDHook(net, 'conv8_2')
            self.preds_conf.append(net_conf)
            self.preds_loc.append(net_loc)

            net = slim.conv2d(net, 128, [1, 1], scope='conv9')
            net = slim.conv2d(net, 256, [3, 3], 2, scope='conv9_2')

            net_conf, net_loc = self.SSDHook(net, 'conv9_2')
            self.preds_conf.append(net_conf)
            self.preds_loc.append(net_loc)
        # Concatenate all preds together into 1 vector, for both classification and localization predictions

        final_pred_conf = tf.concat(self.preds_conf, 1)
        final_pred_loc = tf.concat(self.preds_loc, 1)

        ret_dict = {
            'x': self.x,
            'y_pred_conf': final_pred_conf,
            'y_pred_loc': final_pred_loc,
            'is_training': self.is_training
        }
        return ret_dict


    def cal_loss(self, y_pred_conf, y_pred_loc):
        """
        :param y_pred_conf: [batch_size, num_feature_map_cells * num_default_boxes * num_classes]
        :param y_pred_loc:  [batch_size, num_feature_map_cells * num_default_boxes *4]
        :return:  loss of model
        """

        num_total_preds = 0
        for fm_size in FM_SIZES:
            num_total_preds += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
        num_total_preds_conf = num_total_preds * NUM_CLASSES
        num_total_preds_loc = num_total_preds * 4

        self.y_true_conf = tf.placeholder(tf.int32, [None, num_total_preds], name='y_true_conf')
        self.y_true_loc = tf.placeholder(tf.float32, [None, num_total_preds_loc], name='y_true_loc')
        self.conf_loss_mask = tf.placeholder(tf.float32, [None, num_total_preds], name='conf_loss_mask')

        # confidence loss
        logits = tf.reshape(y_pred_conf, [-1, num_total_preds, NUM_CLASSES])
        conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_true_conf)

        conf_loss = self.conf_loss_mask * conf_loss  ## "zero-out" the loss for don't-care negatives
        conf_loss = tf.reduce_sum(conf_loss)


        # localization loss(smooth L1 loss)


        ##num_total_preds += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES * 4
        diff = self.y_true_loc - y_pred_loc
        loc_loss_l2 = 0.5 * (diff ** 2)
        loc_loss_l1 = tf.abs(diff) - 0.5
        smooth_l1_condition = tf.less(tf.abs(diff), 1.0)

        ## in tensorflow 1.0 replace the tf.select by tf.where
        loc_loss = tf.where(smooth_l1_condition, loc_loss_l2, loc_loss_l1)
        loc_loss_mask = tf.minimum(self.y_true_conf, 1)  # have non-zero localization loss only where we have matching ground-truth box
        loc_loss_mask = tf.to_float(loc_loss_mask)
        loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)  # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
        loc_loss_mask = tf.reshape(loc_loss_mask, [-1, num_total_preds_loc])  # removing the inner-most dimension of above
        loc_loss = loc_loss_mask * loc_loss
        loc_loss = tf.reduce_sum(loc_loss)

        # Weighted average of confidence loss and localization loss
        # Also add regularization loss
        loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss + tf.reduce_sum(slim.losses.get_regularization_losses())

        optimizer = OPT.minimize(loss)

        # Class probabilities and predictions
        probs_all = tf.nn.softmax(logits)
        probs, preds_conf = tf.nn.top_k(probs_all)  # take top-1 probability, and the index is the predicted class
        probs = tf.reshape(probs, [-1, num_total_preds])
        preds_conf = tf.reshape(preds_conf, [-1, num_total_preds])

        ret_dict = {

            'y_true_conf': self.y_true_conf,
            'y_true_loc': self.y_true_loc,
            'conf_loss_mask': self.conf_loss_mask,
            'conf_loss': conf_loss,
            'loc_loss': loc_loss,
            'loss': loss,
            'probs': probs,
            'preds_conf': preds_conf,
            'preds_loc': y_pred_loc,
            'optimizer': optimizer,
        }

        return ret_dict





