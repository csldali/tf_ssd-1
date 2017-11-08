from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
from conf.ssd_config import *
from PIL import Image
from tools.nms import nms
from tools.vistual_utils import *

class inferenceSolver(object):

    def __init__(self, image, sess, mode, sign_map):
        """
        image: Numpy array representing a single RGB image
        model: Dict of tensor references
        sess: TensorFlow session reference
        """
        self.image = np.array(image)
        self.sess = sess
        self.sign_map = self.sign_map

    def run_inference(self):
        image_orig = np.copy(self.image)

        # Get relevant tensors
        x = self.model['x']
        is_training = self.model['is_training']
        preds_conf = self.model['preds_conf']
        preds_loc = self.model['preds_loc']
        probs = self.model['probs']

        image = Image.fromarray(self.image)
        orig_w, orig_h = image.size

        if NUM_CHANNELS == 1:
            image = image.convert('L')
            image = image.resize((IMG_W, IMG_H), Image.LANCZOS)  # high-quality downsampling filter
            image = np.asarray(image)

        images = np.array([image])  # create a "batch" of 1 image
        if NUM_CHANNELS == 1:
            images = np.expand_dims(images, axis=-1)  # need extra dimension of size 1 for grayscale
        # Perform object detection
        t0 = time.time()  # keep track of duration of object detection + NMS
        preds_conf_val, preds_loc_val, probs_val = self.sess.run([preds_conf, preds_loc, probs],
                                                            feed_dict={x: images, is_training: False})

        print('Inference took %.1f ms (%.2f fps)' % ((time.time() - t0) * 1000, 1 / (time.time() - t0)))

        # Gather class predictions and confidence values
        y_pred_conf = preds_conf_val[0]  # batch size of 1, so just take [0]
        y_pred_conf = y_pred_conf.astype('float32')
        prob = probs_val[0]

        # Gather localization predictions
        y_pred_loc = preds_loc_val[0]

        # Perform NMS
        boxes = nms(y_pred_conf, y_pred_loc, prob)
        print('Inference + NMS took %.1f ms (%.2f fps)' % ((time.time() - t0) * 1000, 1 / (time.time() - t0)))

        # Rescale boxes' coordinates back to original image's dimensions
        # Recall boxes = [[x1, y1, x2, y2, cls, cls_prob], [...], ...]
        scale = np.array([orig_w / IMG_W, orig_h / IMG_H, orig_w / IMG_W, orig_h / IMG_H])
        if len(boxes) > 0:
            boxes[:, :4] = boxes[:, :4] * scale

        # Draw and annotate boxes over original image, and return annotated image
        image = image_orig
        for box in boxes:
            # box_coords = [int(round(x)) for x in box[:4]]
            # cls = int(box[4])
            # cls_prob = box[5]
            # ymin, xmin, ymax, xmax
            draw_bounding_box_on_image_array(image,box[1], box[0], box[3], box[2])

        return image










