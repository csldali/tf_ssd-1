from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split
import math
import time
from conf.ssd_config import *
import pickle
from ssd.nets import SSDNet
from data.data_generate import DataGenerate

class ssdSolver(object):
    # AlexNet as base Model
    def construct_model(self):
        ssd = SSDNet()
        model = ssd.inference()
        model_helper =ssd.cal_loss(model['y_pred_conf'], model['y_pred_loc'])
        ssd_model = {}
        for k in model.keys():
            ssd_model[k] = model[k]
        for k in model_helper.keys():
            ssd_model[k] = model_helper[k]
        return ssd_model


    def _train(self, data_path, image_dir,model_save_path):
        """
        Load training and test data
        Run training process
        plot train/validation losses
        report test loss
        save model
        :return: saved model
        """
        # Load training and test data

        with open(data_path, mode='rb') as f:
            train = pickle.load(f)

        # Format the data
        X_train = []
        y_train_conf = []
        y_train_loc = []
        for image_file in train.keys():
            X_train.append(image_file)
            y_train_conf.append(train[image_file]['y_true_conf'])
            y_train_loc.append(train[image_file]['y_true_loc'])

        # Train/validation split
        X_train, X_valid, y_train_conf, y_valid_conf, y_train_loc, y_valid_loc = train_test_split(X_train, y_train_conf, y_train_loc, test_size=VALIDATION_SIZE, random_state=1)

        # Launch the graph
        with tf.Graph().as_default(), tf.Session() as sess:
            self.model = self.construct_model()
            x = self.model['x']
            y_true_conf = self.model['y_true_conf']
            y_true_loc = self.model['y_true_loc']
            conf_loss_mask = self.model['conf_loss_mask']
            is_training = self.model['is_training']
            optimizer = self.model['optimizer']
            reported_loss = self.model['loss']

            # Training process
            # TF saver to save/restore trained model
            #

            if RESUME:
                print('Restoring previously trained model at %s' % model_save_path)
                saver = tf.train.Saver()
                saver.restore(sess, model_save_path)

                # Restore previous loss history
                with open('loss_history.p', 'rb') as f:
                    loss_history = pickle.load(f)
            else:
                print('Training model from scratch')
                sess.run(tf.global_variables_initializer())
                # For book-keeping, keep track of training and validation loss over epochs, like such:
                # [(train_acc_epoch1, valid_acc_epoch1), (train_acc_epoch2, valid_acc_epoch2), ...]
                loss_history = []

            last_time = time.time()
            train_start_time = time.time()

            for epoch in range(NUM_EPOCH):
                dataGenerate_train = DataGenerate(X_train, y_train_conf, y_train_loc, BATCH_SIZE, image_dir)
                train_gen = dataGenerate_train.next_batch()
                # print(len(X_train))
                num_batches_train = math.ceil(len(X_train)/ BATCH_SIZE)
                losses = []  # list of loss values for book-keeping
                for _ in range(num_batches_train):
                    images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(train_gen)

                    # Perform gradient update (i.e. training step) on current batch
                    _, loss = sess.run([optimizer, reported_loss], feed_dict={
                        # _, loss, loc_loss_dbg, loc_loss_mask, loc_loss = sess.run([optimizer, reported_loss, model['loc_loss_dbg'], model['loc_loss_mask'], model['loc_loss']],feed_dict={  # DEBUG
                        x: images,
                        y_true_conf: y_true_conf_gen,
                        y_true_loc: y_true_loc_gen,
                        conf_loss_mask: conf_loss_mask_gen,
                        is_training: True
                    })

                    losses.append(loss)  # TODO: Need mAP metric instead of raw loss

                # A rough estimate of loss for this epoch (overweights the last batch)
                train_loss = np.mean(losses)

                dataGenerate_val = DataGenerate(X_valid, y_valid_conf, y_valid_loc, BATCH_SIZE,image_dir)
                valid_gen = dataGenerate_val.next_batch()
                num_batches_valid = math.ceil(len(X_valid) / BATCH_SIZE)
                losses = []
                for _ in range(num_batches_valid):
                    images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(valid_gen)

                    # Perform forward pass and calculate loss
                    loss = sess.run(reported_loss, feed_dict={
                        x: images,
                        y_true_conf: y_true_conf_gen,
                        y_true_loc: y_true_loc_gen,
                        conf_loss_mask: conf_loss_mask_gen,
                        is_training: False
                    })
                    losses.append(loss)
                valid_loss = np.mean(losses)

                # Record and report train/validation/test losses for this epoch
                loss_history.append((train_loss, valid_loss))

                # Print accuracy every epoch
                print('Epoch %d -- Train loss: %.4f, Validation loss: %.4f, Elapsed time: %.2f sec' % \
                      (epoch + 1, train_loss, valid_loss, time.time() - last_time))
                last_time = time.time()

            total_time = time.time() - train_start_time
            print('Total elapsed time: %d min %d sec' % (total_time / 60, total_time % 60))

            test_loss = 0.  # TODO: Add test set
            '''
            # After training is complete, evaluate accuracy on test set
            print('Calculating test accuracy...')
            test_gen = next_batch(X_test, y_test, BATCH_SIZE)
            test_size = X_test.shape[0]
            test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE, accuracy, x, y, keep_prob, sess)
            print('Test acc.: %.4f' % (test_acc,))
            '''

            if SAVE_MODEL:
                # Save model to disk
                save_path = saver.save(sess, model_save_path)
                print('Trained model saved at: %s' % save_path)

                # Also save accuracy history
                print('Loss history saved at loss_history.p')
                with open('loss_history.p', 'wb') as f:
                    pickle.dump(loss_history, f)

                    # Return final test accuracy and accuracy_history
        return test_loss, loss_history







