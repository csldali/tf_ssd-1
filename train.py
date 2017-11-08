import tensorflow as tf
from Solver.ssd_solver import ssdSolver
tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags

flags.DEFINE_string('data_path','','path to the prepared data')
flags.DEFINE_string('output_dir','','path to the output model')
flags.DEFINE_string('resized_image_dir','','path to the resized image dir')
FLAGS = flags.FLAGS

def main(_):
    assert FLAGS.data_path, '`data_path` is missing'
    assert FLAGS.output_dir, '`output_dir` is missing'
    assert FLAGS.resized_image_dir, '`resized_image_dir` is missing'
    train_data = FLAGS.data_path
    model_save_path = FLAGS.output_dir + '/model.ckpt'
    image_dir = FLAGS.resized_image_dir
    ssd_solver = ssdSolver()
    ssd_solver._train(train_data, image_dir, model_save_path)

if __name__ == '__main__':
    tf.app.run()



