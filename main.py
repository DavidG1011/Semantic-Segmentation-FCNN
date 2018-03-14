import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import cv2
import scipy.misc
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


################### Tunable Parameters ##################################################

# Toggle if training or inferencing.
training = True

# Toggle if inferencing a video.
video = False

# Kernel regularizer value for nn layers.
regularizer = 1e-3

# Kernel initializer value for nn layers. Defines how random starting weights are set.
initializer = 0.01

# Desired epoch amount.
epochs = 50;

# Desired batch size.
batch_size = 10;

#########################################################################################



def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path) 
    graph = tf.get_default_graph()
    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    layer7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, (1,1), padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
                                   kernel_initializer= tf.random_normal_initializer(stddev=initializer))

    layer7_transpose = tf.layers.conv2d_transpose(layer7_conv, num_classes, 4, (2,2), padding='same', 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(regularizer),
                                               kernel_initializer= tf.random_normal_initializer(stddev=initializer))

    scale_4 = tf.multiply(vgg_layer4_out, 0.0001)


    layer4_conv = tf.layers.conv2d(scale_4, num_classes, 1, (1,1), padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
                                   kernel_initializer= tf.random_normal_initializer(stddev=initializer))

    skip1 = tf.add(layer7_transpose, layer4_conv)


    scale_3 = tf.multiply(vgg_layer3_out, 0.01)


    layer3_conv = tf.layers.conv2d(scale_3, num_classes, 1, (1,1), padding='same',
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
                                   kernel_initializer= tf.random_normal_initializer(stddev=initializer))

    skip1_transpose = tf.layers.conv2d_transpose(skip1, num_classes, 4, (2,2), padding='same', 
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
                                              kernel_initializer= tf.random_normal_initializer(stddev=initializer))

    skip2 = tf.add(layer3_conv, skip1_transpose)


    final = tf.layers.conv2d_transpose(skip2, num_classes, 16, (8,8), padding='same', 
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(regularizer),
                                       kernel_initializer= tf.random_normal_initializer(stddev=initializer))

    return final


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                                       (logits= nn_last_layer, labels= correct_label))

    #cross_entropy_loss = tf.add(cross_entropy_loss, tf.losses.get_regularization_loss())

    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)

    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    sess.run(tf.global_variables_initializer())

    for n in range(epochs):
        print ("Epoch", n)
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={input_image: image,
                                          correct_label: label,
                                          keep_prob: 0.5,
                                          learning_rate: 0.0009
                                         })
            print (loss)

    print ("Done.")

tests.test_train_nn(train_nn)


def video_pipeline(input_vid, runs_dir, sess, image_shape, 
                   logits, keep_prob, image_pl):

    capture = cv2.VideoCapture(input_vid)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(runs_dir + '/output.mp4',fourcc, fps, (image_shape[1],image_shape[0]))
    count = 0

    while (True):

        read_correct, frame =  capture.read()

        if read_correct:
            image = scipy.misc.imresize(frame, image_shape)
            im_softmax = sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
            im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
            segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)

            mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
            mask = scipy.misc.toimage(mask, mode="RGBA")


            street_im = scipy.misc.toimage(image)
            street_im.paste(mask, box=None, mask=mask)

            out.write(np.array(street_im))

            print ("Frame count: ", count)
        
            count = count + 1

        else:

            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_dir = './data/model/model.ckpt'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # epochs = 50;
        # batch_size = 10;

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes])
        learning_rate = tf.placeholder(tf.float32)

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        tf_saver = tf.train.Saver(max_to_keep=5)

        if training:

            train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)

            model_save = tf_saver.save(sess, model_dir)

            print ("Model saved to: ", model_dir)

        else:

            tf_saver.restore(sess, model_dir)

            print ("Model restored from: ", model_dir) 


        if video:

            input_vid = "./videoinput.mp4" 

            video_pipeline(input_vid, runs_dir, sess, image_shape, logits, keep_prob, input_image)

        else:

            helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


if __name__ == '__main__':
    run()
