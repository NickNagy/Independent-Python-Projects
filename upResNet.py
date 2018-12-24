"""
The format of this file is influenced by jakeret's implementation of Unet
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from collections import OrderedDict
import os
from image_util import ImageDataProvider as generator
import shutil
from skimage.transform import resize
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# layers & variable definitions
def weight_variable(shape, stddev, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name)


def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        return tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID'), b), keep_prob_)

def deconv2d(x, W, midpoint, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], tf.to_int32(tf.to_float(x_shape[1]) * 2),
                                 tf.to_int32(tf.to_float(x_shape[2]) * 2), x_shape[3]]) # // 2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID',
                                      name="conv2d_transpose")


def get_image_summary(img, idx=0):
    """
    Code from jakeret unet implementation
    """

    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255

    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V


def crop_to_shape(data, shape):
    """
    Code from jakeret unet implementation
    """
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    if offset0 == 0 or offset1 == 0:
        return data
    return data[:, offset0:(-offset0), offset1:(-offset1)]


# no pooling layers
def create_network(x, keep_prob, padding=False, resolution=3, features_root=16, channels=3, filter_size=3, deconv_size=2,
                    layers_per_transpose = 2, summaries=True):
    """
    :param x: input tensor, shape [?,width,height,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in input image
    :param padding: boolean, True if inputs are padded before convolution
    :param resolution: corresponds to how large the output should be relative to the input
    :param features_root: number of features in the first layer
    :param filter_size: size of convolution filter
    :param deconv_size: size of deconv strides
    :param summaries: flag if summaries should be created
    """

    logging.info("Resolution x{resolution}, features {features}, filter size {filter_size}x{filter_size}".format(
        resolution=resolution,
        features=features_root,
        filter_size=filter_size))

    with tf.name_scope("preprocessing"):
        width = tf.shape(x)[1]
        height = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, width, height, channels]))
        in_node = x_image

    weights = []
    biases = []
    convs = []
    convsDict = OrderedDict()
    deconvsDict = OrderedDict()

    size = 1
    which_conv = 0
    which_up_conv = 0
    stddev = 0
    out_features = features_root
    in_features = channels
    while size < resolution:
        # Convolutions...
        with tf.name_scope("Conv{}".format(str(which_conv))):
            stddev = np.sqrt(2 / (filter_size ** 2 * out_features))
            for layer in range (0, layers_per_transpose):
                out_features = features_root #size * 2 ** layer * features_root
                if which_conv == 0:
                    in_features = channels
                else:
                    in_features = out_features
                w = weight_variable([filter_size, filter_size, in_features, out_features], stddev, name="w" + str(layer))
                b = bias_variable([out_features], name="b" + str(layer))
                if padding:
                    in_node = tf.pad(in_node, paddings=[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
                conv = conv2d(in_node, w, b, keep_prob)
                convsDict[which_conv] = tf.nn.relu(conv)
                in_node = convsDict[which_conv]
                weights.append(w)
                biases.append(b)
                convs.append(conv)
                which_conv += 1

        # Upscalings...
        with tf.name_scope("Up_Conv{}".format(which_up_conv)):
            stddev = np.sqrt(2 / (filter_size ** 2 * out_features))
            midpoint = which_up_conv % 2 == 0
            wd = weight_variable([deconv_size, deconv_size, out_features, out_features], stddev, name="wd")
            bd = bias_variable([out_features], name="bd")
            deconv = tf.nn.relu(deconv2d(in_node, wd, midpoint, deconv_size) + bd)
            deconvsDict[which_up_conv] = deconv
            in_node = deconv
            which_up_conv += 1
            size += 1

    # output
    with tf.name_scope("output"):
        weight = weight_variable([1, 1, features_root, channels], stddev)
        bias = bias_variable([channels], name="bias")
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
        output = tf.nn.relu(conv)
        convsDict["out"] = output

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c) in enumerate(convs):
                tf.summary.image('summary_conv_%02d' % i, get_image_summary(c))
            tf.summary.image('summary_output', get_image_summary(convsDict["out"]))
            for k in deconvsDict.keys():
                tf.summary.image('summary_deconv_%02d' % k, get_image_summary(deconvsDict[k]))

    variables = []
    for w in weights:
        variables.append(w)
    for b in biases:
        variables.append(b)

    return output, variables


class upResNet(object):
    def __init__(self, padding, channels=3, resolution=2, layers_per_transpose=2, **kwargs):
        tf.reset_default_graph()

        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None, None, None, channels], name='x')
        self.y = tf.placeholder("float", shape=[None, None, None, channels], name='y')
        # weighted map would be a map of detected edges
        self.w = tf.placeholder("float", shape=[None, None, None, 1], name='w')
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")

        logits, self.variables = create_network(self.x, keep_prob=0.75, channels=channels, resolution=resolution,
                                                padding=padding, layers_per_transpose=layers_per_transpose, **kwargs)

        self.cost = self._get_cost(logits)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.logits = logits

        with tf.name_scope("results"):
            self.predicter = logits  # may change?
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.to_int32(self.logits), tf.to_int32(self.y)), tf.float32))

    def _get_cost(self, logits):
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, channels])  # channels may not reach this scope
            flat_labels = tf.reshape(self.y, [-1, channels])
            flat_map = tf.reshape(self.w, [-1, 1])

            """
            Current plan for loss function is a square of the pixelwise difference between X and Y
            """
            loss_map = tf.math.squared_difference(flat_logits, flat_labels)
            loss = tf.reduce_mean(tf.multiply(loss_map, flat_map))

            return loss

    def predict(self, path, x_test):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            lgts = sess.run(init)
            self.restore(sess, path)
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1] * resolution, x_test.shape[2] * resolution, channels))
            w_dummy = np.empty((x_test.shape[0], x_test.shape[1] * resolution, x_test.shape[2] * resolution, channels))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.w: w_dummy})
        return prediction

    def save(self, sess, path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, path)
        return save_path

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, path)
        logging.info("Model restored from file: %s" % path)


class Trainer(object):
    def __init__(self, net, batch_size=1, validation_batch_size=1, opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self, global_step):
        learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
        self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, beta1=0.9,
                                           epsilon=1e-8, **self.opt_kwargs).minimize(self.net.cost,
                                                                                     global_step=global_step)
        return optimizer

    def _initialize(self, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")
        # self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
        #                                        name="norm_gradients")
        # if self.net.summaries and self.net.norm_grads:
        #     tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)

        tf.summary.image('prediction', get_image_summary(self.net.logits))

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, training_data_provider, validation_data_provider, testing_data_provider, restore_path,
              output_path, total_validation_data, training_iters=10, epochs=100,
              dropout=0.75, display_step=1, include_map=True, restore=False,
              write_graph=False, test_after=False, prediction_path='prediction'):

        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        epoch_offset = 0
        try:
            epoch_file = open(output_path + "\\last_epoch.txt", "r")
            if restore:
                try:
                    epoch_offset = int(epoch_file.readline())
                except(ValueError):
                    epoch_offset = 0
            epoch_file.close()
        except(FileNotFoundError):
            print("Note: last_epoch.txt was not found. Assumed starting @ epoch 0")

        init = self._initialize(output_path, restore, prediction_path)

        validation_avg_losses = []
        training_avg_losses = []
        training_accuracies = []
        validation_accuracies = []

        try:
            training_file = open(output_path + "\\training_data.txt", "r")
            validation_file = open(output_path + "\\validation_data.txt", "r")
            if restore:
                try:
                    # TODO: better way?
                    training_avg_losses = [float(i) for i in training_file.readline()[1:-2].split(', ')]
                    print(str(training_avg_losses))
                    training_accuracies = [float(i) for i in training_file.readline()[1:-1].split(', ')]
                    validation_avg_losses = [float(i) for i in validation_file.readline()[1:-2].split(', ')]
                    validation_accuracies = [float(i) for i in validation_file.readline()[1:-1].split(', ')]
                except(ValueError):
                    print("No prior training or validation data exists. Assuming new model")
            training_file.close()
            validation_file.close()
        except(FileNotFoundError):
            print("No prior training or validation data exists. Assuming new model")

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(restore_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            pred_shape, validation_avg_losses, validation_accuracies = self.validate(sess, total_validation_data,
                                                                                    validation_data_provider,
                                                                                    include_map,
                                                                                    training_avg_losses,
                                                                                    training_accuracies,
                                                                                    validation_avg_losses,
                                                                                    validation_accuracies, "_init")

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            for epoch in range(epochs):
                total_loss = 0
                total_acc = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y, batch_w = training_data_provider(self.batch_size)
                    if not include_map:
                        batch_w = np.ones(batch_w.shape)
                    y_cropped = crop_to_shape(batch_y, pred_shape)
                    w_cropped = crop_to_shape(batch_w, pred_shape)
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: y_cropped,
                                   self.net.w: w_cropped,
                                   self.net.keep_prob: dropout})
                    if step % display_step == 0:
                        acc = self.output_minibatch_stats(sess, summary_writer, step, batch_x, y_cropped, w_cropped)
                    total_loss += loss
                    total_acc += acc
                training_avg_losses.append(total_loss / training_iters)
                training_accuracies.append(total_acc / training_iters)
                true_epoch = epoch + epoch_offset
                self.output_epoch_stats(true_epoch, total_loss, training_iters, lr)
                _, validation_avg_losses, validation_accuracies = self.validate(sess, total_validation_data,
                                                                             validation_data_provider, include_map,
                                                                             training_avg_losses,
                                                                             training_accuracies,
                                                                             validation_avg_losses,
                                                                             validation_accuracies,
                                                                             name="epoch_%s" % true_epoch)
                epoch_file = open(output_path + "\\last_epoch.txt", "w")
                epoch_file.write(str(true_epoch + 1))
                epoch_file.close()
                # TODO: find more efficient way
                training_file = open(output_path + "\\training_data.txt", "w")
                training_file.write(str(training_avg_losses) + "\n")
                training_file.write(str(training_accuracies))
                training_file.close()
                validation_file = open(output_path + "\\validation_data.txt", "w")
                validation_file.write(str(validation_avg_losses) + "\n")
                validation_file.write(str(validation_accuracies))
                validation_file.close()
                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished")

            if test_after:
                for i in range(0, testing_data_provider._get_number_of_files()):
                    print(str(i))
                    test_x, test_y, test_w = training_data_provider(1)
                    fig,ax = plt.subplots(1,3,sharex=True,sharey=True)
                    ax[0].imshow(test_y[0].astype('uint8'), aspect="auto")
                    ax[0].set_title("Truth")
                    ax[1].imshow(resize(test_x[0].astype('uint8'), (pred_shape[1],pred_shape[2])), aspect="auto")
                    ax[1].set_title("skimage.transform.resize")
                    prediction = sess.run(self.net.predicter, feed_dict={self.net.x: test_x,
                                                                         self.net.y: test_y,
                                                                         self.net.w: test_w,
                                                                         self.net.keep_prob: 1})
                    ax[2].imshow(prediction[0].astype('uint8'), aspect="auto")
                    ax[2].set_title("Prediction")
                    plt.savefig("%s/%s.jpg" % (self.prediction_path, i))
            return save_path

    def validate(self, sess, total_validation_data, validation_data_provider, include_map,
                 training_average_losses, training_accuracies, validation_average_losses,
                 validation_accuracies, name):
        total_validation_loss = 0
        total_validation_acc = 0
        validation_iters = int(total_validation_data / self.validation_batch_size)
        last_batch_size = total_validation_data - (validation_iters * self.validation_batch_size)

        for i in range(0, validation_iters):
            test_x, test_y, test_w = validation_data_provider(self.validation_batch_size)
            if not include_map:
                test_w = np.ones
            if i == validation_iters - 1 and last_batch_size == 0:
                pred_shape, loss, accuracy = self.store_prediction(sess, test_x, test_y, test_w, name=name,
                                                                   training_losses=training_average_losses,
                                                                   training_accuracies=training_accuracies,
                                                                   validation_losses=validation_average_losses,
                                                                   validation_accuracies=validation_accuracies,
                                                                   save_img=1)
            else:
                _, loss, accuracy = self.store_prediction(sess, test_x, test_y, test_w)
            total_validation_loss += loss
            total_validation_acc += accuracy

        if last_batch_size != 0:
            test_x, test_y, test_w = validation_data_provider(last_batch_size)
            pred_shape, loss, accuracy = self.store_prediction(sess, test_x, test_y, test_w, name=name,
                                                               training_losses=training_average_losses,
                                                               training_accuracies=training_accuracies,
                                                               validation_losses=validation_average_losses,
                                                               validation_accuracies=validation_accuracies,
                                                               save_img=1)

        validation_average_losses.append(total_validation_loss / total_validation_data)
        validation_accuracies.append(total_validation_acc / total_validation_data)

        logging.info("Average validation loss= {:.4f}".format(total_validation_loss / total_validation_data))
        return pred_shape, validation_average_losses, validation_accuracies

    def store_prediction(self, sess, batch_x, batch_y, batch_w, name=None, training_losses=None,
                         training_accuracies=None, validation_losses=None,
                         validation_accuracies=None, save_img=0):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.w: batch_w,
                                                             self.net.keep_prob: 1})
        self.prediction = prediction

        pred_shape = prediction.shape
        y_cropped = crop_to_shape(batch_y, pred_shape)
        w_cropped = crop_to_shape(batch_w, pred_shape)

        loss, accuracy = sess.run((self.net.cost, self.net.accuracy), feed_dict={self.net.x: batch_x,
                                                                                 self.net.y: y_cropped,
                                                                                 self.net.w: w_cropped,
                                                                                 self.net.keep_prob: 1})
        logging.info("Validation loss= {:.4f}".format(loss))
        x_resize = resize(batch_x[0], (pred_shape[1],pred_shape[2]))

        # for logic checking purposes
        debug=False
        if debug:
            fig, ax = plt.subplots(2, 4, sharex=False, sharey=False)
            ax[0,0].imshow(batch_y[0], aspect="auto")
            ax[1,0].imshow(batch_y[0].astype('uint8'), aspect="auto")
            ax[0,1].imshow(batch_x[0], aspect="auto")
            ax[1,1].imshow(batch_x[0].astype('uint8'), aspect="auto")
            ax[0,2].imshow(x_resize, aspect="auto")
            ax[1,2].imshow(x_resize.astype('uint8'), aspect="auto")
            ax[0,3].imshow(prediction[0], aspect="auto")
            ax[1,3].imshow(prediction[0].astype('uint8'), aspect="auto")
            plt.show()

        if save_img:
            plt.rcParams.update({'font.size': 8})
            training_color = 'b'
            validation_color = 'r'
            moving_window_size = 25
            fig = plt.figure()
            # define axes
            grid_size = (3, 3)
            ax_y_img = plt.subplot2grid(grid_size, (0, 0), rowspan=1, colspan=1)
            ax_y_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_y_img.imshow(batch_y[0].astype('uint8'), aspect="auto")
            ax_x_img = plt.subplot2grid(grid_size, (1, 0), rowspan=1, colspan=1)
            ax_x_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_x_img.imshow(x_resize.astype('uint8'), aspect="auto")
            ax_h_img = plt.subplot2grid(grid_size, (2, 0), rowspan=1, colspan=1)
            ax_h_img.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax_h_img.imshow(prediction[0].astype('uint8'), aspect="auto")
            ax_loss = plt.subplot2grid(grid_size, (0, 1), rowspan=1, colspan=1)
            ax_loss.set_title("Loss")
            ax_acc = plt.subplot2grid(grid_size, (0, 2), rowspan=1, colspan=1)
            ax_acc.set_title("Accuracy")
            ax_loss_move = plt.subplot2grid(grid_size, (1, 1), rowspan=1, colspan=1)
            ax_acc_move = plt.subplot2grid(grid_size, (1, 2), rowspan=1, colspan=1)

            # plot whichever datasets are available

            if training_losses is not None:
                length = len(training_losses)
                x_axis = [i + 1 for i in range(length)]
                ax_loss.plot(x_axis, training_losses, color=training_color, label='Training')
                if length > 0:
                    last_val = training_losses[length - 1]
                    x = 0
                    if length > moving_window_size:
                        x = length - moving_window_size
                        x_axis = [i + 1 + (length - moving_window_size) for i in range(moving_window_size)]
                        ax_loss_move.plot(x_axis, training_losses[-moving_window_size:], color=training_color,
                                          label='Training')
                    else:
                        ax_loss_move.plot(x_axis, training_losses, color=training_color, label='Training')
                    ax_loss_move.text(x=x, y=last_val, s="{0:.2f}".format(last_val), color=training_color)

            if validation_losses is not None:
                length = len(validation_losses)
                x_axis = [i for i in range(length)]
                ax_loss.plot(x_axis, validation_losses, color=validation_color, label='Validation')
                if length > 0:
                    last_val = validation_losses[length - 1]
                    x = length - 1
                    if length - 1 > moving_window_size:
                        x_axis = [i + (length - moving_window_size) for i in range(moving_window_size)]
                        ax_loss_move.plot(x_axis, validation_losses[-moving_window_size:], color=validation_color,
                                          label='Validation')
                    else:
                        ax_loss_move.plot(x_axis, validation_losses, color=validation_color, label='Validation')
                    ax_loss_move.text(x=x, y=last_val, s="{0:.2f}".format(last_val), color=validation_color)

            if training_accuracies is not None:
                length = len(training_accuracies)
                x_axis = [i + 1 for i in range(length)]
                ax_acc.plot(x_axis, training_accuracies, color=training_color, label='Training')
                if length > 0:
                    last_val = training_accuracies[length - 1]
                    x = 0
                    if length > moving_window_size:
                        x = length - moving_window_size
                        x_axis = [i + 1 + (length - moving_window_size) for i in range(moving_window_size)]
                        ax_acc_move.plot(x_axis, training_accuracies[-moving_window_size:], color=training_color,
                                         label='Training')
                    else:
                        ax_acc_move.plot(x_axis, training_accuracies, color=training_color, label='Training')
                    ax_acc_move.text(x=x, y=last_val, s="{0:.2f}".format(last_val), color=training_color)

            if validation_accuracies is not None:
                length = len(validation_accuracies)
                x_axis = [i for i in range(length)]
                ax_acc.plot(x_axis, validation_accuracies, color=validation_color, label='Validation')
                if length > 0:
                    last_val = validation_accuracies[length - 1]
                    x = length - 1
                    if length - 1 > moving_window_size:
                        x_axis = [i + (length - moving_window_size) for i in range(moving_window_size)]
                        ax_acc_move.plot(x_axis, validation_accuracies[-moving_window_size:], color=validation_color,
                                         label='Validation')
                    else:
                        ax_acc_move.plot(x_axis, validation_accuracies, color=validation_color, label='Validation')
                    ax_acc_move.text(x=x, y=last_val, s="{0:.2f}".format(last_val), color=validation_color)

            plt.figlegend(loc='upper right')
            plt.savefig("%s/%s.jpg" % (self.prediction_path, name))

        return pred_shape, loss, accuracy

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y, batch_w):
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.net.cost,
                                                        self.net.accuracy,
                                                        self.net.predicter],
                                                       feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.w: batch_w,
                                                                  self.net.keep_prob: 1})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info(
            "Iter {:}, Minibatch Loss= {:.4f}, Training Accuracy= {:.4f}".format(step,loss, acc))
        return acc


training_path = 'D:\\upResNet\\temp\\training'
validation_path = 'D:\\upResNet\\temp\\training'
testing_path = 'D:\\upResNet\\temp\\training'

output_path = "D:\\upResNet\\temp\\output_eva"
restore_path = output_path

restore = True
padding = True
test_after = True
channels = 3
resolution = 2
batch_size = 20
validation_batch_size = 20
epochs = 500
learning_rate = 0.01
layers_per_transpose = 3

training_generator = generator(search_path=training_path + "/*npy", data_suffix="_x.npy", mask_suffix="_y.npy",
                               weight_suffix="_w.npy", shuffle_data=False, n_class=channels)
validation_generator = generator(search_path=validation_path + "/*npy", data_suffix="_x.npy", mask_suffix="_y.npy",
                                 weight_suffix="_w.npy", shuffle_data=False, n_class=channels)
testing_generator = generator(search_path=testing_path + "/*npy", data_suffix="_x.npy", mask_suffix="_y.npy",
                              weight_suffix="_w.npy", shuffle_data=False, n_class=channels)

training_iters = int(training_generator._get_number_of_files() / batch_size) + \
                     (training_generator._get_number_of_files() % batch_size > 0)
total_validation_data = validation_generator._get_number_of_files()

net = upResNet(padding=padding, channels=channels, resolution=resolution, layers_per_transpose=layers_per_transpose)

opt_kwargs = dict(learning_rate=learning_rate)

trainer = Trainer(net=net, batch_size=batch_size, validation_batch_size=validation_batch_size, opt_kwargs=opt_kwargs)
trainer.train(training_data_provider=training_generator, validation_data_provider=validation_generator,
              testing_data_provider=testing_generator, restore_path=restore_path, output_path=output_path,
              total_validation_data=total_validation_data, training_iters=training_iters, epochs=epochs,
              restore=restore, test_after=test_after, prediction_path=output_path+'\\prediction')
