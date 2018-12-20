"""
The format of this file is influenced by jakeret's implementation of Unet
"""

import numpy as np
import tensorflow as tf
import os

# layers & variable definitions
def weight_variable(shape, stddev, name="weight"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name="bias"):
    return tf.Variable(tf.constant(0.1,shape=shape),name=name)

def conv2d(x,W,b,keep_prob_):
    with tf.name_scope("conv2d"):
        return tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID'),b), keep_prob_)

# TODO: I want a pattern of upscaling like so: (200x200,300x300,400x400,600x600,800x800,1200x1200,1600x1600...)
def deconv2d(x,W,midpoint,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        if midpoint:
            ratio = 1.5
        else
            ratio = 1.3333
        output_shape = tf.stack([x_shape[0],x_shape[1]*ratio, x_shape[2]*ratio, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,stride,stride,1], padding='VALID', name="conv2d_transpose")

# no pooling layers
def create_newtwork(x, keep_prob, channels, padding=False, resolution=3, features_root=16, filter_size=3, deconv_size=2, summaries=True):

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
        features = features_root,
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
    
    size = width
    which_conv = 0
    which_up_conv = 0
    features = channels
    while size < (width*resolution):
        # Convolutions...
        with tf.name_scope("Conv{}".format(str(which_conv))):

            # TODO: decide on how many features I want @ each step
            prev_step_features = features
            features = features_root
            stddev = np.sqrt(2 / (filter_size**2*features))

            w1 = weight_variable([filter_size, filter_size, prev_step_features, features], stddev, name="w1")
            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            
            b1 = bias_variable([features], name="b1")
            b2 = bias_variable([features], name="b2")

            if padding:
                in_node = tf.pad(in_node, paddings=[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
            conv1 = conv2d(in_node, w1, b1, keep_prob)
            tmp_conv = tf.nn.relu(conv1)
            convsDict[which_conv] = tmp_conv
            which_conv += 1
            if padding:
                tmp_conv = tf.pad(tmp_conv, paddings=[[0,0],[1,1],[1,1],[0,0]], mode='SYMMETRIC')
            conv2 = conv2d(tmp_conv, w2, b2, keep_prob)
            convsDict[which_conv] = tf.nn.relu(conv2)

            weights.append((w1,w2))
            biases.append((b1,b2))
            convs.append((conv1,conv2))

        # Upscalings...
        with tf.name_scope("Up_Conv{}".format(which_up_conv)):
            in_node = convsDict[which_conv]
            midpoint = which_up_conv%2 != 0
            wd = weight_variable([deconv_size, deconv_size,features,features], stddev, name="wd") #what is the size?
            bd = bias_variable([features], name="bd")
            deconv = tf.nn.relu(deconv2d(in_node, wd, midpoint, deconv_size) + bd)
            deconvDict[which_up_conv] = deconv
            in_node = deconv
            which_up_conv += 1
            if midpoint:
                size *= 1.5
            else
                size *= 1.3333

    # output
    with tf.name_scope("output"):
        weight = weight_variable([1,1, features_root, channels], stddev)
        bias = bias_variable([channels], name="bias")
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
        output = tf.nn.relu(conv)
        convsDict("out") = output

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))
            tf.summary.image('summary_output', get_image_summary(convsDict("out")))
            for k in deconvsDict.keys():
                tf.summary.image('summary_deconv_%02d' % k, get_image_summary(deconvsDict[k])))
            
    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        varialbes.append(w2)
    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output, variables

class upResNet(object):

    def __init__ (self, padding, channels=3, resolution=2):
        tf.reset_default_graph()

        self.summaries = kwargs.get("summaries", True)

        self.x = tf.placeholder("float", shape=[None,None,None,channels], name='x')
        self.y = tf.placeholder("float", shape=[None,None,None,channels], name='y')
        # weighted map would be a map of detected edges
        self.w = tf.placeholder("float", shape=[None,None,None,1], name='w')

        logits, self.variables = create_network(self.x, channels, resolution, padding, **kwargs)

        self.cost = self._get_cost(logits)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        self.logits = logits

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.logits, self.y), float32))
        

    def _get_cost(self, logits):
        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, channels]) # channels may not reach this scope
            flat_labels = tf.reshape(self.y, [-1, channels])
            flat_map = tf.reshape(self.w, [-1,1])

            """
            Current plan for loss function is a square of the pixelwise difference between X and Y
            """
            loss_map = tf.math.square(tf.math.subtract(flat_logits,flat_labels))
            loss = tf.reduce_mean(tf.multiply(loss_map, flat_map))

            return loss

    def predict(self, path, x_test):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            lgts = sess.run(init)
            self.restore(sess, path)

class Trainer(object):

    def __init__(self, net, batch_size=1, validation_batch_size=1, opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.validation_batch_size = validation_batch_size

    def _get_optimizer(self, training_iters, global_step):
        learning_rate = self.opt_kwargs.pop("learning_rate", 0.001)
        self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, beta1=0.9,
                                           epsilon=1e-8, **self.opt_kwargs).minimize(self.net.cost,
                                                                                     global_step=global_step)
        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0, name="global_step")
        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                               name="norm_gradients")
        if self.net.summaries and self.net.norm_grads:
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)

        tf.summary.image('prediction', get_image_summary(self.net.logits))

        self.optimizer = self._get_optimizer(training_iters, global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = self.global_variables_initializer()

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

    def train(self, training_data_provider, validation_data_provider, restore_path,
              output_path, total_validation_data, training_iters=10, epochs=100,
              dropout=0.75, display_step=1, include_map=True, restore=False,
              write_graph=False, prediction_path='prediction'):
        
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore, prediction_path)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

                sess.run(init)

                if restore:
                    ckpt = tf.train.get_checkpoint_state(restore_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        self.net.resotre(sess, ckpt.model_checkpoint_path)

                validation_avg_losses = []
                training_avg_losses = []
                training_accuracies = []
                validation_accuracies = []


    def validate(self, sess, total_validation_data, validation_data_provider, include_map,
                 training_average_losses, training_accuracies, validation_average_losses,
                 validation_accuracies, name):
        total_validation_loss = 0
        total_validation_acc = 0
        validation_iters = int(total_validation_data/self.validation_batch_size)
        last_batch_size = total_validation_data - (validation_iters*self.validation_batch_size)

        for i in range(0, validation_iters):
            test_x, test_y, test_w = validation_data_provider(self.verification_batch_size)
            if not include_map:
                test_w = np.ones
            pred_shape, loss, accuracy = self.store_prediction(sess, test_x, test_y, test_w, name=name, save_img=)
        test_x, test_y, test_w = validation_data_provider(last_batch_size)
        
