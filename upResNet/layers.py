import tensorflow as tf

# layers & variable definitions
def weight_variable(shape, stddev, name="weight", trainable=True):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name, trainable=trainable)


def bias_variable(shape, name="bias", trainable=True):
    return tf.Variable(tf.constant(0.1, shape=shape), name=name, trainable=trainable)


def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        return tf.nn.dropout(tf.nn.bias_add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID'), b), keep_prob_)


def deconv2d(x, W, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], tf.to_int32(tf.to_float(x_shape[1]) * 2),
                                 tf.to_int32(tf.to_float(x_shape[2]) * 2), x_shape[3]])  # // 2])
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
