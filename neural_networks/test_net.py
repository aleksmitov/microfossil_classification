import tensorflow as tf

from cnn_building_blocks import conv2d, max_pool, weight_variable, bias_variable


def test_net(x, number_of_classes, dropout_rate, is_training=False):
    net_name = "Test_Net"
    image_channels = x.shape[3].value

    # First convolutional layer - maps one image to 32 feature maps
    with tf.variable_scope("Conv_1"):
        w_conv1 = weight_variable([5, 5, image_channels, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X
        h_pool1 = max_pool(h_conv1, size=(2,2))

    with tf.variable_scope("Conv_2"):
        # Second convolutional layer - maps 32 feature maps to 64
        w_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

        # Second pooling layer
        h_pool2 = max_pool(h_conv2, size=(2,2))

    with tf.variable_scope("Conv_3"):
        # Third convolutional layer - maps 64 feature maps to 64
        w_conv3 = weight_variable([5, 5, 64, 64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)

        # Second pooling layer
        h_pool3 = max_pool(h_conv3, size=(2,2))
        layer3_size = h_pool3.shape[1].value * h_pool3.shape[2].value * h_pool3.shape[3].value

    with tf.variable_scope("FC_1"):
        # Fully connected layer 1 - after 3 rounds of downsampling, our 200x200
        # image is down to 25x25x64 feature maps - maps this to 1024 features.
        w_fc1 = weight_variable([layer3_size, 1024])
        b_fc1 = bias_variable([1024])

        h_pool3_flat = tf.reshape(h_pool3, [-1, layer3_size])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
        #print("Shape: ", h_fc1.get_shape().as_list())

    with tf.variable_scope("FC_2"):
        # Map the 1024 features to the classes
        w_fc2 = weight_variable([1024, number_of_classes])
        b_fc2 = bias_variable([number_of_classes])

        h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
        return h_fc2, net_name
