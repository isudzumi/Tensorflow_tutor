import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def inference(images_placeholder, keep_prob):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding ='SAME')

    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 1])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 1, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

    with tf.name_scope('softmax') as scope:
        y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def loss(logits, labels):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
    tf.summary.scalar('cross_entropy', cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

if __name__ == '__main__':
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 784])
        y_ = tf.placeholder(tf.float32, [None, 10])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        # y = tf.nn.softmax(tf.matmul(x, W) + b)
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar("dropout_keep_probability", keep_prob)

        # 学習
        logits = inference(x, keep_prob)
        loss_value = loss(logits, y_)
        train_op = training(loss_value, 1e-4)
        accuracy = accuracy(logits, y_)

        sess = tf.Session()

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("MNIST_data" + '/train', sess.graph)
        test_writer = tf.summary.FileWriter("MNIST_data" + '/test', sess.graph)

        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(5000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                
                # if you write like this, then <Type Error: invalid type <type 'numpy.float32'>>
                # "... accuracy = sess.run([..., accuracy], feed_dict=... "

                summary, acc = sess.run([summary_op, accuracy], feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0
                    })
                test_writer.add_summary(summary, i)
                print("step %d, training accuracy %g" % (i, acc))
            summary, _ = sess.run([summary_op, train_op], feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0
                })
            train_writer.add_summary(summary, i)

        summary, acc = sess.run([summary_op, accuracy], feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
            })
        test_writer.add_summary(summary, i)
        print("test accuracy: %g" %  acc)
