import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/MNIST_data', one_hot=True)

def loss(logits, labels):
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.summary.scalar('cross_entropy', cross_entropy)
        return cross_entropy

def training(loss, learning_rate):
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return train_step

def accuracy(logits, labels):
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

if __name__ == '__main__':
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, [None, 28, 28])
        y_ = tf.placeholder(tf.float32, [None, 10])
        W = tf.Variable(tf.random_normal([128, 10]))
        b = tf.Variable(tf.random_normal([10]))
        # y = tf.nn.softmax(tf.matmul(x, W) + b)
        keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope('RNN') as scope:
            xs = tf.unstack(x, 28, 1)
            lstm_cell = rnn.BasicLSTMCell(128, forget_bias=1.0)
            outputs, states = tf.nn.static_rnn(lstm_cell, xs, dtype=tf.float32)

        # 学習
        logits = tf.matmul(outputs[-1], W) + b
        loss_value = loss(logits, y_)
        train_op = training(loss_value, 0.001)
        accuracy = accuracy(logits, y_)

        with tf.Session() as sess:

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("MNIST_data" + '/train', sess.graph)
            test_writer = tf.summary.FileWriter("MNIST_data" + '/test', sess.graph)

            init = tf.global_variables_initializer()
            sess.run(init)

            for i in range(5000):
                # batch_x & batch_y must be <class 'numpy.ndarray'> (Not Tensor object)
                batch_x, batch_y = mnist.train.next_batch(128)
                batch_x = batch_x.reshape((128, 28, 28))
                if i % 100 == 0:
                    summary, acc = sess.run([summary_op, accuracy], feed_dict={
                        x: batch_x, y_: batch_y, keep_prob: 1.0
                        })
                    test_writer.add_summary(summary, i)
                    print("step %d, training accuracy %g" % (i, acc))
                summary, _ = sess.run([summary_op, train_op], feed_dict={
                    x: batch_x, y_: batch_y, keep_prob: 1.0
                    })
                train_writer.add_summary(summary, i)

            mnist_test_image = mnist.test.images[:128].reshape((-1, 28, 28))
            mnist_test_label = mnist.test.labels[:128]
            summary, acc = sess.run([summary_op, accuracy], feed_dict={
                x: mnist_test_image, y_: mnist_test_label, keep_prob: 1.0
                })
            test_writer.add_summary(summary, i)
            print("test accuracy: %g" %  acc)
