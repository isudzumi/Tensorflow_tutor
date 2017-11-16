import tensorflow as tf
import numpy as np

# y = 0.5 * x + 10 の１次関数をTensorflowで予測する

# 学習データ作成
x_data = np.random.rand(100).astype(np.float32)
y_data = 0.5 * x_data + 10

# モデル作成
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

cost = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.AdamOptimizer(0.5)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
