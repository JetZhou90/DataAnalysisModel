from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os
import sys
import six.moves.cPickle as pickle
import gzip
import theano.tensor as T
def load_data(dataset):
    with gzip.open(dataset, 'rb') as f:
        try:
            x,y = pickle.load(f, encoding='latin1')
        except:
            x,y = pickle.load(f)
    return x,y
            
            
datasets=load_data('data_64.pkl.gz')


x=datasets[0]

y=datasets[1]
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=0)
batch_size=10

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 64*64])   # 28x28
ys = tf.placeholder(tf.float32, [None, 4])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 64, 64, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
### Construct the first convolutional pooling layer:
W_conv1 = weight_variable([5,5, 1,18]) # patch 5x5, in size 1(image depth), out size 18
b_conv1 = bias_variable([18])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 18, 36]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([36])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # input size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## fc1 layer ##
W_fc1 = weight_variable([16*16*36, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 16*16*36])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
# important step
init = tf.global_variables_initializer()
sess.run(init)

for i in range(20):
    batch_xs = X_train
    batch_ys = y_train
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    print(compute_accuracy(X_test, y_test))

    #y_pre=sess.run(prediction, feed_dict={xs: X_test[10], keep_prob: 1})
    #print y_pre[0].flatten(),y_test[10]
    #for x in y_pre:
        #x=np.where(x==np.max(x))
        #print 'y_pre',x[0] 
    #for y in y_test:
        #y=np.where(y==np.max(y))
        #print 'y_test',y[0]
    
        #print mnist.test.images[i],mnist.test.labels[i]
