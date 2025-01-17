import tensorflow as tf
import numpy as np
## Save to file

#W= tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
#b=tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#saver= tf.train.Saver()
#with tf.Session () as sess:
    #sess.run(tf.global_variables_initializer())
    #save_path=saver.save(sess, "my_model/save_net.ckpt")
    #print ("Save to path:",save_path)
    
W=tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b=tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "my_model/save_net.ckpt")
    print("weight",sess.run(W))
    print ("biases",sess.run(b))