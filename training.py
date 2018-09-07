import numpy as np 
import tensorflow as tf 

# each picture is 64x64 pixles
height = 64 
width = 64
# 3 channels
channels = 3

n_inputs = height * width * channels
n_outputs = 200 # 200 different classes

# reset_graph()

X = tf.placeholder(tf.float32, shape=[-1, height, width, channels] )
y = tf.placeholder(tf.int32, shape=[None] )

#input shape [-1, 64, 64, 3]
conv1 = tf.layers.conv2d( inputs=X, filters=32, kernel_size=[5,5], padding='SAME', activation=tf.nn.relu )

#shape after conv1: [-1, 64, 64, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

conv2 = tf.layers.conv2d( inputs=pool1, filters=64, kernel_size=[5,5], padding='SAME', activation=tf.nn.relu )
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Dense Layer
pool2FinalOutput = tf.reshape(pool2, [-1, 8 * 8 * 64])
dense = tf.layers.dense(inputs=pool2FinalOutput, units=1024, activation=tf.nn.relu)
dropout = tf.layers.dropout(inputs=dense, rate=0.4)
dropoutReshaped = tf.reshape(dropout, [-1, 8 * 8 * 64])

# Logits Layer
logits = tf.layers.dense(inputs=dropoutReshaped, units=200 )
Y1 = tf.nn.softmax(logits)

ent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(ent)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[9]:

n_epochs = 10
batch_size = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for batch in get_next_batch():
            X_batch, y_batch = batch[0], batch[1]
            #print ('Training set', X_batch.shape, y_batch.shape)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
       
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: val_images, y: val_labels_encoded})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./tiny_imagenet")

