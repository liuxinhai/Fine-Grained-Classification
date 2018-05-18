import tensorflow as tf
import numpy as np
import vgg19_trainable as vgg19
import os
import utils
import math
import random
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='2'

test_path = '/home3/lhl/modelnet40_total_v2/test/'
test_files = os.listdir(test_path)
test_files.sort()
test_labels = np.load('./modelnet40_total_12v/test_labels.npy')

batch_size = 12
model_views = 12
class_nums = 40
test_models = 2468
learning_rate = 0.0002
batch_models = batch_size/model_views

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [None, class_nums])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)
vgg.reshape_layer = tf.reshape(vgg.relu7, [-1,12*4096], 'reshape')
vgg.fc8 = vgg.fc_layer(vgg.reshape_layer, 12*4096, 4096, "fc8")
vgg.fc9 = vgg.fc_layer(vgg.fc8, 4096, class_nums, "fc9")

vgg.y = tf.nn.softmax(vgg.fc9, name='result')


loss = tf.reduce_sum((vgg.y - y_)**2)
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(vgg.y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

test_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
test_label = np.zeros((batch_size/model_views, class_nums), dtype=np.float32)

for epo in range(100):
	sum = .0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		saver.restore(sess, './train_models/batch_size_60_epo_%d_total_v2.ckpt'%(epo+1))
		for i in range(test_models/batch_models):
			for j in range(batch_size):
				test_batch[j] = utils.load_image(test_path+test_files[i*batch_size+j])
			for j in range(batch_models):
				test_label[j] = test_labels[i*batch_models+j]
        		test_acc = sess.run(accuracy, feed_dict={images: test_batch, y_: test_label, train_mode:False})
			sum += test_acc
		sess.close()
	print('epo %d,total acc:%g'%(epo+1, sum/test_models))
