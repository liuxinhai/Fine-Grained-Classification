import tensorflow as tf
import numpy as np
import vgg19_trainable as vgg19
import os
import utils
import math
import random
import cv2

os.environ['CUDA_VISIBLE_DEVICES']='3'

train_path = '/home2/lxh/Faster_RCNN_Train_Data/airplane_v1/train/'
train_files = os.listdir(train_path)
train_files.sort()
train_labels = np.load('/home2/lxh/Faster_RCNN_Train_Data/airplane_v1/train_labels.npy')

batch_size = 60
model_views = 12
epoch = 100001
class_nums = 40
learning_rate = 0.0002
train_models = 9843


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

saver = tf.train.Saver(max_to_keep=500)

train_batch = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
batch_models = batch_size / model_views
labels = np.zeros((batch_models, class_nums), dtype=int)
perm = np.arange(train_models)

num_batches = train_models*model_views / batch_size

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, './train_models/batch_size_60_epo_8_total_v2.ckpt')
	for epo in range(epoch):
		np.random.shuffle(perm)
		for i in range(num_batches):
  			for j in range(batch_models):
  				num = i*batch_models + j
            			for k in range(model_views):
                			train_batch[j*model_views + k] = utils.load_image(train_path + train_files[perm[num]*model_views + k])
				labels[j] = train_labels[perm[num]]
            		_,cost_compute = sess.run([train,loss], feed_dict={images:train_batch, y_:labels, train_mode:True})
            		print('step %d,loss %f' % (i, cost_compute))
        	if epo > 0:
			saver.save(sess, "./train_models/batch_size_{}_epo_{}_total_v2.ckpt".format(batch_size, epo+8))
