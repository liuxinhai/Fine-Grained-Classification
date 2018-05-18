import tensorflow as tf
import numpy as np
import vgg19_trainable as vgg19
import os
import utils
import math
import random

os.environ['CUDA_VISIBLE_DEVICES']='0'

test_path = '/home3/lxh/modelnet/modelnet40v1_1/test/'
test_files = os.listdir(test_path)
test_files.sort()
train_path = '/home3/lxh/modelnet/modelnet40v1_1/train/'
train_files = os.listdir(train_path)
train_files.sort()

train_numbers = [80]*40
train_numbers[6] = 64
train_numbers[10] = 79

batchsize = 3
epoch = 5
display_per_epoch = 1

train_labels = [[i]*train_numbers[i] for i in range(40)]
train_labels = sum(train_labels,[])

images = tf.placeholder(tf.float32, [batchsize, 224, 224, 3])
realOut = tf.placeholder(tf.float32, [batchsize, 40])
train_mode = tf.placeholder(tf.bool)

sess = tf.Session()
vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)
vgg.fc9 = vgg.fc_layer(vgg.relu7, 4096, 40, "fc9")
vgg.myprob = tf.nn.softmax(vgg.fc9, name='myprob')

print(vgg.get_var_count())

sess.run(tf.global_variables_initializer())
cost = tf.reduce_sum((vgg.myprob-realOut)**2)
train = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)
correct_prediction = tf.equal(tf.argmax(vgg.myprob, 1), tf.argmax(realOut, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for epo in range(epoch):
	if epo%display_per_epoch == 0:
		output_flag = 1
	else:
		output_flag = 0
	batch_row_size = int(math.ceil(3183.0/batchsize)) 	#3183
	print batch_row_size
	pos_begin = [3183-batchsize if batchsize*(i+1)>3182 else batchsize*i for i in range(batch_row_size)]
	for i in range(batch_row_size):
		imfiles = train_files[pos_begin[i]:pos_begin[i]+batchsize]
		imlist = [utils.load_image(train_path+imfiles[j]) for j in range(batchsize)]
		imbatch = [im.reshape((224, 224, 3)) for im in imlist]
		im_true_result = [[1 if k == train_labels[pos_begin[i]+j] else 0 for k in range(40)] for j in range(batchsize)]  
		_,cost_compute = sess.run([train,cost], feed_dict={images:imbatch, realOut:im_true_result, train_mode:True})
		if output_flag == 1:
			print "loss:",cost_compute," in No.",i,"batch, totally",batch_row_size
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={images:imbatch, realOut:im_true_result,keep_prob:1.0})
			print ('step %d,training accuracy %g' % (i, train_accuracy))
	if output_flag == 1:
		vgg.save_npy(sess, '/home2/liuxinhai/modelnet/vgg19/signal-view/M7D13'+str(epo)+'batchsize'+str(batchsize)+'.npy')



