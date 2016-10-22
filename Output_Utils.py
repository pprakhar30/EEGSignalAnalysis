import scipy.io
from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utils import cart2sph, pol2cart
import tensorflow as tf
import os
import cv2
import csv
import sklearn as sk

def Y_Output():

	mylist = [1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,0]
	myfile = open("sandeep_Y.csv", 'wb')
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	wr.writerow(mylist)

	with open('prakhar_Y.csv', 'rb') as csvfile:
		     spamreader = csv.reader(csvfile)
		     for row in spamreader:
				print row
def Get_Convpool(train_images,test_images):
	with open("vgg16.tfmodel", mode='rb') as f:
  		fileContent = f.read()

	graph_def = tf.GraphDef()
	graph_def.ParseFromString(fileContent)
	images = tf.placeholder(tf.float32,shape = (None, 64, 64, 3))
	tf.import_graph_def(graph_def, input_map={ "images": images })
	print "graph loaded from disk"

	graph = tf.get_default_graph()
	with tf.Session() as sess:
	    init = tf.initialize_all_variables()
	    sess.run(init)
	    #batch = np.reshape(input_vars,(-1, 224, 224, 3))
	    n_timewin = 7   
	    convnets_train = []
	    convnets_test = []
	    for i in xrange(n_timewin):
	        pool_tensor = graph.get_tensor_by_name("import/pool5:0")
	        feed_dict = { images:train_images[:,i,:,:,:] }
	        convnet_train = sess.run(pool_tensor, feed_dict=feed_dict)
	        convnets_train.append(tf.contrib.layers.flatten(convnet_train))
	        feed_dict = { images:test_images[:,i,:,:,:] }
	        convnet_test = sess.run(pool_tensor, feed_dict=feed_dict)
	        convnets_test.append(tf.contrib.layers.flatten(convnet_test))
	        
	    convpool_train = tf.pack(convnets_train, axis = 1)
	    convpool_test = tf.pack(convnets_test ,axis = 1)
	    x = convpool_train.get_shape()[2]
	    convpool_train = sess.run(convpool_train)
	    convpool_test = sess.run(convpool_test)
	return convpool_train,convpool_test,x

