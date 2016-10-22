import time

import numpy as np
np.random.seed(1234)

import math as m

import scipy.io
from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utils import cart2sph, pol2cart
from Output_Utils import Get_Convpool
from sklearn.metrics import precision_score,recall_score,f1_score
import tensorflow as tf
import os
import cv2
import csv
import sklearn as sk

def LoadData():
    X = []
    Y = []
    source = ["ashish2.mp4","Mehul2.mp4","Prakhar.mp4","rahul.mp4","raj.mp4","rishi.mp4","rohit.mp4","rohitkm.mp4","sandeep.mp4","satish.mp4","shubham.mp4","touqeer.mp4"]
    os.chdir("./Data")  
    for file in source:
        os.chdir("./"+file+"/TimedVersion/")
        for i in xrange(25):
            x = []
            for j in xrange(7):
                file_name = "Word_"+str(i)+"Seq_"+str(j)+".jpg"
                img = cv2.imread(file_name)
                x.append(img)
            X.append(x)
        os.chdir("../..")
        file = file[:-4]+"_Y.csv"
        with open(file, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for i in row:
                    if i=='1':
                        Y.append([0,1])
                    else:
                        Y.append([1,0])

    os.chdir("..")
    X = np.array(X)
    train_images = X[:275,:,:,:,:]
    test_images = X[275:,:,:,:,:]
    Y = np.array(Y)
    train_labels = Y[:275,:]
    test_labels = Y[275:,:]
    return (train_images,train_labels,test_images,test_labels)


'''def Get_Pool5_Tensor(input_vars, graph):
    
    with tf.Session() as sess:
      init = tf.initialize_all_variables()
      sess.run(init)
      batch = np.reshape(input_vars,(-1, 224, 224, 3))
      feed_dict = { images: batch }

      pool_tensor = graph.get_tensor_by_name("import/pool5:0")
      pool_tensor = sess.run(pool_tensor, feed_dict=feed_dict)
    return pool_tensor'''


def Get_Pre_Trained_Weights(input_vars,name):
    with open("vgg16.tfmodel", mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    images = tf.placeholder(tf.float32,shape = (None, 64, 64, 3),name=name)
    tf.import_graph_def(graph_def, input_map={ "images": images })
    print "graph loaded from disk"

    graph = tf.get_default_graph()
    with tf.Session() as sess:
        init = tf.initialize_all_variables()
        sess.run(init)
        #batch = np.reshape(input_vars,(-1, 224, 224, 3))
        n_timewin = 7   
        convnets = []
        for i in xrange(n_timewin):
            feed_dict = { images:input_vars[:,i,:,:,:] }
            pool_tensor = graph.get_tensor_by_name("import/pool5:0")
            pool_tensor = sess.run(pool_tensor, feed_dict=feed_dict)
            convnets.append(tf.contrib.layers.flatten(pool_tensor))
        convpool = tf.pack(convnets, axis = 1)
        return convpool

def build_convpool_mix(convpool, nb_classes, GRAD_CLIP=100, imSize=64, n_colors=3, n_timewin=7,train=False):
    """
    Builds the complete network with LSTM and 1D-conv layers combined
    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param GRAD_CLIP:  the gradient messages are clipped to the given value during
                        the backward pass.
    :return: a pointer to the output of last layer
    """
    convpool = tf.convert_to_tensor(convpool)
    reformConvpool = tf.expand_dims(convpool,2)
    conv_out = None
    x = int (convpool.get_shape()[2])
    print type(x)
    with tf.variable_scope("CONVFINAL"):
        weights = tf.get_variable("weights", [3,1,x,64],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        bias = tf.get_variable("biases", [64], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(reformConvpool, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
        conv = conv[:, :, 0, :]
        conv_out = tf.nn.relu(conv + bias)

    conv_out = tf.contrib.layers.flatten(conv_out)
    print conv_out.get_shape(),"SSSSSSSS"
    # Input to LSTM should have the shape as (batch size, SEQ_LENGTH, num_features)
    #lstm = LSTMLayer(convpool, num_units=128, grad_clipping=GRAD_CLIP,
    #    nonlinearity=lasagne.nonlinearities.tanh)
    num_hidden = 128
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True, cell_clip=GRAD_CLIP,initializer=tf.constant_initializer(0.0),
                state_is_tuple=True)
    lstm = tf.nn.dynamic_rnn(cell,convpool,scope="LSTM", dtype=tf.float32)
    print type(lstm), type(lstm[0]), type(lstm[1])
    # After LSTM layer you either need to reshape or slice it (depending on whether you
    # want to keep all predictions or just the last prediction.
    # http://lasagne.readthedocs.org/en/latest/modules/layers/recurrent.html
    # https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py
    # lstm_out = SliceLayer(convpool, -1, 1)        # bypassing LSTM
    lstm_out = tf.slice(lstm[0],[0,n_timewin-1,0],[-1,1,-1])
    lstm_out = tf.contrib.layers.flatten(lstm_out)
    # Merge 1D-Conv and LSTM outputs
    #print lstm_out.get_shape(),"$$$$$$$$"
    dense_input = tf.concat(1,[conv_out, lstm_out])
    if train:
        dense_input = tf.nn.dropout(dense_input, 0.5)
    #print dense_input.get_shape(),"FFFFFFFF"
    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    shape = dense_input.get_shape().as_list()
    print shape
    dim = 1
    for d in shape[1:]:
        dim *= d
    x = tf.reshape(dense_input, [-1, dim])
    print x.get_shape(),"LLLLLLL",type(dim)
    with tf.variable_scope("FC1"):
        weights = tf.get_variable("weights",[dim,512], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.0))
        x = tf.nn.bias_add(tf.matmul(x, weights), bias)
        convpool = tf.nn.relu(x)
    # We only need the final prediction, we isolate that quantity and feed it
    # to the next layer.
    with tf.variable_scope("FC2"):
        weights = tf.get_variable("weights",[512,nb_classes], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable("biases", [nb_classes], initializer=tf.constant_initializer(0.0))
        convpool = tf.nn.bias_add(tf.matmul(convpool, weights), bias)
        convpool = tf.nn.softmax(convpool)
    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    print type(convpool),convpool.get_shape()
    return convpool


if __name__ == '__main__':
    
    
    '''locs = scipy.io.loadmat('path')
    locs_3d = locs['A']
    locs_2d = []
    # Convert to 2D
    for e in locs_3d:
        locs_2d.append(azim_proj(e))
    feats = scipy.io.loadmat('path')
    test_images = scipy.io.loadmat('path')
    images = gen_images(np.array(locs_2d),
                        feats['features'][:, :192],
                        32, augment=True, pca=True, n_components=2)
    test_images = gen_images(np.array(locs_2d),
                        test_images['features'][:, :192],
                        32, augment=True, pca=True, n_components=2)
    test_y = scipy.io.loadmat('path')
    answer = scipy.io.loadmat('path')'''
    train_images, train_labels, test_images, test_labels =  LoadData()
    (convpool_train,convpool_test,x) = Get_Convpool(train_images,test_images)
    print convpool_train.shape
    X = tf.placeholder(tf.float32,shape=(None,7,x),name='Input')
    y = tf.placeholder(tf.float32)
    train = tf.placeholder(tf.bool)
    network = build_convpool_mix(X, 2, 90, train)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(network), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.initialize_all_variables()
    batch_size = 64
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            batch_no = 0
            while (batch_no*batch_size) < convpool_train.shape[0]:
                ind = batch_no*batch_size
               # print ind
                if ind + batch_size < convpool_train.shape[0]:
                    batch_images = convpool_train[ind:ind+batch_size,:,:]
                    batch_labels = train_labels[ind:ind+batch_size,:]
                    sess.run([train_step], feed_dict={X: batch_images, y: batch_labels, train: True })
                else:
                    batch_images = convpool_train[ind:,:,:]
                    batch_labels = train_labels[ind:,:]
                    sess.run([train_step], feed_dict={X: batch_images, y: batch_labels, train: True })
                batch_no += 1
            print "Train step for epoch "+str(i)+" Done!!"
            test_accuracy = sess.run([accuracy], feed_dict={
                X: convpool_test, y: test_labels, train: False})
            print("step %d, training accuracy %g" % (i, test_accuracy[0]))
        y_true = np.argmax(test_labels,1)
        y_p = sess.run(network, feed_dict={X: convpool_test, y:test_labels,  train: False})
    	#y_p = y_p[0,:,:]
    	print type(y_p),y_p.shape
        y_pred = np.argmax(y_p, 1)
    	print y_pred
    	print y_true
    	print len(y_pred)
        print "Precision", precision_score(y_true, y_pred)
        print "Recall", recall_score(y_true, y_pred)
        print "f1_score", f1_score(y_true, y_pred)
        print "confusion_matrix"
        print sk.metrics.confusion_matrix(y_true, y_pred)

    
    
