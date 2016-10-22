import time

import numpy as np
np.random.seed(1234)

import math as m

import scipy.io
from scipy.interpolate import griddata
from scipy.misc import bytescale
from sklearn.preprocessing import scale
from utils import cart2sph, pol2cart
from sklearn.metrics import precision_score,recall_score,f1_score
import tensorflow as tf
import os
import cv2
import csv

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.
    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)

def gen_images(locs, features, nGridPoints, normalize=True, n_bands=3,
               augment=False, stdMult=0.1,edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode
    :param loc: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param nGridPoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each feature over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param stdMult:     Standard deviation of noise for augmentation
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """

    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes

    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] / nElectrodes
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])



    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):nGridPoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):nGridPoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, nGridPoints, nGridPoints]))

    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)

    for i in xrange(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method='cubic', fill_value=np.nan)
        print 'Interpolating {0}/{1}\r'.format(i+1, nSamples),

    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])

    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]

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


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.contrib.layers.xavier_initializer_conv2d())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases),weights

def build_cnn(input_var=None, W_init=None, n_layers=(4, 2, 1), n_filters_first=32, imSize=64, n_colors=3, id=None):
    """
    Builds a VGG style CNN network followed by a fully-connected layer and a softmax layer.
    Stacks are separated by a maxpool layer. Number of kernels in each layer is twice
    the number in previous stack.
    input_var: Theano variable for input to the network
    outputs: pointer to the output of the last layer of network (softmax)
    :param input_var: theano variable as input to the network
    :param n_layers: number of layers in each stack. An array of integers with each
                    value corresponding to the number of layers in each stack.
                    (e.g. [4, 2, 1] == 3 stacks with 4, 2, and 1 layers in each.
    :param n_filters_first: number of filters in the first layer
    :param W_init: Initial weight values
    :param imSize: Size of the image
    :param n_colors: Number of color channels (depth)
    :return: a pointer to the output of last layer
    """
    #g = tf.Graph()
    #with g.as_default():
    weights = []        # Keeps the weights for all layers
    count = 0
    # If no initial weight is given, initialize with GlorotUniform


    # Input layer
    #network = tf.placeholder(tf.float32,shape=(None, imSize, imSize, n_colors),name='Input')
    network = input_var
    if W_init is None:
        for i, s in enumerate(n_layers):
            if i==0:
                input_channel=3
            output_channel=n_filters_first * (2 ** i)
            for l in range(s):
                with tf.variable_scope(str(id)+"conv"+str(i)+"_"+str(l)):
                    network,W = conv_relu(network, [3, 3, input_channel, output_channel], [output_channel])
                    weights.append(W)
                    input_channel=output_channel
            network = tf.nn.max_pool(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID')
    else:
        for i, s in enumerate(n_layers):
            bias_shape=[n_filters_first * (2 ** i)]
            for l in range(s):
                with tf.variable_scope(str(id)+"conv" + str(i) + "_" + str(l)):
                    biases = tf.get_variable("biases", bias_shape,
                                             initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv2d(network, W_init[count],
                                        strides=[1, 1, 1, 1], padding='SAME')
                    network = tf.nn.relu(conv + biases)
                    count+=1
            network = tf.nn.max_pool(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='VALID')
        weights = W_init
    return network, weights


def build_convpool_mix(input_vars, nb_classes, GRAD_CLIP=100, imSize=64, n_colors=3, n_timewin=7,train=False):
    """
    Builds the complete network with LSTM and 1D-conv layers combined
    :param input_vars: list of EEG images (one image per time window)
    :param nb_classes: number of classes
    :param GRAD_CLIP:  the gradient messages are clipped to the given value during
                        the backward pass.
    :return: a pointer to the output of last layer
    """
    print input_vars.get_shape()
    convnets = []
    #convnets = tf.variable('convnets',shape = [None,1])
    W_init = None
    # Build 7 parallel CNNs with shared weights
    for i in xrange(n_timewin):
        if i == 0:
            convnet, W_init = build_cnn(input_vars[:,i,:,:,:], imSize=imSize, n_colors=n_colors, id=i)
        else:
            convnet, _ = build_cnn(input_vars[:,i,:,:,:], W_init=W_init, imSize=imSize, n_colors=n_colors,id=i)
        print convnet.get_shape()
        convnets.append(tf.contrib.layers.flatten(convnet))
        #convnets = tf.concat(1,convnet)
        
    #convnets = np.array(convnets)
    #print convnets[0].get_shape()
    convpool = tf.pack(convnets,axis = 1)
    print convpool.get_shape()
    # at this point convnets shape is [numTimeWin][n_samples, features]
    # we want the shape to be [n_samples, features, numTimeWin]
   # convpool = tf.concat(1,convnets)
    # convpool = ReshapeLayer(convpool, ([0], -1, numTimeWin))
    #convpool = tf.reshape(convpool, [tf.shape(convpool)[0], n_timewin, tf.shape(convnets[0])[1]])
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
    return convpool


if __name__ == '__main__':
    X = tf.placeholder(tf.float32,shape=(None, 7, 64, 64, 3),name='Input')
    y = tf.placeholder(tf.float32)
    train = tf.placeholder(tf.bool)
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
    #print train_images.shape,train_labels.shape,test_images.shape,test_labels.shape
    #print os.getcwd()
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
            while (batch_no*batch_size) < train_images.shape[0]:
                ind = batch_no*batch_size
                # print ind
                if ind + batch_size < train_images.shape[0]:
                    batch_images = train_images[ind:ind+batch_size,:,:,:,:]
                    batch_labels = train_labels[ind:ind+batch_size,:]
                    sess.run([train_step], feed_dict={X: batch_images, y: batch_labels, train: True })
                else:
                    batch_images = train_images[ind:,:,:,:,:]
                    batch_labels = train_labels[ind:,:]
                    sess.run([train_step], feed_dict={X: batch_images, y: batch_labels, train: True })
                batch_no += 1
            print "Train step for epoch "+str(i)+" Done!!"
            test_accuracy = sess.run([accuracy], feed_dict={
                X: test_images, y: test_labels, train: False})
            print "Test accuracy for "+str(i),test_accuracy 
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
    
