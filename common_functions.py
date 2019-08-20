# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:17:30 2019

@author: RayKMAllen
"""

import os
import cv2
import sqlite3
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from utils import get_random_eraser, rpy_to_unit_vector
from scipy.ndimage import rotate, shift
import timeit
from functools import partial

np.set_printoptions(suppress = True, precision = 8)

def get_data(resolution, ram = True, num_data = None):

    global names
    
    imagepath = 'aflw/data/rescaled/' + str(resolution) + '/all/'
    dbpath = 'aflw/data/aflw.sqlite'
    
    #Image counter
    counter = 1

    #Open the sqlite database
    conn = sqlite3.connect(dbpath)
    c = conn.cursor()
    
    query_string = 'select face_id, roll, pitch, yaw from facepose where face_id <= 65384'

    names = []
    labels = []
    images = []
    paths = []

    for row in c.execute(query_string):
        
        name = row[0]

        input_path = imagepath + str(name) + '.jpg'

        if(os.path.isfile(input_path)  == True):
            
            if ram:
                image = cv2.imread(input_path)

            roll   = row[1]
            pitch  = row[2]
            yaw    = row[3]
            label = rpy_to_unit_vector((roll, pitch, yaw))

            if counter % 1000 == 0:
                print ("Counter: " + str(counter))

            counter = counter + 1 
            
            names.append(name)
            labels.append(np.array(label, dtype = np.float32))
            if ram:
                images.append(image)
            paths.append(input_path)

        else:
            raise ValueError('Error: I cannot find the file specified: ' + str(input_path))

        if num_data != None:
            if counter > num_data:
                break

    c.close()
    
    names, labels, images, paths = np.array(names), np.array(labels), np.array(images), np.array(paths)
    
    return names, labels, images, paths

def get_generator(data, idxs, resolution, batch_size, ram,
                  shuffle = True,
                  shift = False,
                  flip = False,
                  rotate = False,
                  cutout = False,
                  ):
    (all_names, all_labels, all_images, all_paths) = data
 
    ppf, width_shift, height_shift = None, 0.0, 0.0
    if shift:
        width_shift, height_shift = 0.2, 0.2
    if cutout:
        ppf = get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3,
                      v_l=0.0, v_h=1.0, pixel_level=True)

    names, labels, images = all_names[idxs], all_labels[idxs], all_images[idxs]
    
    datagen = ImageDataGenerator(
            rescale=1./255,
            width_shift_range = width_shift,
            height_shift_range = height_shift,
            preprocessing_function = ppf,
            )
    

    gen = datagen.flow(images, labels, batch_size=batch_size, shuffle=shuffle)

    if flip:
        gen = flip_gen(gen)
        
    if rotate:
        gen = rotate_gen(gen)
    
    return gen

def flip_gen(image_generator, flip_p=0.5):
    for x, y in image_generator:
         h_flip_selector = np.random.binomial(1, flip_p, size=(x.shape[0])) == 1
         v_flip_selector = np.random.binomial(1, flip_p, size=(x.shape[0])) == 1
         x[h_flip_selector,:,:,:] = x[h_flip_selector,:,::-1,:]
         y[h_flip_selector, 1] = (-1) * y[h_flip_selector, 1]
         x[v_flip_selector,:,:,:] = x[v_flip_selector,::-1,:,:]
         y[v_flip_selector, 2] = (-1) * y[v_flip_selector, 2]
         yield x, y

def rotate_gen(image_generator, rot_range = 90):
    for x, y in image_generator:
         rot_degrees = np.random.uniform(-rot_range, rot_range, size = x.shape[0])
         for i in range(x.shape[0]):
             x[i] = rotate(x[i], rot_degrees[i], reshape=False)
             theta = np.deg2rad(rot_degrees[i])
             y[i][1], y[i][2] = y[i][1]*np.cos(theta) + y[i][2]*np.sin(theta), y[i][2]*np.cos(theta) - y[i][1]*np.sin(theta)
         yield x, y

def shift_gen(image_generator, shift_range = 0.2):
    for x, y in image_generator:
        resolution = x.shape[1]
        rnge = shift_range*resolution
        x_shift = np.random.uniform(-rnge, rnge, size = x.shape[0])
        y_shift = np.random.uniform(-rnge, rnge, size = x.shape[0])
        for i in range(x.shape[0]):
            x[i] = shift(x[i], [x_shift[i], y_shift[i], 0], mode='nearest', prefilter=False)
        yield x, y


def setup_sess(net_specs):
    global sess, net
    
    tf.reset_default_graph()
    
    if tf.test.is_gpu_available():
        device = "/gpu:0"
    else:
        device = "/cpu:0"
        
    (model, resolution, kwargs) = net_specs
    
    with tf.device(device):
        try:
            net = model(input_dims = (resolution, resolution, 3), **kwargs)
        except TypeError:
            net = model(resolution = resolution, **kwargs)
   
    config = tf.ConfigProto(inter_op_parallelism_threads=os.cpu_count(),
                            intra_op_parallelism_threads=os.cpu_count(),
                            allow_soft_placement=True,
                            log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    return sess, net


def run_batches(sess, net, gen, batches, batch_size, fit, verbose = None):
    losses, degs, alphas, kappas = [],[],[],[]
    batch_count = 0
    for x_batch, y_batch in gen:
        #Catch error due to small batch at end
        if len(x_batch) < batch_size:
            continue
        y_batch = np.stack(y_batch)
        if fit:
            d, l, a, k, out, loss, _, _ = sess.run([net.degrees, net.log_likelihood, net.alpha, net.kappa, net.output, net.loss, net.fit, net.model.updates], feed_dict={net.main_input: x_batch, net.targets: y_batch, net.is_training: True})
        else:
            d, l, a, k, out, loss = sess.run([net.degrees, net.log_likelihood, net.alpha, net.kappa, net.output, net.loss], feed_dict={net.main_input: x_batch, net.targets: y_batch})
        losses.append(loss)
        degs.append(d)
        alphas.append(a)
        kappas.append(k)
        if verbose != None:
            if batch_count % verbose == 0:
                print('\nBatch',batch_count)
                for i in range(len(d)):
                    print(d[i][0], '\t', k[i], '\t', a[i], '\t', -l[i])
            elif batch_count % 10 == 0:
                print(batch_count, end=' ')
        batch_count += 1
        if batch_count >= batches:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    loss = np.round(np.mean(losses), 4)
    degree = np.round(np.mean(degs), 4)
    weighted_kappas = np.sum(np.concatenate(alphas)*np.concatenate(kappas), axis = 1)
    corr = np.round(np.corrcoef(np.ravel(degs), np.ravel(weighted_kappas))[0][1], 4)
    wk = np.mean(weighted_kappas, axis=0)
    kappa = np.mean(np.concatenate(kappas), axis=0)
    alpha = np.mean(np.concatenate(alphas), axis=0)

    return loss, degree, kappa, alpha, wk, corr

def run_po_batches(sess, net, gen, batches, batch_size, fit, verbose = None):
    losses, degs = [],[]
    batch_count = 0
    for x_batch, y_batch in gen:
        #Catch error due to small batch at end
        if len(x_batch) < batch_size:
            continue
        y_batch = np.stack(y_batch)
        if fit:
            d, out, loss, _ = sess.run([net.degrees, net.output, net.loss, net.fit], feed_dict={net.main_input: x_batch, net.targets: y_batch, net.is_training: True})
        else:
            d, out, loss = sess.run([net.degrees, net.output, net.loss], feed_dict={net.main_input: x_batch, net.targets: y_batch})
        losses.append(loss)
        degs.append(d)
        if verbose != None:
            if batch_count % verbose == 0:
                print('\nBatch', batch_count)
                for i in range(len(d)):
                    print(d[i][0], '\t', loss[i])
            elif batch_count % 10 == 0:
                print(batch_count, end=' ')
        batch_count += 1
        if batch_count >= batches:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    loss = np.round(np.mean(losses), 4)
    degree = np.round(np.mean(degs), 4)

    return loss, degree

def run_training(resolution, epochs, batches, batch_size, train_gen, val_gen, point_only=False, plot=True):
    global sess, net
    
    train_losses, train_degrees, train_kappas, train_alphas, train_wks, train_corrs = [],[],[],[],[],[]
    val_losses, val_degrees, val_kappas, val_alphas, val_wks, val_corrs = [],[],[],[],[],[]
    (train_batches, val_batches) = batches
    
    patience = 50
    loss_threshold = 2.5
    loss_increase_threshold = 1.0
    deg_threshold = 85
    
    for epoch in range(epochs):
        print("Epoch:", epoch)
        print('Optimising...')
        
        if point_only:
            train_loss, train_degree = run_po_batches(
                sess, net, train_gen, train_batches, batch_size, fit=True, verbose = 1000)
            train_losses.append(train_loss)
            train_degrees.append(train_degree)
            print()
            print('Training loss: ',train_loss)
            print('Training degree error: ',train_degree)
        
        else:
            train_loss, train_degree, train_kappa, train_alpha, train_wk, train_corr = run_batches(
                    sess, net, train_gen, train_batches, batch_size, fit=True, verbose = 1000)
            train_losses.append(train_loss)
            train_degrees.append(train_degree)
            train_kappas.append(train_kappa)
            train_alphas.append(train_alpha)
            train_wks.append(train_wk)
            train_corrs.append(train_corr)
            print()
            print('Training loss: ',train_loss)
            print('Training degree error: ',train_degree)
            print("Training correlation:", train_corr)
        
        print('Calculating validation loss...')

        if point_only:
            val_loss, val_degree = run_po_batches(
                sess, net, val_gen, val_batches, batch_size, fit=False, verbose = 250)
            val_losses.append(val_loss)
            val_degrees.append(val_degree)
            print('Validation loss: ',val_loss)
            print('Validation degree error: ',val_degree)

        else:
            val_loss, val_degree, val_kappa, val_alpha, val_wk, val_corr = run_batches(
                    sess, net, val_gen, val_batches, batch_size, fit=False, verbose = 250)
            val_losses.append(val_loss)
            val_degrees.append(val_degree)
            val_kappas.append(val_kappa)
            val_alphas.append(val_alpha)
            val_wks.append(val_wk)
            val_corrs.append(val_corr)
            print('Validation loss: ',val_loss)
            print('Validation degree error: ',val_degree)
            print("Validation correlation:", val_corr)
        
        print()
        print("Training losses:",train_losses)
        print('Training degree errors:',train_degrees)

        if point_only == False:
            print("Training correlations:", train_corrs)
        print()
        print("Validation losses:",val_losses)
        print('Validation degree errors:',val_degrees)

        if point_only == False:
            print("Validation correlations:", val_corrs)
        print()
        
        #Early stopping if overfitting:
        if np.isnan(val_loss) or epoch - np.argmin(val_losses) >= patience or val_loss > loss_threshold or val_loss - min(val_losses) > loss_increase_threshold or (epoch >= patience and val_degree >= deg_threshold):
            break      
   
    if plot:
    
        plt.plot(train_losses, label = 'Training loss')
        plt.plot(val_losses, label = 'Validation loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()
        plt.plot(train_degrees, label = 'Training degree error')
        plt.plot(val_degrees, label = 'Validation degree error')
        plt.xlabel('Epochs')
        plt.legend()
        plt.show()

    print("Overall validation loss:", val_loss)
    print("Overall validation degree error:", val_degree)
    
    return train_losses, val_losses, train_degrees, val_degrees, val_loss, val_degree

def run_one(val_gen):
    global sess, net
    
    x, y = next(val_gen)
    im = np.expand_dims(x[0], 0)
    out, ll = sess.run([net.point_est, net.point_est_log_likelihood], feed_dict = {net.main_input: im})
    
def time_one(val_gen, reps = 10, number = 100):
    times = timeit.Timer(partial(run_one, val_gen)).repeat(reps, number)
    time_taken = min(times) / number
    return time_taken

def count_params():
    total_parameters = 0
    for variable in tf.trainable_variables():

        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
            
        total_parameters += variable_parameters
    return total_parameters    
    