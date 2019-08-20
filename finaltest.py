# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:32:33 2019

@author: rka1
"""

import tensorflow as tf
from models import InceptionV3
from common_functions import get_data, get_generator, run_batches
import pickle
import os

MAX_NUM_DATA = 24384

components, num_data = 10, None
validation_prop, test_prop = .2, .1

if num_data == None:
    num_data = MAX_NUM_DATA

ids=pickle.load(open('saves/ids/ids_shuffled.pkl', "rb"))

val_start = int((1 - (validation_prop + test_prop))*num_data)
test_start = int((1 - test_prop)*num_data)
train_ids, val_ids, test_ids = ids[:val_start], ids[val_start: test_start], ids[test_start:]

resolution, batch_size = 149, 32
network_dir_name = 'Inception-v3'

ram = True

names, labels, images, paths = get_data(resolution, ram, num_data)
data = (names, labels, images, paths)
num_data = len(names)

test_gen = get_generator(data, test_ids, resolution, batch_size, ram,
                        )

tf.reset_default_graph()

if tf.test.is_gpu_available():
    device = "/gpu:0"
else:
    device = "/cpu:0"
    
net = InceptionV3(resolution = resolution, components = components)
   
config = tf.ConfigProto(inter_op_parallelism_threads=os.cpu_count(),
                        intra_op_parallelism_threads=os.cpu_count(),
                        allow_soft_placement=True,
                        log_device_placement=True)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
net.load(sess, resolution, network_dir_name, 6)

test_batches = len(test_ids)//batch_size


loss, degree, kappa, alpha, wk, corr = run_batches(sess, net, test_gen, test_batches, batch_size, fit=False, dropout=False, verbose=None)

print("Test loss:", loss)
print("Test degree error:",degree)