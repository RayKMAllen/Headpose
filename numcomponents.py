# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:57:18 2019

@author: RayKMAllen
"""

import tensorflow as tf
import numpy as np
from models import LeNet
from matplotlib import pyplot as plt
import pickle
from common_functions import get_data, get_generator, setup_sess, run_training

np.set_printoptions(suppress = True, precision = 8)
MAX_NUM_DATA = 24384

resolution, batch_size, ram, num_data = 32, 32, True, None
validation_prop, test_prop = .2, .1

if num_data == None:
    num_data = MAX_NUM_DATA

names, labels, images, paths = get_data(resolution, ram, num_data)
data = (names, labels, images, paths)

num_data = len(names)
try:
    ids=pickle.load(open('saves/ids/ids_shuffled.pkl', "rb"))
    print("Loading old IDs...")
except FileNotFoundError:
    print("Randomly choosing new IDs...")
    ids=np.random.permutation(num_data)
    pickle.dump(ids, open('saves/ids/ids_shuffled.pkl', "wb"))

val_start = int((1 - (validation_prop + test_prop))*num_data)
test_start = int((1 - test_prop)*num_data)
train_ids, val_ids, test_ids = ids[:val_start], ids[val_start: test_start], ids[test_start:]



train_gen = get_generator(data, train_ids, resolution, batch_size, ram, 
                          shift=True,
#                          flip=True,
#                          rotate=True,
                          cutout=True
                          )
val_gen = get_generator(data, val_ids, resolution, batch_size, ram)

batches = (len(train_ids)//batch_size, len(val_ids)//batch_size)


components = [1,2,3,4,5,6,8,10,20]
results = []
epochs = 100


for cnum in components:
    print("Components:", cnum)
    specs = (LeNet, resolution, {'components':cnum})
    sess, net = setup_sess(specs)
    train_losses, val_losses, train_degrees, val_degrees, val_loss, val_degree = run_training(resolution, epochs, batches, batch_size, train_gen, val_gen)
    train_res = [train_losses, train_degrees]
    val_res = [val_losses, val_degrees]
    res = [train_res, val_res, val_loss, val_degree]
    results.append(res)


plot_titles = ['Training loss', 'Validation loss', 'Training degree error',
                   'Validation degree error']
plot_labs = components

    
savepath = 'saves/componentresults/compresults.pkl'
pickle.dump(results, open(savepath, "wb"))
pickle.dump(ids, open('saves/componentresults/ids.pkl', "wb"))


for i in range(len(plot_titles)):
    for n, j in enumerate(results):
        if i % 2 == 0:
            plt.plot(j[0][i//2], label = plot_labs[n])
        else:
            plt.plot(j[1][i//2], label = plot_labs[n])
    plt.xlabel('Epochs')
    plt.title(plot_titles[i])
    plt.legend()
    plt.grid()
    plt.show()
