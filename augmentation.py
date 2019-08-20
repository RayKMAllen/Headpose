# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:01:24 2019

@author: RayKMAllen
"""

import numpy as np
from models import VGG16
from matplotlib import pyplot as plt
import pickle
from common_functions import get_data, get_generator, setup_sess, run_training

np.set_printoptions(suppress = True, precision = 8)
MAX_NUM_DATA = 24384
    
resolution, batch_size, ram, num_data = 32, 32, True, None
validation_prop, test_prop = .2, .1
epochs = 100

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


val_gen = get_generator(data, val_ids, resolution, batch_size, ram,
                        )

batches = (len(train_ids)//batch_size, len(val_ids)//batch_size)

train_gens = []
train_gens.append(get_generator(data, train_ids, resolution, batch_size, ram))
train_gens.append(get_generator(data, train_ids, resolution, batch_size, ram, shift=True))
train_gens.append(get_generator(data, train_ids, resolution, batch_size, ram, flip=True))
train_gens.append(get_generator(data, train_ids, resolution, batch_size, ram, rotate=True))
train_gens.append(get_generator(data, train_ids, resolution, batch_size, ram, cutout=True))
train_gens.append(get_generator(data, train_ids, resolution, batch_size, ram, 
                          shift=True,
                          flip=True,
                          rotate=True,
                          cutout=True
                          ))

savepath = 'saves/augmentresults/augresults.pkl'

po_results = []
for n, tg in enumerate(train_gens):
    
    print("Version",n)

    specs = (VGG16, resolution, {'point_only': True})
    sess, net = setup_sess(specs)
    res = run_training(resolution, epochs, batches, batch_size, tg, val_gen, point_only=True)
    po_results.append(res)
    
    pickle.dump(po_results, open(savepath + '_pointonly', "wb"))

results = []
for n, tg in enumerate(train_gens):
    
    print("Version",n)

    specs = (VGG16, resolution, {'components':5})
    sess, net = setup_sess(specs)
    res = run_training(resolution, epochs, batches, batch_size, tg, val_gen)
    results.append(res)

    pickle.dump(results, open(savepath + '_prob', "wb"))

plot_titles = ['Training loss', 'Validation loss', 'Training degree error',
                   'Validation degree error']
plot_labs = ['None', 'Shift', 'Flip', 'Rotate', 'Cutout', 'All']

for i in range(len(plot_titles)):
    for n, j in enumerate(po_results):
        plt.plot(j[i], label = plot_labs[n])
    plt.xlabel('Epochs')
    plt.title(plot_titles[i] + ' (Point only)')
    plt.legend()
    plt.grid()
    plt.show()

for i in range(len(plot_titles)):
    for n, j in enumerate(results):
        plt.plot(j[i], label = plot_labs[n])
    plt.xlabel('Epochs')
    plt.title(plot_titles[i] + ' (Probabilistic, k = 3)')
    plt.legend()
    plt.grid()
    plt.show()

for n, r in enumerate(po_results):
    plt.plot(r[2], label = 'Training degree error')
    plt.plot(r[3], label = 'Validation degree error')
    plt.xlabel('Epochs')
    plt.title(plot_labs[n] + ' (Point only)')
    plt.legend()
    plt.grid()
    plt.show()

for n, r in enumerate(results):
    plt.plot(r[0], label = 'Training loss')
    plt.plot(r[1], label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.title(plot_labs[n] + ' (Probabilistic, k = 3)')
    plt.legend()
    plt.grid()
    plt.show()

for n, r in enumerate(results):
    plt.plot(r[2], label = 'Training degree error')
    plt.plot(r[3], label = 'Validation degree error')
    plt.xlabel('Epochs')
    plt.title(plot_labs[n])
    plt.legend()
    plt.grid()
    plt.show()