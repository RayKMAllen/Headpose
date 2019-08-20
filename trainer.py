# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:29:33 2019

@author: RayKMAllen
"""

import tensorflow as tf
import numpy as np

from models import LeNet, VGG16, ResNet34, GoogLeNet, InceptionV3
from common_functions import get_data, get_generator, setup_sess, run_training, time_one, count_params
import pickle
import time
from matplotlib import pyplot as plt
np.set_printoptions(suppress=True)

MAX_NUM_DATA = 24384


networks = [LeNet, VGG16, ResNet34, GoogLeNet, InceptionV3]

epochs, components, ram, num_data = 100, 10, True, None
version = 1
validation_prop, test_prop = .2, .1

if __name__ == "__main__":
    
    if num_data == None:
        num_data = MAX_NUM_DATA
    
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
    
    for res in [32, 64, 112]:
    
        results = []
        plot_labs = []
        train_times = []
        pred_times = []
        num_params = []
    
        resolution, batch_size = res, 32
    
        names, labels, images, paths = get_data(resolution, ram, num_data)
        data = (names, labels, images, paths)
        num_data = len(names)
    
        train_gen = get_generator(data, train_ids, resolution, batch_size, ram, 
                                  shift=True,
                                  flip=True,
                                  rotate=True,
                                  cutout=True
                                  )
        val_gen = get_generator(data, val_ids, resolution, batch_size, ram,
                                )
        
        for n, nw in enumerate(networks):
    
            try:
                
                if nw == InceptionV3 and res == 112:
                    resolution = 149
                    batch_size = 16
                
                    names, labels, images, paths = get_data(resolution, ram, num_data)
                    data = (names, labels, images, paths)
                    num_data = len(names)                
                
                    train_gen = get_generator(data, train_ids, resolution, batch_size, ram, 
                                              shift=True,
                                              flip=True,
                                              rotate=True,
                                              cutout=True
                                              )
                    val_gen = get_generator(data, val_ids, resolution, batch_size, ram,
                                            )
                    
                batches = (len(train_ids)//batch_size, len(val_ids)//batch_size)
                
                savepath = 'saves/mainresults/mainresults_{}_{}.pkl'.format(resolution, version)

                specs = (nw, resolution, {'components': components})
                start = time.time()
                sess, net = setup_sess(specs)
                print("Network:", net.name)
                print("Resolution:",resolution)
                result = run_training(resolution, epochs, batches, batch_size, train_gen, val_gen)
                end = time.time()
                results.append(result)
                plot_labs.append(net.name)
                train_times.append(end - start)
                pred_times.append(time_one(val_gen))
                num_params.append(count_params())
            
                save_info = [results, plot_labs, train_times, pred_times, num_params]
                pickle.dump(save_info, open(savepath, "wb"))
                net.save(sess, resolution, net.name, version = version)
                
            except Exception as err:
                print(err)
        
    plot_titles = ['Training loss', 'Validation loss', 'Training degree error',
                   'Validation degree error']

    
    for i in range(len(plot_titles)):
        for n, j in enumerate(results):
            plt.plot(j[i], label = plot_labs[n])
        plt.xlabel('Epochs')
        plt.title(plot_titles[i] + ' (Probabilistic, k = {})'.format(components))
        plt.legend()
        plt.grid()
        plt.show()
    
    for n, r in enumerate(results):
        plt.plot(r[0], label = 'Training loss')
        plt.plot(r[1], label = 'Validation loss')
        plt.xlabel('Epochs')
        plt.title(plot_labs[n] + ' (Probabilistic, k = {})'.format(components))
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