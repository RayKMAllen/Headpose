# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 20:44:05 2019

@author: RayKMAllen
"""

import pickle
from matplotlib import pyplot as plt

#%%

#Augmentation results:

savepath = 'saves/augmentresults/augresults.pkl'
results = pickle.load(open(savepath + '_prob', 'rb'))
po_results = pickle.load(open(savepath + '_pointonly', 'rb'))


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
    plt.title(plot_titles[i] + ' (Probabilistic, k = 5)')
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
    plt.title(plot_labs[n] + ' (Probabilistic, k = 5)')
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
    

#%%
    
#Num components results:
    
savepath = 'saves/componentresults/compresults.pkl'
results = pickle.load(open(savepath, 'rb'))

components = [1,2,3,4,5,6,8,10,20]
plot_titles = ['Training loss', 'Validation loss', 'Training degree error',
                   'Validation degree error']
plot_labs = components

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

#%%
    
#Main results by network:

plot_titles = ['Training loss', 'Validation loss', 'Training degree error',
                   'Validation degree error']

resolution, version = 112, 0
savepath = 'saves/mainresults/mainresults_{}_{}.pkl'.format(resolution, version)
save_info = pickle.load(open(savepath, 'rb'))
[results, plot_labs, train_times, pred_times, num_params] = save_info

results = results[:5]


for i in range(len(plot_titles)):
    for n, j in enumerate(results):
        plt.plot(j[i], label = plot_labs[n])
    plt.xlabel('Epochs')
    plt.title(plot_titles[i] + ' (Probabilistic, k = 10)')
    plt.legend()
    plt.grid()
    plt.show()

for n, r in enumerate(results):
    plt.plot(r[0], label = 'Training loss')
    plt.plot(r[1], label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.title(plot_labs[n] + ' (Probabilistic, k = 10)')
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
    
#%%

#Main results by resolution:

resolutions = [32, 64, 112]

version = 0
savepath = 'saves/mainresults/mainresults_{}_{}.pkl'

allresults = []
for res in resolutions:
    save_info = pickle.load(open(savepath.format(res, version), 'rb'))
    [results, plot_labs, train_times, pred_times, num_params] = save_info
    results = results[:5]
    allresults.append(results)
    
vallosses = []
for i in range(5):
    for j in range(3):
        lab = resolutions[j]
        if lab == 112 and i == 4:
            lab = 149
        plt.plot(allresults[j][i][1], label = lab)
    plt.xlabel('Epochs')
    plt.title(plot_labs[i])
    plt.legend()
    plt.grid()
    plt.show()
