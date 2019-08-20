# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:11:09 2019

@author: RayKMAllen
"""

import math
import numpy as np
import sys
from tensorflow.keras.layers import Lambda
import tensorflow as tf
from types import ModuleType, FunctionType
from gc import get_referents
import operator as op
from functools import reduce
from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt

def rpy_to_unit_vector(rpy):
    #Converts roll, pitch, yaw triple to vector on the sphere.
    #Note roll is actually irrelevant.
    (roll, pitch, yaw) = rpy
    
    x = math.cos(yaw)*math.cos(pitch)
    y = math.sin(yaw)*math.cos(pitch)
    z = math.sin(pitch)
    
    return (x, y, z)


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

class Logger(object):
    def __init__(self):
        self.stdout = sys.stdout
        self.terminal = sys.stdout
        self.log = open("log.txt", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass  
    
    def __del__(self):
        sys.stdout = self.stdout
        self.log.close()
        
def kerasize(tf_func, input, *args, **kwargs):
    return Lambda(lambda x: tf_func(x, *args, **kwargs))(input)

def create_weight(name, shape, initializer=None, trainable=True, seed=None):
  if initializer is None:
    initializer = tf.contrib.keras.initializers.he_normal(seed=seed)
  return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def getsize(obj):
    """sum size of object & members."""
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            try:
                if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            except ReferenceError:
                pass
        objects = get_referents(*need_referents)
    return size

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)

def excavate(dct, lst):
    for k, v in dct.items():
        if type(v) is dict:
            excavate(v, lst)
        else:
            lst.append(v)
    return lst
    
def yawpitchtomaae(yaw, pitch):
    
    phi = np.deg2rad(yaw)
    theta = np.deg2rad(pitch)
    xrad = np.cos(phi)*np.cos(theta)

    return np.rad2deg(np.arccos(xrad))

def nnelu(inpt):
    #Computes the Non-Negative Exponential Linear Unit
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(inpt))

def imshow(ary):
    im = ary*255
    im = im.astype(np.uint8)
    display(Image.fromarray(im))

def vecshow(vec):    
    plt.figure(figsize=(1,1))
    ax = plt.axes()
    
    ax.arrow(0, 0, -vec[1], -vec[0], head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.arrow(0, 0, -vec[1], vec[2], head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.show()
    









