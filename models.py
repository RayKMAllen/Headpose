# -*- coding: utf-8 -*-
"""
Created on Thu May 23 16:27:24 2019

@author: RayKMAllen
"""

import tensorflow as tf
import numpy as np
import math
import tensorflow_probability as tfp
tfd = tfp.distributions

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, MaxPooling2D, add, concatenate, AveragePooling2D, GlobalAveragePooling2D, Lambda, Dropout, Softmax
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from adabound.AdaBound import AdaBoundOptimizer
from tensorflow_probability import distributions as tfd
import os
from utils import nnelu

RADDEG = 180/math.pi
MAX_NUM_DATA = 24384


class Network:
    
    def __init__(self, input_dims, num_outputs, components, layer_sizes, kernel_size, reg_constant, learning_rate, optimizer, momentum = 0.9, epsilon = 1e-08, point_only = False, name = None):
        self.input_dims = input_dims
        self.num_outputs = num_outputs
        self.main_input = Input(shape = self.input_dims, name = 'main_input')
        self.targets = tf.placeholder(tf.float32,[None,self.num_outputs])
        self.components = components
        self.layer_sizes = layer_sizes
        self.kernel_size = kernel_size
        self.reg_const = reg_constant
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.epsilon = epsilon
        self.is_training = tf.placeholder_with_default(False, (), 'is_training')
        self.name = name
        
        self._build_model()
        self.model = Model(inputs = self.main_input, outputs = self.pre_output)
        
        if point_only == False:
            self._build_outputs()
            
            gm = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=self.alpha),
                components_distribution=tfd.VonMisesFisher(
                    mean_direction = self.output,       
                    concentration = self.kappa))
            self.log_likelihood = gm.log_prob(self.targets)
            
            self.loss = -self.log_likelihood            
            
            self._get_point_est(gm)
            self.point_est_log_likelihood = gm.log_prob(self.point_est)
            self.degrees = tf.multiply(tf.acos(tf.reduce_sum( tf.multiply( self.point_est, self.targets ), 1, keepdims=True )), RADDEG)
        
        else:
            self._build_point_only()
            
        if self.name == 'GoogLeNet':
            self._weighted_loss()
        
        self.opt = self._get_optimizer()
        self.fit = self.opt.minimize(self.loss)
        
        self.saver = tf.train.Saver()
        
    def _get_point_est(self, gm, n_samples = 1000):
        
        if self.components > 1:
            samples = gm.sample(n_samples)
            samples = tf.transpose(samples, [1, 0, 2])
            cossim = tf.matmul(samples, samples, transpose_b = True)
            cossim_clipped = tf.clip_by_value(cossim, clip_value_min = -1, clip_value_max = 1)
            dist = tf.acos(cossim_clipped)
            self.exp_err = tf.reduce_mean(dist, axis=-1)
            self.min_err_idx = tf.argmin(self.exp_err, output_type = tf.int32, axis=-1)
            self.point_est = tf.gather_nd(samples, tf.stack([tf.range(tf.shape(self.min_err_idx)[0]), self.min_err_idx], 1))
        else:
            self.point_est = tf.reduce_mean(self.output, axis = -2)


    def _get_optimizer(self):

        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate = self.learning_rate, epsilon = self.epsilon)
        elif self.optimizer == 'sgd-nesterov':
            return tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = self.momentum, use_nesterov = True)
        elif self.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate = self.learning_rate, momentum = self.momentum, epsilon = self.epsilon)
        elif self.optimizer == 'nadam':
            return tf.contrib.opt.NadamOptimizer(learning_rate = self.learning_rate, epsilon = self.epsilon)
        elif self.optimizer == 'adabound':
            return AdaBoundOptimizer(learning_rate=self.learning_rate, final_lr=0.1, beta1=0.9, beta2=0.999, amsbound=False)


    def save(self, sess, resolution, name, version = 0):
        savepath = 'saves/models/{}/{}/version-'.format(resolution, name)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.saver.save(sess, savepath, global_step = version)

    def load(self, sess, resolution, name, version):
        filepath = 'saves/models/{}/{}/version--{}'.format(resolution, name, version)
        self.saver.restore(sess, filepath)

        
    def conv_layer(self, x, filters, kernel_size, batch_norm = False, max_pool = False, padding = 'same', stride=1):

        x = Conv2D(
            filters = filters
            , kernel_size = kernel_size
            , strides = stride
            , data_format="channels_last"
            , padding = padding
            , kernel_initializer = 'he_normal'
            , kernel_regularizer = regularizers.l2(self.reg_const)
            )(x)

        if batch_norm:
            x = BatchNormalization(axis=-1)(x, training=self.is_training)
            
        x = LeakyReLU()(x)

        if max_pool:
            x = MaxPooling2D(padding = 'same')(x)

        return x

    def dense_layer(self, x, neurons, activation):
        
        x = Dense(
            neurons
            , kernel_initializer = 'he_normal'
            , kernel_regularizer=regularizers.l2(self.reg_const)
            )(x)

        if activation:
            x = LeakyReLU()(x)
            
        return x
    
    def identity_block(self, input_block, num_layers, kernel_size):
        num_channels = input_block.get_shape().as_list()[-1]
        
        x = self.conv_layer(input_block, num_channels, kernel_size, True, False)
        
        if num_layers == 3:
            x = self.conv_layer(x, num_channels, kernel_size, True, False)
            
        x = Conv2D(
            filters = num_channels
            , kernel_size = kernel_size
            , data_format="channels_last"
            , padding = 'same'
            , kernel_initializer = 'he_normal'
            , kernel_regularizer = regularizers.l2(self.reg_const)
            )(x)

        x = BatchNormalization(axis=-1)(x, training=self.is_training)

        x = add([input_block, x])

        x = LeakyReLU()(x)
        
        return x
        
    def convolutional_block(self, input_block, num_layers, filters, kernel_size, stride):
        in_channels = input_block.get_shape().as_list()[-1]
        
        #Residual:
        x = self.conv_layer(input_block, filters, kernel_size, True, False, stride=stride)
        
        if num_layers == 3:
            x = self.conv_layer(x, filters, kernel_size, True, False)
            
        x = Conv2D(
            filters = filters
            , kernel_size = kernel_size
            , data_format="channels_last"
            , padding = 'same'
            , kernel_initializer = 'he_normal'
            , kernel_regularizer = regularizers.l2(self.reg_const)
            )(x)

        x = BatchNormalization(axis=-1)(x, training=self.is_training)
        
        #Shortcut
        if in_channels == filters:
            skip_block = tf.identity(input_block)    
        else:
            skip_block = Conv2D(
                filters = filters
                , kernel_size = 1          
                , strides = stride
                , data_format="channels_last"
                , padding = 'same'
                , kernel_initializer = 'he_normal'
                , kernel_regularizer = regularizers.l2(self.reg_const)
                )(input_block)
    
            skip_block = BatchNormalization(axis=-1)(skip_block, training=self.is_training)

        x = add([skip_block, x])

        x = LeakyReLU()(x)
        
        return x

    def inception_module(self, x,
                         filters_1x1,
                         filters_3x3_reduce,
                         filters_3x3,
                         filters_5x5_reduce,
                         filters_5x5,
                         filters_pool_proj):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1))
        
        conv_3x3 = self.conv_layer(x, filters_3x3_reduce, (1,1))
        conv_3x3 = self.conv_layer(conv_3x3, filters_3x3, (3,3))
    
        conv_5x5 = self.conv_layer(x, filters_5x5_reduce, (1,1))
        conv_5x5 = self.conv_layer(conv_5x5, filters_5x5, (5,5))
    
        pool_proj = MaxPooling2D(3, 1, padding='same')(x)
        pool_proj = self.conv_layer(pool_proj, filters_pool_proj, (1,1))
    
        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
        
        return output
    
    def incep_a(self, x,
                 filters_1x1,
                 filters_5x5_reduce,
                 filters_5x5,
                 filters_3x3dbl_reduce,
                 filters_3x3dbl_1,
                 filters_3x3dbl_2,
                 filters_pool_proj):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1), True)
    
        conv_5x5 = self.conv_layer(x, filters_5x5_reduce, (1,1), True)
        conv_5x5 = self.conv_layer(conv_5x5, filters_5x5, (5,5), True)
        
        conv_3x3dbl = self.conv_layer(x, filters_3x3dbl_reduce, (1,1), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_1, (3,3), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_2, (3,3), True)
        
        pool_proj = AveragePooling2D(3, 1, padding='same')(x)
        pool_proj = self.conv_layer(pool_proj, filters_pool_proj, (1,1), True)
    
        output = concatenate([conv_1x1, conv_5x5, conv_3x3dbl, pool_proj], axis=3)
        
        return output

    def incep_b(self, x,
                 filters_1x1,
                 filters_7x7_reduce,
                 filters_7x7_1,
                 filters_7x7_2,
                 filters_7x7dbl_reduce,
                 filters_7x7dbl_1,
                 filters_7x7dbl_2,
                 filters_7x7dbl_3,
                 filters_7x7dbl_4,
                 filters_pool_proj):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1), True)
    
        conv_7x7 = self.conv_layer(x, filters_7x7_reduce, (1,1), True)
        conv_7x7 = self.conv_layer(conv_7x7, filters_7x7_1, (7,1), True)
        conv_7x7 = self.conv_layer(conv_7x7, filters_7x7_2, (1,7), True)
        
        conv_7x7dbl = self.conv_layer(x, filters_7x7dbl_reduce, (1,1), True)
        conv_7x7dbl = self.conv_layer(conv_7x7dbl, filters_7x7dbl_1, (7,1), True)
        conv_7x7dbl = self.conv_layer(conv_7x7dbl, filters_7x7dbl_2, (1,7), True)
        conv_7x7dbl = self.conv_layer(conv_7x7dbl, filters_7x7dbl_3, (7,1), True)
        conv_7x7dbl = self.conv_layer(conv_7x7dbl, filters_7x7dbl_4, (1,7), True)
        
        pool_proj = AveragePooling2D(3, 1, padding='same')(x)
        pool_proj = self.conv_layer(pool_proj, filters_pool_proj, (1,1), True)
    
        output = concatenate([conv_1x1, conv_7x7, conv_7x7dbl, pool_proj], axis=3)
        
        return output

    def incep_c(self, x,
                 filters_1x1,
                 filters_3x3_reduce,
                 filters_3x3,
                 filters_3x3dbl_reduce,
                 filters_3x3dbl_1,
                 filters_3x3dbl_2,
                 filters_pool_proj):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1), True)
    
        conv_3x3 = self.conv_layer(x, filters_3x3_reduce, (1,1), True)
        conv_3x3_1 = self.conv_layer(conv_3x3, filters_3x3, (1,3), True)
        conv_3x3_2 = self.conv_layer(conv_3x3, filters_3x3, (3,1), True)
        
        conv_3x3dbl = self.conv_layer(x, filters_3x3dbl_reduce, (1,1), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_1, (3,3), True)
        conv_3x3dbl_1 = self.conv_layer(conv_3x3dbl, filters_3x3dbl_2, (1,3), True)
        conv_3x3dbl_2 = self.conv_layer(conv_3x3dbl, filters_3x3dbl_2, (3,1), True)
        
        pool_proj = AveragePooling2D(3, 1, padding='same')(x)
        pool_proj = self.conv_layer(pool_proj, filters_pool_proj, (1,1), True)
    
        output = concatenate([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3dbl_1, conv_3x3dbl_2, pool_proj], axis=3)
        
        return output
    
    def incep_reduction_a(self, x,
                          filters_3x3,
                          filters_3x3dbl_reduce,
                          filters_3x3dbl_1,
                          filters_3x3dbl_2,
                          padding = 'valid'):
    
        conv_3x3 = self.conv_layer(x, filters_3x3, (3,3), True, stride=2, padding=padding)
        
        conv_3x3dbl = self.conv_layer(x, filters_3x3dbl_reduce, (1,1), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_1, (3,3), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_2, (3,3), True, stride=2, padding=padding)
        
        pool_proj = MaxPooling2D(3, 2, padding=padding)(x)
    
        output = concatenate([conv_3x3, conv_3x3dbl, pool_proj], axis=3)
        
        return output

    def incep_reduction_b(self, x,
                          filters_3x3_reduce,
                          filters_3x3,
                          filters_7x7x3_reduce,
                          filters_7x7x3_1,
                          filters_7x7x3_2,
                          filters_7x7x3_3,
                          padding = 'valid'):
    
        conv_3x3 = self.conv_layer(x, filters_3x3_reduce, (1,1), True)
        conv_3x3 = self.conv_layer(conv_3x3, filters_3x3, (3,3), True, stride=2, padding=padding)
        
        conv_7x7x3 = self.conv_layer(x, filters_7x7x3_reduce, (1,1), True)
        conv_7x7x3 = self.conv_layer(conv_7x7x3, filters_7x7x3_1, (1,7), True)
        conv_7x7x3 = self.conv_layer(conv_7x7x3, filters_7x7x3_2, (7,1), True)
        conv_7x7x3 = self.conv_layer(conv_7x7x3, filters_7x7x3_3, (3,3), True, stride=2, padding=padding)
        
        pool_proj = MaxPooling2D(3, 2, padding=padding)(x)
    
        output = concatenate([conv_3x3, conv_7x7x3, pool_proj], axis=3)
        
        return output

    def inc_res_a(self, x,
                 filters_1x1,
                 filters_3x3_reduce,
                 filters_3x3,
                 filters_3x3dbl_reduce,
                 filters_3x3dbl_1,
                 filters_3x3dbl_2,
                 scale):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1), True)
    
        conv_3x3 = self.conv_layer(x, filters_3x3_reduce, (1,1), True)
        conv_3x3 = self.conv_layer(conv_3x3, filters_3x3, (3,3), True)
        
        conv_3x3dbl = self.conv_layer(x, filters_3x3dbl_reduce, (1,1), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_1, (3,3), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_2, (3,3), True)

        concat = concatenate([conv_1x1, conv_3x3, conv_3x3dbl], axis=3)
        
        filters_proj = x.shape[-1].value
        up = self.conv_layer(concat, filters_proj, (1,1), True)
    
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                          arguments={'scale': scale})([x, up])
        output = LeakyReLU()(x)
        
        return output

    def inc_res_b(self, x,
                 filters_1x1,
                 filters_7x7_reduce,
                 filters_7x7_1,
                 filters_7x7_2,
                 scale):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1), True)
    
        conv_7x7 = self.conv_layer(x, filters_7x7_reduce, (1,1), True)
        conv_7x7 = self.conv_layer(conv_7x7, filters_7x7_1, (1,7), True)
        conv_7x7 = self.conv_layer(conv_7x7, filters_7x7_2, (7,1), True)

        concat = concatenate([conv_1x1, conv_7x7], axis=3)
        
        filters_proj = x.shape[-1].value
        up = self.conv_layer(concat, filters_proj, (1,1), True)
    
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                          arguments={'scale': scale})([x, up])

        output = LeakyReLU()(x)
        
        return output

    def inc_res_c(self, x,
                 filters_1x1,
                 filters_3x3_reduce,
                 filters_3x3_1,
                 filters_3x3_2,
                 scale,
                 activation = True):
        
        conv_1x1 = self.conv_layer(x, filters_1x1, (1,1), True)
    
        conv_3x3 = self.conv_layer(x, filters_3x3_reduce, (1,1), True)
        conv_3x3 = self.conv_layer(conv_3x3, filters_3x3_1, (1,3), True)
        conv_3x3 = self.conv_layer(conv_3x3, filters_3x3_2, (3,1), True)

        concat = concatenate([conv_1x1, conv_3x3], axis=3)
        
        filters_proj = x.shape[-1].value
        up = self.conv_layer(concat, filters_proj, (1,1), True)
    
        x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                          arguments={'scale': scale})([x, up])

        if activation:
            output = LeakyReLU()(x)
        else:
            output = x
        
        return output

    def inc_res_reduc_b(self, x,
                          filters_3x3_1_reduce,
                          filters_3x3_1,
                          filters_3x3_2_reduce,
                          filters_3x3_2,
                          filters_3x3dbl_reduce,
                          filters_3x3dbl_1,
                          filters_3x3dbl_2,
                          padding = 'valid'):
        
        conv_3x3_1 = self.conv_layer(x, filters_3x3_1_reduce, (1,1), True)
        conv_3x3_1 = self.conv_layer(conv_3x3_1, filters_3x3_1, (3,3), True, stride=2, padding=padding)

        conv_3x3_2 = self.conv_layer(x, filters_3x3_2_reduce, (1,1), True)
        conv_3x3_2 = self.conv_layer(conv_3x3_2, filters_3x3_2, (3,3), True, stride=2, padding=padding)
        
        conv_3x3dbl = self.conv_layer(x, filters_3x3dbl_reduce, (1,1), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_1, (3,3), True)
        conv_3x3dbl = self.conv_layer(conv_3x3dbl, filters_3x3dbl_2, (3,3), True, stride=2, padding=padding)
        
        pool_proj = MaxPooling2D(3, 2, padding=padding)(x)
    
        output = concatenate([conv_3x3_1, conv_3x3_2, conv_3x3dbl, pool_proj], axis=3)
        
        return output

    def _build_outputs(self):
                
        self.prekappa = Dense(self.components)(self.pre_output)
        self.kappa = nnelu(self.prekappa)

        self.prealpha = Dense(self.components, kernel_initializer = 'zeros')(self.pre_output)
        self.alpha = Softmax()(self.prealpha)
        self.alpha = tf.clip_by_value(self.alpha, clip_value_min = 1e-10, clip_value_max = 1)
        
        output = []
        for i in range(self.components):
            output.append(tf.nn.l2_normalize(self.dense_layer(self.pre_output, self.num_outputs, False), axis = -1))
        self.output = tf.stack(output, axis = -2)
        
    def _build_aux_outputs(self, num_aux_outs):
        
        self.aux_kappa_1 = nnelu(self.dense_layer(self.aux_pre_output_1, self.components, False))
        self.aux_alpha_1 = Softmax()(self.dense_layer(self.aux_pre_output_1, self.components, False))
        
        output_1 = []
        for i in range(self.components):
            output_1.append(tf.nn.l2_normalize(self.dense_layer(self.aux_pre_output_1, self.num_outputs, False), axis = -1))
        self.aux_output_1 = tf.stack(output_1, axis = -2)

        gm1 = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self.aux_alpha_1),
            components_distribution=tfd.VonMisesFisher(
                mean_direction = self.aux_output_1,       
                concentration = self.aux_kappa_1))
        self.aux_ll_1 = gm1.log_prob(self.targets)
        
        self.aux_loss_1 = -tf.reduce_mean(self.aux_ll_1, axis = -1)

        if num_aux_outs == 2:

            self.aux_kappa_2 = nnelu(self.dense_layer(self.aux_pre_output_2, self.components, False))
            self.aux_alpha_2 = Softmax()(self.dense_layer(self.aux_pre_output_2, self.components, False))
            
            output_2 = []
            for i in range(self.components):
                output_2.append(tf.nn.l2_normalize(self.dense_layer(self.aux_pre_output_2, self.num_outputs, False), axis = -1))
            self.aux_output_2 = tf.stack(output_2, axis = -2)    
    
            gm2 = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=self.aux_alpha_2),
                components_distribution=tfd.VonMisesFisher(
                    mean_direction = self.aux_output_2,       
                    concentration = self.aux_kappa_2))
            self.aux_ll_2 = gm2.log_prob(self.targets)
            
            self.aux_loss_2 = -tf.reduce_mean(self.aux_ll_2, axis = -1)

    def _build_point_only(self):
            
        self.output = tf.nn.l2_normalize(self.dense_layer(self.pre_output, self.num_outputs, False), axis = -1)
        self.point_est = self.output
        
        self.cos_sim = tf.reduce_sum( tf.multiply( self.point_est, self.targets ), 1, keepdims=True )
        self.loss = 1 - self.cos_sim
        self.degrees = tf.multiply(tf.acos(self.cos_sim), RADDEG)

        

class LeNet(Network):
    
    def __init__(self, resolution, num_outputs = 3, components = 1, kernel_size = (5,5), layer_sizes = (32,32,64,256), reg_constant = .01, learning_rate = 0.0001, optimizer = 'adam', momentum = 0.9, epsilon = 1e-08, bn = False, point_only = False):
        self.resolution = resolution
        input_dims = (resolution, resolution, num_outputs)
        Network.__init__(self, input_dims, num_outputs, components, layer_sizes, kernel_size, reg_constant, learning_rate, optimizer, momentum, epsilon, point_only=point_only, name = 'LeNet-5')
        

    def _build_model(self):
        
        x = self.main_input
        
        x = self.conv_layer(x, self.layer_sizes[0], self.kernel_size, False, True)
        x = self.conv_layer(x, self.layer_sizes[1], self.kernel_size, False, True)
        x = self.conv_layer(x, self.layer_sizes[2], self.kernel_size, False, False)
        
        x = Flatten()(x)
        
        x = self.dense_layer(x, self.layer_sizes[3], True)

        self.pre_output = x


class VGG16(Network):
    
    def __init__(self, resolution, num_outputs = 3, components = 1, layer_sizes = (64,64,128,128,256,256,256,512,512,512,512,512,512,256,256), kernel_size = (3,3), reg_constant = .01, learning_rate = 0.0001, optimizer = 'adam', momentum = 0.9, epsilon = 1e-08, use_bn = False, point_only = False):
        if resolution not in [32, 64, 112, 224]:
            raise Exception("Invalid resolution: must be one of 32, 64, 112, 224")
        self.resolution = resolution
        input_dims = (resolution, resolution, num_outputs)
        self.use_bn = use_bn
        Network.__init__(self, input_dims, num_outputs, components, layer_sizes, kernel_size, reg_constant, learning_rate, optimizer, momentum, epsilon, point_only=point_only, name = 'VGG-16')
        
    def _build_model(self):
        
        x = self.main_input    #224
        bn = self.use_bn
        
        x = self.conv_layer(x, self.layer_sizes[0], self.kernel_size, bn, False)
        reduce = (True if self.resolution == 224 else False)
        x = self.conv_layer(x, self.layer_sizes[1], self.kernel_size, bn, reduce)    #112

        x = self.conv_layer(x, self.layer_sizes[2], self.kernel_size, bn, False)
        reduce = (True if self.resolution >= 112 else False)
        x = self.conv_layer(x, self.layer_sizes[3], self.kernel_size, bn, reduce)    #56

        x = self.conv_layer(x, self.layer_sizes[4], self.kernel_size, bn, False)
        x = self.conv_layer(x, self.layer_sizes[5], self.kernel_size, bn, False)
        reduce = (True if self.resolution >= 64 else False)
        x = self.conv_layer(x, self.layer_sizes[6], self.kernel_size, bn, reduce)    #28/32

        x = self.conv_layer(x, self.layer_sizes[7], self.kernel_size, bn, False)
        x = self.conv_layer(x, self.layer_sizes[8], self.kernel_size, bn, False)
        x = self.conv_layer(x, self.layer_sizes[9], self.kernel_size, bn, True)    #14/16

        x = self.conv_layer(x, self.layer_sizes[10], self.kernel_size, bn, False)
        x = self.conv_layer(x, self.layer_sizes[11], self.kernel_size, bn, False)
        x = self.conv_layer(x, self.layer_sizes[12], self.kernel_size, bn, True)    #7/8
        
        x = Flatten()(x)
        
        x = self.dense_layer(x, self.layer_sizes[13], True)
        x = self.dense_layer(x, self.layer_sizes[14], True)
        
        self.pre_output = x


class ResNet34(Network):
    
    def __init__(self, resolution, num_outputs = 3, components = 1, layer_sizes = (64,128,256,512), kernel_size = (3,3), reg_constant = .01, learning_rate = 0.0001, optimizer = 'adam', momentum = 0.9, epsilon = 1e-08):
        if resolution not in [32, 64, 112, 224]:
            raise Exception("Invalid resolution: must be one of 32, 64, 112, 224")
        self.resolution = resolution
        input_dims = (resolution, resolution, num_outputs)
        Network.__init__(self, input_dims, num_outputs, components, layer_sizes, kernel_size, reg_constant, learning_rate, optimizer, momentum, epsilon, name = 'ResNet-34')
    
    def _build_model(self):
        
        x = self.main_input    #224
        
        stride = (2 if self.resolution == 224 else 1)
        x = self.conv_layer(x, self.layer_sizes[0], (7,7), True, False, stride=stride)    #112
        if self.resolution >= 112:
            x = MaxPooling2D(padding = 'same', pool_size = 3, strides = 2)(x)    #56
        
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        
        if self.resolution >= 64:
            x = self.convolutional_block(x, 2, self.layer_sizes[1], (3,3), 2)     #28/32   
        else:
            x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))

        x = self.convolutional_block(x, 2, self.layer_sizes[2], (3,3), 2)    #14/16
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))
        
        x = self.convolutional_block(x, 2, self.layer_sizes[3], (3,3), 2)    #7/8
        x = self.identity_block(x, 2, (3,3))
        x = self.identity_block(x, 2, (3,3))

        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        
        self.pre_output = x

        
class GoogLeNet(Network):
    
    def __init__(self, resolution, num_outputs = 3, components = 1, layer_sizes = (), kernel_size = (3,3), reg_constant = .01, learning_rate = 0.0001, optimizer = 'adam', momentum = 0.9, epsilon = 1e-08):
        if resolution not in [32, 64, 112, 224]:
            raise Exception("Invalid resolution: must be one of 32, 64, 112, 224")
        self.resolution = resolution
        input_dims = (resolution, resolution, num_outputs)
        Network.__init__(self, input_dims, num_outputs, components, layer_sizes, kernel_size, reg_constant, learning_rate, optimizer, momentum, epsilon, name = 'GoogLeNet')

    def _build_model(self):    
    
        x = self.main_input    #224
        
        stride = (2 if self.resolution == 224 else 1)
        x = self.conv_layer(x, 64, (7,7), stride=stride)    #112
        if self.resolution >= 112:
            x = MaxPooling2D(3, 2, padding='same')(x)    #56
        x = self.conv_layer(x, 64, (1,1))
        x = self.conv_layer(x, 192, (3,3))
        if self.resolution >= 64:
            x = MaxPooling2D(3, 2, padding='same')(x)    #28/32
        
        x = self.inception_module(x, 64, 96, 128, 16, 32, 32)
        x = self.inception_module(x, 128, 128, 192, 32, 96, 64)
        x = MaxPooling2D(3, 2, padding='same')(x)    #14/16
        
        x = self.inception_module(x, 192, 96, 208, 16, 48, 64)
        self.aux_pre_output_1 = tf.reduce_mean(x, [1, 2])
        
        x = self.inception_module(x, 160, 112, 224, 24, 24, 64)
        x = self.inception_module(x, 128, 128, 256, 24, 64, 64)
        x = self.inception_module(x, 112, 144, 288, 32, 64, 64)
        self.aux_pre_output_2 = tf.reduce_mean(x, [1, 2])
    
        x = self.inception_module(x, 256, 160, 320, 32, 128, 128)
        x = MaxPooling2D(3, 2, padding='same')(x)    #7/8
        
        x = self.inception_module(x, 256, 160, 320, 32, 128, 128)
        x = self.inception_module(x, 384, 192, 384, 48, 128, 128)
        
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x, training=self.is_training)
        
        self.pre_output = x
        
    def _weighted_loss(self):
        
        self._build_aux_outputs(2)
        
        self.main_loss = tf.reduce_mean(self.loss, axis = -1)
        
        losses = tf.stack([self.main_loss, self.aux_loss_1, self.aux_loss_2])
        loss_weights = tf.constant([1/1.6, 0.3/1.6, 0.3/1.6])
        
        self.loss = tf.tensordot(losses, loss_weights, 1)


class InceptionV3(Network):
    
    def __init__(self, resolution, num_outputs = 3, components = 1, layer_sizes = (), kernel_size = (3,3), reg_constant = .01, learning_rate = 0.0001, optimizer = 'adam', momentum = 0.9, epsilon = 1e-08):
        if resolution not in [32, 64, 149, 299]:
            raise Exception("Invalid resolution: must be one of 32, 64, 149, 299")
        self.resolution = resolution
        input_dims = (resolution, resolution, num_outputs)
        Network.__init__(self, input_dims, num_outputs, components, layer_sizes, kernel_size, reg_constant, learning_rate, optimizer, momentum, epsilon, name = 'Inception-v3')

    def _build_model(self):    
    
        x = self.main_input    #299 
        reduction_padding = ('valid' if self.resolution >= 149 else 'same')

        if self.resolution == 299:
            x = self.conv_layer(x, 32, (3,3), True, stride=(2, 2), padding='valid')    #149
        else:
            x = self.conv_layer(x, 32, (3,3), True, stride=(1,1), padding='same')    #149
        x = self.conv_layer(x, 32, (3,3), True, padding=reduction_padding)    #147
        x = self.conv_layer(x, 64, (3,3),  True)
        if self.resolution >= 149:
            x = MaxPooling2D((3, 3), strides=(2, 2))(x)    #73
    
        x = self.conv_layer(x, 80, (1,1), True)
        x = self.conv_layer(x, 192, (3,3), True, padding=reduction_padding)    #71
        if self.resolution >= 64:
            x = MaxPooling2D((3, 3), strides=(2, 2), padding=reduction_padding)(x)    #35
        
        x = self.incep_a(x, 64, 48, 64, 64, 96, 96, 32)
        x = self.incep_a(x, 64, 48, 64, 64, 96, 96, 64)
        x = self.incep_a(x, 64, 48, 64, 64, 96, 96, 64)
        
        x = self.incep_reduction_a(x, 384, 64, 96, 96, padding = reduction_padding)    #17/16
        
        x = self.incep_b(x, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192)
        x = self.incep_b(x, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        x = self.incep_b(x, 192, 160, 160, 192, 160, 160, 160, 160, 192, 192)
        x = self.incep_b(x, 192, 192, 192, 192, 192, 192, 192, 192, 192, 192)
        
        x = self.incep_reduction_b(x, 192, 320, 192, 192, 192, 192, padding = reduction_padding)    #8
        
        x = self.incep_c(x, 320, 384, 384, 448, 384, 384, 192)
        x = self.incep_c(x, 320, 384, 384, 448, 384, 384, 192)
        
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x, training=self.is_training)
        
        self.pre_output =x





