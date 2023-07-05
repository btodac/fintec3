#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 23:41:54 2023

@author: mtolladay
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:50:52 2023

@author: mtolladay
"""
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

log = logging.getLogger(__name__)

class NNModel(object):
    def __init__(self,input_shape, output_shape):
        '''
        NNAgent is a subclass of Agent that uses a multi layer 
        perceptron model to make predictions

        Parameters
        ----------
        ticker : str
            Ticker of financial product
        columns : list
            list of strings containing features
        params : dict, optional
            see Agent.
        observer : TYPE, optional
           see Agent
        target_generator : TYPE, optional
            see Agent

        Returns
        -------
        None.

        '''
        # Model
        self.max_norm = 5
        self.dropout = 0.2
        self.n_layers = 5
        self.n_units = 128
        # Fitting
        self.batch_size = 64
        self.epochs = 500
        # Optimizer
        self.optimizer_args = {
            "learning_rate" : 0.0001,
            "clipnorm" : 2,
            }
        # Metric
        self.metric_args = {
            "name" : "cat_acc",
            }
        # Loss fcn
        self.loss_fcn_args = {
            "from_logits" : False,
            "label_smoothing" : 0.01,
            "axis" : -1,
            "reduction" : "auto", 
            "name" : "crossentropy",
            }
        self.callback_args = {
            #"monitor" : 'val_cat_acc', "mode" : "max",
            "monitor" :'val_loss', "mode" : 'min',
            "patience" : 10,
            "restore_best_weights" : False,#True,
            }

        self._model = self._create_model(input_shape, output_shape)
        self._is_fit = False
    
    def _create_model(self, input_data_size, output_data_size):
        model = keras.Sequential()
        model.add(layers.Input(shape=input_data_size))
        model.add(layers.Flatten())
        for i in range(self.n_layers):
            model.add(layers.Dense(
                self.n_units, activation='relu',
                kernel_constraint=keras.constraints.MaxNorm(self.max_norm),
            ))
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(output_data_size, activation='softmax'))
        
        optimizer = keras.optimizers.Adam(**self.optimizer_args)
        loss_fcn = tf.keras.losses.CategoricalCrossentropy(**self.loss_fcn_args)
        metrics = [tf.keras.metrics.CategoricalAccuracy(**self.metric_args)]
        model.compile(
            optimizer=optimizer, loss=loss_fcn, weighted_metrics=metrics
            )
        return model
    
    def make_prediction(self, observations):
        if len(observations.shape) == len(self._model.input_shape):
            y = self._model.predict(observations)
        elif len(observations.shape) == len(self._model.input_shape[1:]):
            observation = tf.convert_to_tensor(observations.squeeze())
            observation = tf.expand_dims(observations, 0)
            y = np.array(self._model(observation))
        else:
            raise ValueError(f'Observations had shape {observations.shape} but '\
                             f'model has input shape of {self._model.input_shape}')
        return y
    
    def fit(self, observations, targets, validation_data):
        if not self._is_fit:
            class_weight = targets.mean(axis=0)
            class_weight = class_weight.max() / class_weight
            print(f'Normalised weights: {class_weight}')
            if any(class_weight>4):
                print('The class weights are large enough to cause over fitting')
            class_weight = dict(zip(np.arange(len(class_weight)), class_weight))
            #self._model = self._create_model(
            #    input_data_size=observations.shape[1:],
            #    output_data_size=targets.shape[-1]
            #    )
        
            self._is_fit = True # Bit hacky to change this before it is fit
            callback = tf.keras.callbacks.EarlyStopping(**self.callback_args)
            return self._model.fit(
                observations, targets,
                validation_data=validation_data,
                class_weight=class_weight,
                verbose=2,  # 2,
                epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                callbacks=[callback],
                )
        
    def __reduce__(self):
        state = self.__dict__.copy()
        if self._is_fit:
            state.update({'weights' : self._model.get_weights(),})
            return type(self), (self._model.input_shape, self._model.output_shape[-1]), state
        else:
            state = self.__dict__.copy()
            del state["_model"]
            return type(self), (self._model.input_shape, self._model.output_shape[-1]), state
        
    def __setstate__(self, state):
        if 'weights' in state.keys():
            weights = state.pop('weights')
        else:
            weights = None
        
        self.__dict__.update(state)
        #self._model = self.create_model(self._model.input_shape, self._model.output_shape)
        if weights is not None:
            self._model.set_weights(weights)