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

class NNAgent(object):
    def __init__(self,):
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
        
        self.max_norm = 3
        self.dropout = 0.2
        self.n_layers = 5
        self.n_units = 128
        self.batch_size = 32
        self.epochs = 500
        self.metrics = [tf.keras.metrics.CategoricalAccuracy(name="cat_acc"),]
        self.optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
            clipnorm=0.3
            )
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.05,
            axis=-1,
            reduction="auto", name="crossentropy",
            )
        self.callback = tf.keras.callbacks.EarlyStopping(
            #monitor='val_cat_acc', mode="max",
            monitor='val_loss', mode='min',
            patience=10,
            restore_best_weights=False,#True,
            )
        
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
        model.compile(optimizer=self.optimizer, 
                      loss=self.loss_fn, 
                      weighted_metrics=self.metrics)
        return model
    
    def make_prediction(self, observations):
        if len(observations.shape) >= 2:
            y = self._model.predict(observations)
        else:
            observation = tf.convert_to_tensor(observations.squeeze())
            observation = tf.expand_dims(observations, 0)
            y = self._model(observation)
        return y
    
    def fit(self, observations, targets, validation_data):
        if not self._is_fit:
            class_weight = targets.mean(axis=0)
            class_weight = class_weight.max() / class_weight
            print(f'Normalised weights: {class_weight}')
            if any(class_weight>4):
                print('The class weights are large enough to cause over fitting')
            class_weight = dict(zip(np.arange(len(class_weight)), class_weight))
            self._model = self._create_model(
                input_data_size=observations.shape[1:],
                output_data_size=targets.shape[-1]
                )
        
            self._is_fit = True # Bit hacky to change this before it is fit
            return self._model.fit(
                observations, targets,
                validation_data=validation_data,
                class_weight=class_weight,
                verbose=2,  # 2,
                epochs=self.epochs, batch_size=self.batch_size, shuffle=True,
                callbacks=[self.callback],
                )