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

from agents import Agent
#from agenttesting.results import Results, OutcomeSimulator
#from utillities.timesanddates import get_ticker_time_zone, opening_and_closing_times

log = logging.getLogger(__name__)

PARAMS = {
    '^DJI': {
        # Position paramaters
        'take_profit': 75,
        'stop_loss': 20,
        'time_limit': 30,
        # Training params
        'up': 20,  # 20,
        'down': -20,  # 20,
        'to': 15,
    },
    "^NDX": {
        'take_profit': 30,#100
        'stop_loss': 3,
        'time_limit': 10,#15,
        'live_tp': 30,
        'live_sl': 10,
        'live_tl': 10,
        'to' : 10,
    },
    '^GDAXI': {
        'take_profit': 25,
        'stop_loss': 5,
        'time_limit': 15,
        'live_tp': 25,
        'live_sl': 5,
        'live_tl': 10,
        'to' : 10,
    },
    '^FCHI': {
        'take_profit': 17,
        'stop_loss': 4,
        'time_limit': 30,
        'up': 5,
        'down': -5,
        'to': 15,
    },
    '^FTSE': {
        'take_profit': 15,
        'stop_loss': 3,
        'time_limit': 30,
        'up': 10,
        'down': -10,
        'to': 30,
    },
    'EUR=X': {
        'take_profit': 30 * 1e-8,
        'stop_loss': 5 * 1e-8,
        'time_limit': 30,
        'up': 0.0005,
        'down': -0.0005,
        'to': 20,
    }
}


class NNAgent(Agent):
    def __init__(self, ticker: str, columns: list, params=None, 
                 observer=None, target_generator=None):
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
        
        if params is None:
            params = PARAMS[ticker]
        
        super().__init__(ticker, columns, params=params, observer=observer,
                         target_generator=target_generator)
        
        self.max_norm = 3
        self.dropout = 0.2
        self._model = self._create_model(input_data_size=len(columns),
                                         output_data_size=3)
    
    def _create_model(self, input_data_size, output_data_size):
        model = keras.Sequential()
        model.add(layers.Input(shape=input_data_size))
        model.add(layers.Flatten())
        for i in range(5):
            model.add(layers.Dense(
                128, activation='relu',
                kernel_constraint=keras.constraints.MaxNorm(self.max_norm),
            ))
            model.add(layers.Dropout(self.dropout))
        model.add(layers.Dense(output_data_size, activation='softmax'))
        return model
    
    def make_prediction(self, observations):
        if len(observations.shape) == 2:
            y = self._model.predict(observations)
        else:
            observation = tf.convert_to_tensor(observations.squeeze())
            observation = tf.expand_dims(observations, 0)
            y = self._model(observation)
        return y
    
    def fit(self, training_data, validation_data):
        x_train, y_train, _ = self.get_observations_and_targets(training_data)
        class_weight = y_train.mean(axis=0)
        class_weight = class_weight.max() / class_weight
        print(f'Normalised weights: {class_weight}')
        if any(class_weight>4):
            print('The class weights are large enough to cause over fitting')
        #class_weight[:2] = 0.5 * class_weight[:2]
        class_weight = dict(zip(np.arange(len(class_weight)), class_weight))
    
        x_valid, y_valid, _ = self.get_observations_and_targets(validation_data)
    
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="cat_acc"),]
        callback = tf.keras.callbacks.EarlyStopping(
            #monitor='val_cat_acc', mode="max",
            monitor='val_loss', mode='min',
            patience=10,
            restore_best_weights=True,
            )
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
            clipnorm=0.3
            )
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.05,
            axis=-1,
            reduction="auto", name="crossentropy",
            )
    
        self._model.compile(optimizer=optimizer, loss=loss_fn, weighted_metrics=metrics)
        #print(self._model.summary())
    
        return self._model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid),
            class_weight=class_weight,
            verbose=2,  # 2,
            epochs=500, batch_size=64, shuffle=True,
            callbacks=[callback],
            )