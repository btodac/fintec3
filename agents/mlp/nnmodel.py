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
from agenttesting.results import Results, OutcomeSimulator
from utillities.timesanddates import get_ticker_time_zone, opening_and_closing_times

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
        'stop_loss': 2,
        'time_limit': 10,#15,
        'live_tp': 25,
        'live_sl': 5,
        'live_tl': 15,
    },
    '^GDAXI': {
        'take_profit': 25,
        'stop_loss': 5,
        'time_limit': 15,
        'live_tp': 25,
        'live_sl': 5,
        'live_tl': 10,
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
    def __init__(self, ticker, columns, params=None, observer=None, both_directions_hold=False):
        super().__init__(ticker, columns, params=params, observer=observer)
        
        self._both_as_hold = both_directions_hold
        self.max_norm = 4
        self.dropout = 0.2
        self._model = self.create_model(input_data_size=len(columns),
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
        x_train, y_train, _ = self._get_observations_and_targets(training_data)
        class_weight = y_train.mean(axis=0)
        class_weight = class_weight.max() / class_weight
        print(f'Normalised weights: {class_weight}')
        class_weight[:2] = 0.95 * class_weight[:2]
        #class_weight = np.array([1.1,1.1,1])#np.ones(3)
        print(class_weight)
        class_weight = dict(zip(np.arange(len(class_weight)), class_weight))
    
        x_valid, y_valid, _ = self._get_observations_and_targets(validation_data)
    
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="cat_acc"),]
        callback = tf.keras.callbacks.EarlyStopping(
            #monitor='val_cat_acc', mode="max",
            monitor='val_loss', mode='min',
            patience=10,
            restore_best_weights=True,
            )
        optimizer = keras.optimizers.Adam(
            learning_rate=0.0001,
            clipnorm=0.5
            )
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.02,
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
            epochs=100, batch_size=32, shuffle=True,
            callbacks=[callback],
            )
    
    def _get_observations_and_targets(self, data):
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz'])

        observations, order_datetimes = self.observer.make_observations(
            data, self._params['ticker'], self._params['columns'],
        )

        targets = self._get_targets(data, order_datetimes)
        return observations, targets, order_datetimes
    
    def _get_targets(self, data, order_datetimes,):
        outcome_simulator = OutcomeSimulator()

        orders = pd.DataFrame()
        orders.loc[:, 'Opening_Value'] = data.loc[order_datetimes, 'Close']
        orders.loc[:, 'Ticker'] = self._params['ticker']
        orders.loc[:, 'Take_Profit'] = self._params['take_profit']
        orders.loc[:, 'Stop_Loss'] = self._params['stop_loss']
        orders.loc[:, 'Time_Limit'] = self._params['time_limit']

        # buys
        orders.loc[:, 'Type'] = "BUY"
        buy_orders = outcome_simulator.limit_times(orders, data)
        is_tp_buy, _, _ = outcome_simulator.outcome_bool(buy_orders)
        # sells
        orders.loc[:, 'Type'] = "SELL"
        sell_orders = outcome_simulator.limit_times(orders, data)
        is_tp_sell, _, _ = outcome_simulator.outcome_bool(sell_orders)

        is_buy = np.logical_and(
            is_tp_buy,
            np.logical_not(is_tp_sell)
        )
        is_sell = np.logical_and(
            is_tp_sell,
            np.logical_not(is_tp_buy)
        )
        is_both = np.logical_and(
            is_tp_buy,
            is_tp_sell
        )
        if self._both_as_hold:
            is_hold = np.logical_not(np.logical_xor(is_buy, is_sell))
        else:
            buy_faster = buy_orders['Take_Profit_Time'].to_numpy() \
                > sell_orders['Take_Profit_Time'].to_numpy()
            is_buy[np.logical_and(is_both, np.logical_not(buy_faster))] = False
            is_sell[np.logical_and(is_both, buy_faster)] = False
            is_hold = np.logical_not(np.logical_or(is_buy, is_sell))
        probs = np.stack((is_buy, is_sell, is_hold)).T
        return probs

# Standard results methods using take profit, stop loss and time limit exit conditions
class NNResults(Results):
    pass

# Steering results methods using model to close orders additionally to standard exit conditions
class NNSteeringResults(Results):
    
    def construct_orders(self, data, ticker, take_profit, stop_loss, time_limit):
        directions = np.empty(len(self.predictions), dtype='<U4')
        directions[self.predictions == 0] = "BUY"
        directions[self.predictions == 1] = "SELL"
        directions[self.predictions == 2] = "HOLD"
        is_change = directions[:-1] != directions[1:]
        is_change = np.array([directions[0] != 'HOLD'] + is_change.tolist())
        
        old_d = 'HOLD'
        order = {}
        orders = {}
        for i, d, dt in zip(is_change, directions, self.order_datetimes):
            if i:
                new_d = d
                if new_d in ['BUY','SELL']:
                    if old_d != 'HOLD':
                        dt_limit = max(
                            self.order_datetimes[
                                self.order_datetimes.date == order['Opening_Datetime'].date()
                                ]
                            )
                        closing_datetime = min(dt, dt_limit)
                        if closing_datetime == order['Opening_Datetime']:
                            closing_datetime += pd.Timedelta(minutes=1)
                        order['Closing_Datetime'] = closing_datetime
                        orders[order['Opening_Datetime']] = order
                        order = {}
                        
                    order['Opening_Datetime'] = dt
                    order['Type'] = d
                else:
                    if old_d != 'HOLD':
                        dt_limit = max(
                            self.order_datetimes[
                                self.order_datetimes.date == order['Opening_Datetime'].date()
                                ]
                            )
                        closing_datetime = min(dt, dt_limit)
                        if closing_datetime == order['Opening_Datetime']:
                            closing_datetime += pd.Timedelta(minutes=1)
                        order['Closing_Datetime'] = closing_datetime
                        orders[order['Opening_Datetime']] = order
                        order = {}
                old_d = new_d
            else:
                pass
            
        orders = pd.DataFrame(orders).T
        orders['Opening_Value'] = data.loc[orders.index,'Close'].to_numpy()
        orders['Closing_Value'] = data.loc[orders['Closing_Datetime'],'Close'].to_numpy()
        ds = orders['Type'].to_numpy().copy()
        ds[ds=='BUY'] = 1
        ds[ds=='SELL'] = -1
        ds = ds.astype(float)
        orders['Profit'] = ds * (orders['Closing_Value'] - orders['Opening_Value'])
        orders['Ticker'] = ticker
        orders['Take_Profit'] = take_profit * OutcomeSimulator.multipliers[ticker]
        orders['Stop_Loss'] = stop_loss * OutcomeSimulator.multipliers[ticker]
        
        t = orders['Closing_Datetime'] - orders['Opening_Datetime']
        t = np.array([int(np.round(ti.seconds / 60)) for ti in t.to_list()])
        orders['Time_Limit'] = t[:,np.newaxis]
        self.orders = orders.copy()
        
        return orders

