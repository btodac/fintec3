#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:41:57 2023

@author: mtolladay
"""
import logging

import numpy as np
import pandas as pd

from agents.preprocessor.preprocessing import ObservationBuilder
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


class Agent(object):
    def __init__(self, ticker, columns, params=None, observer=None):

        self._params = {
            'ticker': ticker,
            'columns': columns,
            'tz': get_ticker_time_zone(ticker)
        }
        if params == None:
            self._params.update(PARAMS[ticker])
        else:
            self._params.update(params)
            
        self._set_tradeable_time(ticker)
        if observer is None:
            self.observer = ObservationBuilder(columns, self._opening_time, 
                                               self._closing_time, self._params['tz'])
    
    def fit(self, training_data, validation_data):
        raise NotImplementedError()
                
    def make_prediction(self, observation):
        raise NotImplementedError()  
    
    def __call__(self, data):
        observations, _ = self.observer.make_observations(data)
        observation = observations[-1, :]
        pred = np.argmax(self.make_prediction(observation))
        return {0: 'BUY', 1: 'SELL', 2: 'HOLD'}[pred]

    def predict(self, data, ticker):
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz'])
        observations, order_datetimes = self.observer.make_observations(data)
        prob = self.make_prediction(observations)
        pred = np.argmax(prob, axis=1)
        return pred, prob, order_datetimes
        
    def _set_tradeable_time(self, ticker):
        opening_time, closing_time = opening_and_closing_times(ticker)
        opening_time = opening_time.tz_convert(self._params['tz'])
        closing_time = closing_time.tz_convert(self._params['tz'])
        if not np.isnan(self._params['live_tl']):
            closing_time -= pd.Timedelta(minutes=self._params['live_tl'])
        else:
            closing_time -= pd.Timedelta(minutes=1)
        self._opening_time = opening_time
        self._closing_time = closing_time
