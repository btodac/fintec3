#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:26:38 2023

@author: mtolladay
"""
import logging
import numpy as np
import pandas as pd

from agenttesting.datagenerator import GBMDataGen

log = logging.getLogger(__name__)


class MarketDataGen(object):
    def __init__(self, observer):
        self.observer = observer
        
        start_date = pd.Timestamp("2022-01-01")
        self._data_len = pd.Timedelta(weeks=1)
        self._data_gen = GBMDataGen(
            start_date = start_date, 
            end_date = start_date + self._data_len, 
            freq = '1min', 
            start_time = pd.Timestamp("09:00"),
            end_time = pd.Timestamp("17:30"),
            rho=0.01,
            kappa=0.1,
            drift=0,
            theta=1e-4,
            )
        
        self._make_new_data()
        self._counter = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        done = self._counter >= len(self.data)
        if done:
            self._make_new_data()
            self._counter = 0
            
        observation = self.observations[self._counter,:] 
        self._counter +=1
        return observation, done
    
    def get_data_slice(self, look_back):
        index = self.data.index[:self._counter]
        index = index[-look_back:]
        index = index[index.date == index[-1].date()]
            
        return self.data.loc[index,:]
                   
    @property
    def current_price(self):
        return float(self.data['Close'].iloc[self._counter-1])
    
    def _make_new_data(self,):
        #self._data_gen.drift = np.random.normal(scale=0.1) * 1e-9
        self._data_gen.theta = np.random.gamma(8,1/8) * 1e-4
        try:
            self._data_gen.initial_value = self.data['Close'].iloc[-1]
        except AttributeError:
            self._data_gen.initial_value = 13000
            
        self.data = self._data_gen.generate()
        log.info(f"Total std = {self.data['Close'].std()}, "\
                 f"daily std = {self.data['Close'].resample('D').std().mean()}")
        self.observations, _  = self.observer.make_observations(
            self.data,
            self._data_gen.start_time,
            self._data_gen.end_time,
            self._data_gen.tz
            )
        self._data_gen.start_date += self._data_len
        self._data_gen.end_date += self._data_len
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import mplfinance as mpl
    from agents.preprocessing import ObservationBuilder
    observer = ObservationBuilder(columns=['10min_Std'])
    dg = MarketDataGen(observer)
    
    mpl.plot(dg.data.iloc[:100])
    
        
        
        
        
        