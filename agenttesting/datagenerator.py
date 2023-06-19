#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:14:47 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

from utillities.timesanddates import convert_offset_to_seconds

class GBMDataGen(object):
    
    def __init__(self, 
            start_date, end_date, 
            freq='1min', 
            initial_value=10000,
            drift=0, # points per second
            volatility=0.005, 
            start_time=pd.Timestamp("00:00"),
            end_time=pd.Timestamp("23:59")
            ):
        #self.index = pd.bdate_range(start_date, end_date, freq=freq)
        self.start_date = start_date
        self.end_date = end_date
        self.start_time = start_time
        self.end_time = end_time
        self.freq = freq
        self.initial_value = initial_value
        self.drift = drift
        self.volatility = volatility
    
    def generate(self,):
        n_seconds = 0.25 * convert_offset_to_seconds(self.freq) 
        dp_index = self._make_index(f"{n_seconds * 1e6}U")
        n_points = len(dp_index) 
        
        if type(self.drift) == np.array and type(self.volatility) == np.array:
            pass
        else:
            data = self._generate_motion(
                n_points, self.initial_value, 
                self.drift, self.volatility, n_seconds
                )

        data = pd.Series(data, index=dp_index)
        self._raw = data
        return self.conv_to_ohlc(data, self.freq)
            
    def _generate_motion(self, n_points, initial_value, drift, volatility, dt):
        motion = np.exp(
            (drift - volatility**2 / 2) * dt +
            volatility * np.random.normal(0, np.sqrt(dt), size=(n_points))
        )
        
        return initial_value * motion.cumprod()
    
    def conv_to_ohlc(self, data, freq):
        df = data.resample(freq).ohlc()
        df.columns = [c.capitalize() for c in df.columns]
        return df
    
    def _make_index(self, freq):
        index = pd.bdate_range(self.start_date, self.end_date, freq=freq)
        index = index.to_series()
        index = index.between_time(self.start_time.time(), self.end_time.time())
        return index.index
    
    
if __name__ == "__main__":
    import mplfinance as mpl
    
    data_gen = GBMDataGen(start_date=pd.Timestamp("2023-01-01"),
                          end_date=pd.Timestamp("2023-02-01"),
                          freq='1H',#'1min',
                          initial_value=10000,
                          drift=0.00000001, # points per 0.25 second
                          volatility=0.00001, # points per 0.25 second
                          start_time=pd.Timestamp("09:00"),
                          end_time=pd.Timestamp("17:30"),
                          )
    df = data_gen.generate()
    mpl.plot(df.iloc[:50], type='candle', style='yahoo', title='Data')                     
    
    df_rs = data_gen._raw.resample('1D').ohlc()
    mpl.plot(df_rs, type='candle', style='yahoo', title='Data')   