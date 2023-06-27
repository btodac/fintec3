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
            end_time=pd.Timestamp("23:59"),
            tz='UTC',
            rho=0.0,
            kappa=0.001,
            theta=0.005,
            chi=5e-6
            ):
        if start_date.date() == end_date.date():
            end_date += pd.Timedelta(days=1)
        self.start_date = start_date
        self.end_date = end_date
        self.start_time = start_time
        self.end_time = end_time
        self.tz = tz
        self.freq = freq
        self.initial_value = initial_value
        self.drift = drift
        self.volatility = volatility
        
        self.rho = rho
        self.kappa = kappa # rate for return to mean
        self.theta = theta # long term average volatility
        self.chi = chi # volatility of the volatility
        assert (2 * self.kappa * self.theta) > (self.chi ** 2), \
            "Feller condition fail: 2 * kappa * theta <= chi ** 2! "\
            "Volatility may become negative"
    
    def generate(self,):
        tick_seconds = 1/4 * convert_offset_to_seconds(self.freq) # tick in seconds (5datapoints)
        # TODO: ohlc data needs to use the close value of one 
        # time period as the open value of the next!
        dp_index = self._make_index(f"{tick_seconds * 1e6}U")
        n_points = len(dp_index) 
        
        if type(self.drift) == np.array and type(self.volatility) == np.array:
            pass
        else:
            data = self._stochastic_volatility_motion(#self._generate_motion(
                n_points, self.initial_value, tick_seconds
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
    
    def _stochastic_volatility_motion(self, n_points, initial_value, dt):
        mean = np.array([0,0])
        cov = np.array([
            [dt , self.rho * dt],
            [self.rho * dt, dt]
            ])
        
        gmb = np.random.multivariate_normal(mean, cov, size=(n_points))
        v = self.theta # the instantaneous volatility
        volatility = np.zeros(n_points)
        for i in range(n_points):
            dv = (self.kappa * (self.theta - v) * dt \
                  + self.chi * np.sqrt(v) * gmb[i,1])
            v += dv
            volatility[i] = v
        
        motion = np.exp(
            (self.drift - volatility**2 / 2) * dt +
            volatility * gmb[:,0].squeeze()
        )
        return initial_value * motion.cumprod()
    
    def conv_to_ohlc(self, data, freq):
        df = data.resample(freq).ohlc()
        df.columns = [c.capitalize() for c in df.columns]
        df = df.loc[self._make_index(freq),:]
        c = df['Close']
        c.index = c.index + pd.tseries.frequencies.to_offset(freq)
        index = c.index.intersection(df.index)
        df.loc[index,'Open'] = c.loc[index]
        return df
    
    def _make_index(self, freq):
        index = pd.bdate_range(self.start_date, self.end_date, freq=freq)
        index = index.to_series()
        index = index.between_time(self.start_time.time(), self.end_time.time())
        index = index.index
        try:
            index = index.tz_localize(self.tz)
        except TypeError:
            pass
        return index
    
    
if __name__ == "__main__":
    import mplfinance as mpl
    initial_value = 1e4
    final_value = 10050
    change = (final_value -initial_value) / initial_value
    drift = np.log(1 + change) / (511 * 60) # log(1 + R) / (n_points * freq_as_seconds)
    
    data_gen = GBMDataGen(start_date=pd.Timestamp("2023-01-01"),
                          end_date=pd.Timestamp("2023-07-01"),
                          freq='1min',
                          initial_value=initial_value,
                          drift=5e-9,#drift,#0.000000003, # final value = initial_value + (1+drift*nticks*tick_length)
                          volatility=1e-5,#0.00005,#0.00001, # (daily_std/initial_value)**2 * freq_as_seconds
                          start_time=pd.Timestamp("09:00"),
                          end_time=pd.Timestamp("17:30"),
                          rho=0.01,
                          kappa=0.1,
                          theta=np.random.gamma(1,0.5) * 1e-4,
                          chi=1e-4
                          )
    df = data_gen.generate()
    daily_std = df.resample('D').std().mean()
    mpl.plot(df.iloc[:511], type='candle', style='yahoo', title='Data')                     
    df_rs = data_gen._raw.resample('1W').ohlc()
    mpl.plot(df_rs, type='candle', style='yahoo', title='Data')   