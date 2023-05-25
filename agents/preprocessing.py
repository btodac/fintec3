#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:09:23 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

class ObservationBuilder(object):
    
    def __init__(self, columns, opening_time, closing_time, tz):
        self.columns = columns
        self._opening_time = opening_time
        self._closing_time = closing_time
        self.tz = tz
    
    def make_observations(self, data,):
        data = data.copy()
        if data.index.tz != self.tz:
            data.index = data.index.tz_convert(self.tz)
        data = data.between_time(self._opening_time.time(),
                                 self._closing_time.time())
        observations = self._build_observation(data)
        return observations, data.index
    
    def _build_observation(self, data,):
        features = []
        for c in self.columns:
            feature = None
            if 'range' == c:
                feature = data['High'] - data['Low']
            elif 'mom' in c:
                if '_' in c:
                    t_string = c.split('_')[0]
                    feature = data.loc[:, 'Close'].rolling(t_string).apply(
                        lambda x: (x[-1] - x[0]) / x[-1],
                        raw=True
                    )
                else:
                    feature = data['Close'] - data['Open']
            elif 'trend' in c:
                t_string = c.split('_')[0]
                trend = data['Close'] - data['Open']
                feature = trend.rolling(t_string).mean()
            elif 'high' in c:
                t_string = c.split('_')[0]
                h = data.loc[:,'High'].rolling(t_string).max()
                feature = h - data['Close']
            elif 'low' in c:
                t_string = c.split('_')[0]
                h = data.loc[:,'Low'].rolling(t_string).min()
                feature = h - data['Close']
            elif 'mean_diff' in c:
                t_str1, t_str2 = c.split('_')[:2]  # 5min_15min_mean_diff
                m1 = data.loc[:, 'Close'].rolling(t_str1).mean()
                m2 = data.loc[:, 'Close'].rolling(t_str2).mean()
                feature = m1 - m2
            elif 'mean_dist' in c:
                t_str = c.split('_')[0]  # 8min_mean_dist
                m1 = data.loc[:, 'Close'].rolling(t_str).mean()
                m2 = data.loc[:, 'Close']
                feature = (m1 - m2)  # / m2
            elif 'mean' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:, 'Close'].rolling(t_string).mean().rolling("2min").apply(
                    lambda x: (x[-1] - x[0]) / x[1],
                    raw=True
                )
            elif 'std' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:, 'Close'].rolling(t_string).std()
                m = data.loc[:, 'Close'].rolling(t_string).mean()
                feature = feature / m
            elif 'skew' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:, 'Close'].rolling(t_string).skew()
            elif 'kurt' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:, 'Close'].rolling(t_string).kurt()
            elif 'stoch_osc' in c:
                t_str = c.split('_')[0]
                h = data.loc[:, 'High'].rolling(t_str).max()
                l = data.loc[:, 'Low'].rolling(t_str).min()
                so = (data.loc[:, 'Close'] - l) / (h - l)
                m = so.rolling('3min').mean()
                feature = so - m
        
            if feature is not None:
                features.append(feature)
        observations = pd.concat(features, axis=1).to_numpy()
        observations[np.isnan(observations)] = 0
        
        return observations
    
    def _extract_time_periods(self, column):
        return [s for s in '3min_5min_mean'.split('_') if 'min' in s]
       
