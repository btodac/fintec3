#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:09:23 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

### TODO: Normalisation of observables!

class ObservationBuilder(object):
    
    def __init__(self, columns, opening_time, closing_time, tz):
        self.columns = columns
        '''
        for c in columns:
            self._feature_generators[c] = self._parse_feature_name(c):
        '''
        self._opening_time = opening_time
        self._closing_time = closing_time
        self.tz = tz
    
    def make_observations(self, data,):
        data = data.copy()
        if data.index.tz != self.tz:
            data.index = data.index.tz_convert(self.tz)
        data = data.between_time(self._opening_time.time(),
                                 self._closing_time.time())
        
        '''
        features = []
        for column, feature_gen in self._feature_generators.items():
            features.append(feature(data))
        observations = pd.concat(features, axis=1).to_numpy()
        observations[np.isnan(observations)] = 0
        return observations
        '''
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
    
    def _parse_feature_name(self, column):
        c = column.split('_')
        feature = [s for s in c if 'min' not in s]
        t_strings = [s for s in c if 'min' in s]
        return  globals()[feature](t_strings)
          
class High:
    def __init__(self, t_string):
        self.t_string = t_string[0]
        
    def __call__(self, data):
        h = data['High'].rolling(self.t_string).max()
        feature = h - data['Close']
        return feature
    
class Kurt:
    def __init__(self, t_string):
        self.t_string = t_string[0]
        
    def __call__(self, data):
        feature = data['Close'].rolling(self.t_string).kurt()
        return feature
    
class Low:
    def __init__(self, t_string):
        self.t_string = t_string[0]
        
    def __call__(self, data):
        h = data['Low'].rolling(self.t_string).min()
        feature = h - data['Close']
        return feature
    
class Mean:
    def __init__(self, t_string):
        self.t_string = t_string[0]
        
    def __call__(self, data):
        feature = data.loc[:, 'Close'].rolling(self.t_string).mean().rolling("2min").apply(
            lambda x: (x[-1] - x[0]) / x[1],
            raw=True
        )
        return feature

class MeanDiff:
    def __init__(self, t_string):
        self.t_string1 = t_string[0]
        self.t_string2 = t_string[1]
        
    def __call__(self, data):
        c = data['Close']
        m1 = c.rolling(self.t_string1).mean()
        m2 = c.rolling(self.t_string2).mean()
        feature = (m1 - m2) / c
        return feature  

class MeanDist:
    def __init__(self, t_string):
        self.t_string = t_string[0]
        
    def __call__(self, data):
        c = data['Close']
        feature = (c.rolling(self.t_string).mean() - c) / c
        return feature  
    
class Mom:
    def __init__(self, t_string):
        self._one_minute = len(t_string) == 0
        if not self._one_minute:
            self.t_string = t_string[0]           
        
    def __call__(self, data):
        if self._one_minute:
            feature = data['Close'] - data['Open']
        else:
            feature = data['Close'].rolling(self.t_string).apply(
                lambda x: (x[-1] - x[0]),
                raw=True
            )
        feature /= data['Close']
        return feature
    
class Range:
    def __init__(self, t_string):
        self._one_minute = len(t_string) == 0
        if not self._one_minute:
            raise ValueError('Not implemented!')
            #_string = t_string[0] 
            
    def __call__(self, data):
        if self._one_minute:
            feature = data['High'] - data['Low']
        else:
            feature = data['Close'].rolling(self.t_string).apply(
                lambda x: (x[-1] - x[0]) / x[-1],
                raw=True
            )
        feature /= data['Close']
        return feature
 
class Skew:
    def __init__(self, t_string):
        self.t_string = t_string
        
    def __call__(self, data):
        feature = data['Close'].rolling(self.t_string).skew()
        return feature    

class Std:
    def __init__(self, t_string):
        self.t_string = t_string
        
    def __call__(self, data):
        c = data['Close']
        feature = c.rolling(self.t_string).std()
        feature /= c
        return feature

class StochOsc:
    def __init__(self, t_string):
        self.t_string = t_string[0]
    
    def __call__(self, data):
        h = data['High'].rolling(self.t_string).max()
        l = data['Low'].rolling(self.t_string).min()
        so = (data['Close'] - l) / (h - l)
        m = so.rolling('3min').mean()
        feature = so - m
        return feature
    
class Trend:
    def __init__(self, t_string):
        self.t_string = t_string[0]
        
    def __call__(self, data):
        trend = data['Close'] - data['Open']
        feature = trend.rolling(self.t_string).mean() / data['Close']
        return feature     
