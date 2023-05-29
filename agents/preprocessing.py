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
        self._feature_generators = {}
        for c in columns:
            self._feature_generators[c] = self._parse_feature_name(c)
        self._opening_time = opening_time
        self._closing_time = closing_time
        self.tz = tz
    
    def make_observations(self, data,):
        data = data.copy()
        if data.index.tz != self.tz:
            data.index = data.index.tz_convert(self.tz)
        data = data.between_time(self._opening_time.time(),
                                 self._closing_time.time())
        features = []
        for column, feature_gen in self._feature_generators.items():
            features.append(feature_gen(data))
        observations = pd.concat(features, axis=1).to_numpy()
        observations[np.isnan(observations)] = 0
        return observations, data.index
        
    def _parse_feature_name(self, column):
        c = column.split('_')
        feature = [s for s in c if 'min' not in s][0]
        t_strings = [s for s in c if 'min' in s]
        try:
            f = globals()[feature](t_strings)
        except KeyError:
            raise KeyError(f'Unknown feature {feature}')    
        return f

class Feature:
    def __init__(self, t_strings):
        self._one_minute = len(t_strings) == 0
        if not self._one_minute:
            self.t_strings = t_strings       
          
class High(Feature):
    def __call__(self, data):
        h = data['High'].rolling(self.t_strings[0]).max()
        feature = h - data['Close']
        return feature
    
class Kurt(Feature):    
    def __call__(self, data):
        feature = data['Close'].rolling(self.t_strings[0]).kurt()
        return feature
    
class Low(Feature):
    def __call__(self, data):
        h = data['Low'].rolling(self.t_strings[0]).min()
        feature = h - data['Close']
        return feature
    
class Mean(Feature):
    def __call__(self, data):
        feature = data['Close'].rolling(self.t_strings[0]).mean().rolling("2min").apply(
            lambda x: (x[-1] - x[0]),
            raw=True
        )
        feature /= data['Close']
        return feature

class MeanDiff(Feature):    
    def __call__(self, data):
        c = data['Close']
        m1 = c.rolling(self.t_strings[0]).mean()
        m2 = c.rolling(self.t_strings[1]).mean()
        feature = (m1 - m2) / c
        return feature  

class MeanDist(Feature):
    def __call__(self, data):
        c = data['Close']
        feature = (c.rolling(self.t_strings[0]).mean() - c) 
        feature /= c
        return feature  
    
class Mom(Feature):    
    def __call__(self, data):
        if self._one_minute:
            feature = data['Close'] - data['Open']
        else:
            feature = data['Close'].rolling(self.t_strings[0]).apply(
                lambda x: (x[-1] - x[0]),
                raw=True
            )
        feature /= data['Close']
        return feature
    
class Range(Feature):
    def __call__(self, data):
        if self._one_minute:
            feature = data['High'] - data['Low']
        else:
            raise ValueError('Not implemented!')
            '''
            feature = data['Close'].rolling(self.t_strings[0]).apply(
                lambda x: (x[-1] - x[0]) / x[-1],
                raw=True
            )
            '''
        feature /= data['Close']
        return feature
 
class Skew(Feature):
    def __call__(self, data):
        feature = data['Close'].rolling(self.t_strings[0]).skew()
        return feature    

class Std(Feature):
    def __call__(self, data):
        c = data['Close']
        feature = c.rolling(self.t_strings[0]).std()
        feature /= c
        return feature

class StochOsc(Feature):
    def __call__(self, data):
        h = data['High'].rolling(self.t_strings[0]).max()
        l = data['Low'].rolling(self.t_strings[0]).min()
        so = (data['Close'] - l) / (h - l)
        m = so.rolling('3min').mean()
        feature = so - m
        return feature
    
class Trend(Feature):
    def __call__(self, data):
        trend = data['Close'] - data['Open']
        feature = trend.rolling(self.t_strings[0]).mean() #/ data['Close']
        return feature     
