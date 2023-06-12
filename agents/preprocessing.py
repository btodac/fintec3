#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:09:23 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

class ObservationBuilder(object):
    
    def __init__(self, columns, back_features=None):
        self.columns = columns
        self._feature_generators = {}
        for c in columns:
            self._feature_generators[c] = self._parse_feature_name(c)
        self.back_features = back_features
    
    @property
    def shape(self):
        return (len(self.columns), self.back_features[0])
    
    def make_observations(self, data, opening_time, closing_time, tz):
        data = data.copy()
        if data.index.tz != tz:
            data.index = data.index.tz_convert(tz)
        data = data.between_time(opening_time.time(), closing_time.time())
        features = self._calculate_features(data)
        if self.back_features is not None:
            f = []
            for i in range(self.back_features[0]):
                dti = features.index - pd.Timedelta(minutes=self.back_features[1])
                is_nat = dti.time < features.index.time.min()
                dtmin = features.index.floor(str(self.back_features[1]) + 'min')
                dti = dti.to_numpy()
                dti[is_nat] = dtmin[is_nat]
                dti = pd.DatetimeIndex(dti)
                f.append(features.loc[dti,:].to_numpy())
            observations = np.stack(f, axis=2)
        else:
            observations = features.to_numpy()
        observations[np.isnan(observations)] = 0
        return observations, data.index
    
    def _calculate_features(self, data):
        features = []
        for column, feature_gen in self._feature_generators.items():
            features.append(feature_gen(data))
        return pd.concat(features, axis=1)
        
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

class AvgTrueRange(Feature):
    def __call__(self, data):
        if self._one_minute:
            feature = data['High'] - data['Low']
        else:
            feature = (data['High'] - data['Low']).rolling(self.t_strings[0]).mean()
            
        feature /= data['Close']
        return feature
    
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

class Median(Feature):
    def __call__(self, data):
        if self._one_minute:
            feature = 0.5 * (data['High'] + data['Low'])
        else:
            feature = data['Close'].rolling(self.t_strings[0]).median()    
        feature -= data['Close']
        feature /= data['Close']
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

class STrend(Feature):
    def __call__(self, data):
        atr = (data['High'] - data['Low']).rolling(self.t_strings[0]).mean()
        atr = self.t_strings[1] * atr
        med = 0.5 * (data['High'] + data['Low'])
        up = med - atr
        down = med + atr
        y = pd.concat((up,down, data['Close']))
        
        
        direction = pd.Series(np.nan * np.ones(len(up)), index = up.index)
        direction.iloc[up>=0] = -1
        direction.iloc[down<=0] = 1
        #up = up - data['Close']
        #down = down - data['Close']
        return up, down
        
        
class Trend(Feature):
    def __call__(self, data):
        trend = data['Close'] - data['Open']
        feature = trend.rolling(self.t_strings[0]).mean() #/ data['Close']
        return feature     
