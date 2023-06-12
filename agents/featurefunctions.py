#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 13:47:53 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

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
            
            #feature = data['Close'].rolling(self.t_strings[0]).apply(
            #    lambda x: (x[-1] - x[0]) / x[-1],
            #    raw=True
            #)
            
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