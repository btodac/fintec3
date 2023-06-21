#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:09:23 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

from agents import featurefunctions

class ObservationBuilder(object):
    
    def __init__(self, columns, back_features=None):
        self.columns = columns
        self._feature_generators = {}
        for c in columns:
            self._feature_generators[c] = self._parse_feature_name(c)
        self.back_features = back_features
    
    @property
    def shape(self):
        if self.back_features is not None:
            return (len(self.columns), self.back_features[0])
        else:
            return len(self.columns)
    
    def make_observations(self, data, opening_time, closing_time, tz):
        data = data.copy()
        if data.index.tz != tz:
            data.index = data.index.tz_convert(tz)
        data = data.between_time(opening_time.time(), closing_time.time())
        features = self._calculate_features(data)
        if self.back_features is not None:
            f = []
            for i in range(self.back_features[0]):
                t = i * self.back_features[1]
                dti = features.index - pd.Timedelta(minutes=t)
                is_nat = dti.time < features.index.time.min()
                dti = dti.to_series()
                dti.iloc[is_nat] = pd.NaT
                dti = dti.bfill()
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
            cls = getattr(featurefunctions, feature)
            #f = globals()[feature](t_strings)
        except NameError:
            raise NameError(f'Unknown feature {feature}')    
        else:
            return cls(t_strings)
        
