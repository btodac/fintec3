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
            return (len(self.columns), self.back_features[0],)
        else:
            return (len(self.columns),)
    
    def make_observations(self, data, opening_time, closing_time, tz):
        data = data.copy()
        if data.index.tz != tz:
            data.index = data.index.tz_convert(tz)
        data = data.between_time(opening_time.time(), closing_time.time())
        features = self._calculate_features(data)
        if self.back_features is not None:
            f = self._create_back_features(features)
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
    
    def _create_back_features(self, features):
        features_copy = features.copy()
        f = []
        dt = pd.Timedelta(minutes=self.back_features[1])
        for i in range(self.back_features[0]):
            dti = features_copy.index - dt
            is_nat = dti.time < features_copy.index.time.min()
            dti = dti.to_series()
            if all(is_nat):
                dti.iloc[is_nat] = features_copy.index[0]
            else:
                dti.iloc[is_nat] = pd.NaT
                dti = dti.bfill()
            dti = pd.DatetimeIndex(dti)
            features_copy = features.loc[dti,:]
            f.append(features_copy.to_numpy())
        return f
        
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
        
if __name__ == "__main__":
    from utillities.datastore import Market_Data_File_Handler
    from utillities.timesanddates import get_ticker_time_zone

    save_model = True
    ticker = "^NDX"
    # Observation parameters
    columns = [
            '10min_AvgTrueRange',
            '20min_WeightedTrend', '10min_WeightedTrend',
            #'10min_Std',
            '10min_20min_MeanDiff', '10min_120min_MeanDist',
            '15min_StochOsc',
            #'60min_MeanDist',#'120min_MeanDist','240min_MeanDist'
        ]

    data_file = Market_Data_File_Handler(dataset_name="all")
    all_data = data_file.get_ticker_data(ticker, as_list=False)
    split_time = pd.Timestamp("2022-06-22", tz='UTC')
    training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
   # validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

    tz = get_ticker_time_zone(ticker) #'^GDAXI'
    training_data = training_data.tz_convert(tz)
    #validation_data = validation_data.tz_convert(tz)
    training_data = training_data.between_time(
        "09:30", "16:00", # NDX
        #"09:30", '17:30' # GDAXI
        )
    
    observer = ObservationBuilder(columns, (5,1))
    observations = observer.make_observations(
        training_data, 
        training_data.index[0], 
        training_data.index[-1], 
        training_data.index.tz,
        )

