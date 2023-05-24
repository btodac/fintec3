#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:47:23 2023

@author: mtolladay
"""
import logging

import numpy as np
from scipy import stats
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from utillities.timesanddates import get_ticker_time_zone, opening_and_closing_times

log = logging.getLogger(__name__)

class InterdayBayesModel(object):
    def __init__(self, data, ticker, columns):
        data = data.copy()
        self._params = { 
            'ticker' : ticker,
            'columns' : columns,
            'tz' : get_ticker_time_zone(ticker),
            'up' : 50,
            'down' : 50,
            #'to' : 30,
            }
        
        self.fit(data)
        
    def __call__(self, data):
        data = data.copy()
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz']) 
        data = data.between_time(self._opening_time.time(), 
                                 self._closing_time.time())
        observations = self._build_observation(data, self._params['columns'])
        observation = observations[-1,:]
        log.debug(observation)
        probs = []
        for k, prior in self.priors.items():
            pde_vals = []
            for i, d in enumerate(self.var_dists[k]):
                pde_vals.append(d.logpdf(observation[i]))

            probs.append(np.log(prior) + np.sum(pde_vals))
        pred = np.argmax(probs)

        return {0 : 'BUY', 1 : 'SELL', 2 : 'HOLD'}[pred]
        
        
    def fit(self, data):
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz']) 
            
        observations, order_datetimes = self.make_observations(
            data, self._params['ticker'], self._params['columns'], 
            )
        classes, self.priors = self._make_classes(
            data['Close'], order_datetimes, self._params['up'],
            self._params['down'], #self._params['to']
            )
        self.var_dists = self._make_pd_kernels(
            observations, classes, self._params['columns']
            )
        
        pred, _, _ = self.predict(data, self._params['ticker'])
        actual = np.argmax(np.stack(list(classes.values())), axis=0)
        for i, t in enumerate(["Long","Short","Hold"]):
            p = (np.sum((pred==i) * (actual==i)) / np.sum(pred==i))
            r = (np.sum((pred==i) * (actual==i)) / np.sum(actual==i))
            print(f'{t} : precision = {p}, recall = {r}')
    
    def predict(self, data, ticker):
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz']) 
        observations, order_datetimes = self.make_observations(
            data, ticker, self._params['columns']
            )
        
        # Standard 
        probs = []
        for k, prior in self.priors.items():
            pde_vals = []
            for i, d in enumerate(self.var_dists[k]):
                # standard
                #pde_vals.append(d.pdf(observations[:,i]))
                # log probs
                pde_vals.append(d.logpdf(observations[:,i].squeeze()))
            # Standard
            #probs.append(prior * np.prod(np.stack(pde_vals).T, axis=1))
            # Log probs
            probs.append(np.log(prior) + np.sum(np.stack(pde_vals).T, axis=1))
        probs = np.stack(probs).T
        pred = probs.argmax(axis=1)
        d = probs.copy()
        d[probs==-np.inf] = 0
        d /= d.sum(axis=1, keepdims=True)
        probs = d
        return pred, probs, order_datetimes
    
                
    def make_observations(self, data, ticker, columns):
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz']) 
        opening_time, closing_time = self._get_tradeable_time(ticker)
        data = data.between_time(opening_time.time(), closing_time.time())
        observations = self._build_observation(data, ticker, columns)
        order_datetimes = [pd.Timestamp.combine(date=o.date(), time=closing_time.time()) \
                           #- pd.Timedelta(minutes=1)
                               for o in np.unique(data.index.normalize())]
        return observations[:-1], order_datetimes[:-1]

    def _build_observation(self, data, ticker, columns):
        if data.index.tz != self._params['tz']:
            data.index = data.index.tz_convert(self._params['tz']) 
        opening_time, closing_time = self._get_tradeable_time(ticker)
        data = data.between_time(opening_time.time(), closing_time.time())
        
        features = []
        for c in columns:
            feature = None
            if 'mom' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:,'Close'].rolling(t_string).apply(
                    lambda x : x[-1] - x[0], 
                    raw=True
                    ) 
            elif 'mean_diff' in c:
                t_str1, t_str2 = c.split('min_')[:2]
                m1 = data.loc[:,'Close'].rolling(t_str1 + 'min').mean()
                m2 = data.loc[:,'Close'].rolling(t_str2 + 'min').mean()
                feature = m1 - m2
            elif 'mean' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:,'Close'].rolling(t_string).mean().rolling("2min").apply(
                    lambda x: x[-1] - x[0],
                    raw=True
                    )
            elif 'std' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:,'Close'].rolling(t_string).std()
            elif 'skew' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:,'Close'].rolling(t_string).skew()  
            elif 'kurt' in c:
                t_string = c.split('_')[0]
                feature = data.loc[:,'Close'].rolling(t_string).kurt()

            if feature is not None:
                features.append(feature)
        
        features = pd.concat(features, axis=1)
            
        dates = np.unique(features.index.date)
        time_range = pd.date_range(opening_time, closing_time, freq="180min")
        time_range = time_range.append(pd.DatetimeIndex([closing_time]))
        times = [d.time() for d in time_range]
        data = pd.concat([features.at_time(t) for t in times])
        #data = pd.concat([features.at_time(t) for t in times[-2:]])
        data = data.sort_index()
        indx = np.stack([np.where(data.index.date == da)[0] for da in dates])
        #cols = data.columns
        observations = data.to_numpy()
        observations[np.isnan(observations)] = 0
        observations = observations[indx]
        observations = observations.reshape(observations.shape[0], -1)
        
        return observations
    
    def _make_classes(self, prices, order_datetimes, up, down):
        index = np.array(
            [prices.index.get_loc(
                k.tz_localize(get_ticker_time_zone(self._params['ticker']))
                ) for k in order_datetimes]
            )
        p_open = prices.loc[prices.index[index+1],:].to_numpy()
        p_close = prices.loc[prices.index[index],:].to_numpy()
        traces = (p_open - p_close).squeeze()

        is_increasing =  traces > up
        is_decreasing = traces < -down
                              
        is_flat = np.logical_not(np.logical_or(is_increasing, is_decreasing))

        classes = {
            'Increasing' : is_increasing,
            'Decreasing' : is_decreasing,
            'Flat' : is_flat,
            }

        priors = {
            'Increasing' : is_increasing.mean(),
            'Decreasing' : is_decreasing.mean(),   
            'Flat' : is_flat.mean()
            }
        print(priors)
        return classes, priors
    
    def _make_pd_kernels(self, observations, classes, columns):
        # Get pde kernals for the variables
        var_dists = {}
        for key, c in classes.items(): #[is_stop_loss, is_take_profit, is_time_out]:
            dists = []
            for i, v in enumerate(
                    np.repeat(columns, int(observations.shape[1] /  len(columns)))
                    ):
                #dists.append(stats.gaussian_kde(observations[c,i].squeeze()))
                
                if 'mom' in v or 'mean' in v or 'mean_diff' in v\
                    or 'std' in v or 'kurt' in v:
                    p = stats.laplace_asymmetric.fit(observations[c,i])
                    dists.append(stats.laplace_asymmetric(*p))
                    #p = stats.laplace.fit(observations[c,i])
                    #dists.append(stats.laplace(*p))
                elif 'skew' in v:
                    p = stats.exponnorm.fit(observations[c,i])
                    dists.append(stats.exponnorm(*p))
                #elif 'std' in v or 'kurt' in v:
                #    p = stats.laplace_asymmetric.fit(observations[c,i])
                #    dists.append(stats.laplace_asymmetric(*p))
                
            var_dists[key] = dists
        
        return var_dists     
        
    def _get_tradeable_time(self, ticker):
        opening_time, closing_time = opening_and_closing_times(ticker)
        opening_time = opening_time.tz_convert(self._params['tz'])
        closing_time = closing_time.tz_convert(self._params['tz'])
        #closing_time -= pd.Timedelta(minutes=self._params['to'])
        self._opening_time = opening_time
        self._closing_time = closing_time
        return opening_time, closing_time
    
def interday_results(model, data):
    ticker = model._params['ticker']
    tz = get_ticker_time_zone(ticker)
    pred, _, _, = model.predict(data, ticker)
    opening_time, closing_time = opening_and_closing_times(ticker)
    opening_time = opening_time.tz_convert(tz)
    closing_time = closing_time.tz_convert(tz)
    dates = np.unique(data.index.date)
    open_index = [
        pd.Timestamp.combine(date=d, time=opening_time.time()).tz_localize(tz)
        for d in dates
        ] 
    close_index = [
        pd.Timestamp.combine(date=d, time=closing_time.time()).tz_localize(tz)
        for d in dates
        ]
    p_open = data.loc[open_index[1:],'Open'].to_numpy()
    p_close = data.loc[close_index[:-1],'Close'].to_numpy()
    traces = (p_open - p_close).squeeze()
    directions = pred.copy()
    directions[pred==0] = 1
    directions[pred==1] = -1
    directions[pred==2] = 0
    profits = directions * traces
    return profits
    