#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:50:52 2023

@author: mtolladay
"""
import logging

import numpy as np
from scipy import stats
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from agents import Agent
from agenttesting.results import Results

log = logging.getLogger(__name__)

PARAMS = {
    '^DJI' : {
        # Position paramaters
        'take_profit' : 75,
        'stop_loss' : 20,
        'time_limit' : 30,
        # Training params
        'up' : 20,#20,
        'down' : -20,#20,
        'to' : 15,
        },
    "^NDX" : {
        'take_profit' : 40,
        'stop_loss' : 10,
        'time_limit' : 30,
        'up' : 40,
        'down' : -40,
        'to' : 30,
        'live_tl' : 60
        },
    '^GDAXI' : {
        'take_profit' : 40,
        'stop_loss' : 10,
        'time_limit' : 30,
        'up' : 40,
        'down' : -40,
        'to' : 60,
        'live_tl' : 60,
        },
    '^FCHI' : {
        'take_profit' : 17,
        'stop_loss' : 4,
        'time_limit' : 30,
        'up' : 5,
        'down' : -5,
        'to' : 15,
        },
    '^FTSE' : {
        'take_profit' : 15,
        'stop_loss' : 3,
        'time_limit' : 30,
        'up' : 10,
        'down' : -10,
        'to' : 30,
        },
    'EUR=X' : {
        'take_profit' : 30 * 1e-8,
        'stop_loss' : 5 * 1e-8,
        'time_limit' : 30,
        'up' : 0.0005,
        'down' : -0.0005,
        'to' : 20,
        }
    }

class BayesAgent(Agent):
    def __init__(self, ticker, columns, params=None, observer=None,
                 target_generator=None):
        if params is None:
            params = PARAMS[ticker]
        super().__init__(ticker, columns, params=params, observer=observer,
                         target_generator=target_generator)
            
    def make_prediction(self, observations):
        if len(observations.shape) != 2:
            observations = observations.squeeze()[np.newaxis,:]

        probs = []
        for k, prior in self.priors.items():
            pde_vals = []
            for i, d in enumerate(self.distributions[k]):
                pde_vals.append(d.logpdf(observations[:,i].squeeze()))
            probs.append(np.log(prior) + np.sum(np.stack(pde_vals).T, axis=1))
        probs = np.stack(probs).T
        probs = np.abs(1/probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs
    
    def fit(self, training_data, validation_data):
        observations, targets, _ = self.get_observations_and_targets(training_data)
        classes = {
            'Increasing' : targets[:,0].squeeze(),
            'Decreasing' : targets[:,1].squeeze(),
            'Flat' : targets[:,2].squeeze(),
            }
        self.priors = {
            'Increasing' : targets[:,0].mean(),
            'Decreasing' : targets[:,1].mean(),   
            'Flat' : targets[:,2].mean()
            }
        print(self.priors) ### TODO
        self.distributions = self._make_pdfs(
            observations, classes, self._params['columns']
            )
        
        return self.priors
    
    def _make_pdfs(self, observations, classes, columns):
        # Get pde kernals for the variables
        distributions = {}
        for key, c in classes.items(): #[is_stop_loss, is_take_profit, is_time_out]:
            dists = []
            for i, f in enumerate(columns):
                if 'mom' in f or 'mean' in f or 'mean_diff' in f\
                     or 'kurt' in f: #or 'std' in v
                    p = stats.laplace_asymmetric.fit(observations[c,i])
                    dists.append(stats.laplace_asymmetric(*p))
                elif 'std' in f or 'skew' in f or 'stoch_osc' in f\
                    or 'trend' in f:
                    p = stats.skewnorm.fit(observations[c,i])
                    dists.append(stats.skewnorm(*p))
                else:
                    raise ValueError(f'Unkown feature: {f}')
                
            distributions[key] = dists
        
        return distributions 

    def _get_targets(self, data, order_datetimes):
        prices = data['Close']
        order_index = prices.index.get_indexer(order_datetimes)
        indx_i = order_index[:,np.newaxis]
        indx_r = np.arange(self._params['to'])[np.newaxis,:]
        index = indx_i + indx_r
        p = prices.to_numpy()
        traces = p[index].squeeze()
        zeroing = traces[:,0][:,np.newaxis]
        traces = (traces - zeroing) / zeroing
        ups = self._params['up'] / zeroing
        downs = self._params['down'] / zeroing
        tx = 1.2 * self._params['to']

        is_buy = np.logical_and( 
                                traces[:,-1] > ups.squeeze(),
                                np.mean(traces[:,1:] >= ups / tx, axis=1) >= 0.6,
                            )
        is_sell = np.logical_and( 
                                traces[:,-1] < downs.squeeze(),
                                np.mean(traces[:,1:] <= downs / tx, axis=1) >= 0.6,
                            )#traces[:,-1] < down
        is_hold = np.logical_not(np.logical_xor(is_buy, is_sell))
        targets = np.stack((is_buy, is_sell, is_hold)).T
        return targets
