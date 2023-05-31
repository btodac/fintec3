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
        '''
        BayesAgent is a subclass of Agent the uses a naive Bayes
        classifier to make predictions

        '''
        if params is None:
            params = PARAMS[ticker]
        super().__init__(ticker, columns, params=params, observer=observer,
                         target_generator=target_generator)
            
    def make_prediction(self, observations):
        # Make sure single observations have correct shape
        if len(observations.shape) != 2:
            observations = observations.squeeze()[np.newaxis,:]
        # Calculate posteriors
        probs = []
        for k, prior in self.priors.items():
            pde_vals = []
            for i, d in enumerate(self.distributions[k]):
                pde_vals.append(d.logpdf(observations[:,i].squeeze()))
            pde_vals = np.stack(pde_vals).T
            if len(pde_vals.shape) == 1:
                pde_vals = pde_vals[np.newaxis,:]
            probs.append(np.log(prior) + np.sum(pde_vals, axis=1))
        probs_log = np.stack(probs).T.copy()
        probs = np.exp(probs_log)#np.abs(1/probs)
        is_zero = (probs==0).any(axis=1)
        probs[is_zero, probs[is_zero,:].argmax(axis=1)] = 1
        probs /= probs.sum(axis=1, keepdims=True) #probs.sum(axis=1, keepdims=True)
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
        # TEST
        '''
        c = np.stack(classes.values()).T
        print(c.sum(axis=0))
        indx_inc = np.nonzero(c[:,0])[0]
        indx_dec = np.nonzero(c[:,1])[0]
        indx_flat = np.nonzero(c[:,2])[0]
        indx_1 = np.random.choice(indx_inc,size=5000,replace=False)
        indx_2 = np.random.choice(indx_dec,size=5000,replace=False)
        indx_3 = np.random.choice(indx_flat,size=8000,replace=False)
        indx = np.concatenate((indx_1,indx_2,indx_3))
        observations = observations[indx,:]
        for k, t in classes.items():
            classes[k] = t[indx]
        self.priors = {
            'Increasing' : classes['Increasing'].mean(),
            'Decreasing' : classes['Decreasing'].mean(),   
            'Flat' : classes['Flat'].mean()
            }
        print(self.priors) ### TODO
        '''
        # END OF TEST
        
        # Get pde kernals for the variables
        distributions = {}
        for key, c in classes.items(): #[is_stop_loss, is_take_profit, is_time_out]:
            dists = []
            for i, f in enumerate(columns):
                f = f.lower()
                if 'mom' in f or 'mean' in f or 'meandiff' in f\
                     or 'kurt' in f: #or 'std' in v
                    p = stats.laplace_asymmetric.fit(observations[c,i])
                    dists.append(stats.laplace_asymmetric(*p))
                elif 'std' in f or 'skew' in f or 'stochosc' in f\
                    or 'trend' in f:
                    p = stats.skewnorm.fit(observations[c,i])
                    dists.append(stats.skewnorm(*p))
                else:
                    raise ValueError(f'Unkown feature: {f}')
                
            distributions[key] = dists
        
        return distributions 
