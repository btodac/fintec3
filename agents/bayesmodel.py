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

log = logging.getLogger(__name__)

class BayesModel(object):
    '''    
    BayesModel provides an interface to a naive Bayes
    classifier to make predictions
    '''
        
    def __init__(self, columns,):
        '''

        Parameters
        ----------
        columns : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self._columns = columns
        self.dist_functions = self._create_model(columns)
        
        self._is_fit = False
        self.priors = None
        self.distributions = None

    def _create_model(self, columns):
        dists = {}
        for c in columns:
            f = c.split('_')[-1].lower()
            if f in ['kurt','mean','meandist','meandiff',
                     'median','mom','skew','trend',]:
                dists[c] = stats.nct
            elif f == 'High':
                dists[c] = stats.truncexpon
            elif f == 'Low':
                dists[c] = stats.truncexpon
            elif f in  ['range','std','avgtruerange']:
                dists[c] = stats.invweibull
            elif f == 'stochosc':
                dists[c] = stats.laplace_asymmetric
            else:
                raise ValueError(f'Unkown feature: {f}')
        return dists
    
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
    
    def fit(self, observations, targets, valildation_data) -> dict:
        '''
        Fits a Naive Bayes Classifier model to the training data

        Parameters
        ----------
        training_data : pd.DataFrame
            OHLC data to train model
        validation_data : pd.DataFrame
            OHLC data to test model (NOT USED)


        '''
        if not self._is_fit:
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
            print(self.priors) 
    
            distributions = {}
            for target_class, is_target in classes.items():
                class_dists = []
                for i, column in enumerate(self._columns):
                    o = observations[is_target,i]
                    o = o[o!=0]
                    p = self.dist_functions[column].fit(o)
                    class_dists.append(self.dist_functions[column](*p))
                distributions[target_class] = class_dists
            
            self.distributions = distributions
            
            self._is_fit = True

    def __reduce__(self):
        if self._is_fit:
            state = {
                'priors' : self.priors,
                'distributions' : self.distributions,
                '_is_fit' : self._is_fit}
            return type(self), (self._columns,),  state
        else:
            return type(self), (self._columns,)   
                