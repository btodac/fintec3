#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 21:58:52 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

def summarise_outcomes(orders: pd.DataFrame) -> pd.DataFrame:
    '''
    Tilts the data in the orders dataframe to provide show overall results

    Parameters
    ----------
    orders : pd.DataFrame
        A dataframe containing the outcomes of a set of orders. Must include columns:
            'Profit', 'Take_Profit_Hit', 'Stop_Loss_Hit', 'Time_Limit_Hit'

    Returns
    -------
    pd.DataFrame
        A dataframe giving the following properties of the orders:
            'Take_Profit %', 'Stop_Loss %', 'Time_Limit %', 'N_Trades',
            'Profit', 'Avg. Profit', 'Avg. Win', 'Avg. Lose', 'Avg. Timeout'
            'Profit Factor', 'Max Drawdown', 'Gradient'

    '''       
    if len(orders) == 0:
        d = {'Take_Profit %' : np.NaN, 'Stop_Loss %' : np.NaN, 'Time_Limit %' : np.NaN,
             'N_Trades' : np.NaN, 'Profit' : np.NaN, 'Avg. Profit' : np.NaN, 'Avg. Win': np.NaN,
             'Avg. Lose' : np.NaN, 'Avg. Timeout' : np.NaN,  'Profit Factor' : np.NaN,
             'Max Drawdown' : np.NaN, 'Gradient' : np.nan}
        return pd.DataFrame(d, index=[0])
    
    pt = {}
    to_percentage = lambda x : np.sum(x) / len(x) * 100
    
    for p in ['Take_Profit_Hit', 'Stop_Loss_Hit','Time_Limit_Hit']:
        n = p.replace('_Hit', ' %')
        pt[n] = to_percentage(orders[p])

    pt['N_Trades'] = len(orders)
    pt['Win Ratio'] = sum(orders['Profit'] > 0) / sum(orders['Profit'] < 0)
    pt['Profit'] = orders['Profit'].sum()
    pt['Profit (day)'] = orders['Profit'].resample('1B',closed='right',label='right').sum().mean()
    pt['Profit (week)'] = orders['Profit'].resample('1W',closed='right',label='right').sum().mean()
    pt['Avg. Profit'] = pt['Profit'] / pt['N_Trades']
    pt['Avg. Win'] = orders.iloc[orders.loc[:,'Take_Profit_Hit'].to_numpy()].loc[:,'Profit'].mean()
    pt['Avg. Lose'] = orders.iloc[orders.loc[:,'Stop_Loss_Hit'].to_numpy()].loc[:,'Profit'].mean()
    pt['Avg. Timeout'] = orders.iloc[orders.loc[:,'Time_Limit_Hit'].to_numpy()].loc[:,'Profit'].mean()
    profit = orders.iloc[orders.loc[:,'Profit'].to_numpy() > 0].loc[:,'Profit'].sum()
    loss = orders.iloc[orders.loc[:,'Profit'].to_numpy() < 0].loc[:,'Profit'].sum()
    pt['Profit Factor'] = (-profit/loss) #.fillna(np.inf)
    
    rp = np.cumsum(orders['Profit'])
    rp_min = [rp[i:][rp[i:].argmin()] - rp[i] for i in range(len(rp))]
    pt['Max Drawdown'] = min(rp_min) 
    
    xa = np.linspace(0,1,len(orders))
    xx = np.stack((xa,np.ones_like(xa))).T
    A,_,_,_ = np.linalg.lstsq(xx, rp.to_numpy()[:,np.newaxis], rcond=-1)
    pt['Gradient'] = A[0,0]
    
    df = pd.DataFrame(pt, index=[0])
    return df