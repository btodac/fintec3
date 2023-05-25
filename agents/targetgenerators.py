#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:17:13 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd
from agenttesting.results import OutcomeSimulator

class TrendBasedTargetGen(object):
    
    def __init__(self, up, down, time_limit):
        self.up = up
        self.down = down
        self.time_limit = time_limit
    
    def get_targets(self, data, order_datetimes):
        prices = data['Close']
        order_index = prices.index.get_indexer(order_datetimes)
        indx_i = order_index[:,np.newaxis]
        indx_r = np.arange(self.time_limit)[np.newaxis,:]
        index = indx_i + indx_r
        p = prices.to_numpy()
        traces = p[index].squeeze()
        zeroing = traces[:,0][:,np.newaxis]
        traces = (traces - zeroing) / zeroing
        ups = self.up / zeroing
        downs = self.down / zeroing
        tx = 1.2 * self.time_limit

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
    
class OrderBasedTargetGen(object):
    
    def __init__(self, ticker, take_profit, stop_loss, time_limit, both_as_hold=True):
        self.ticker = ticker
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.time_limit = time_limit
        self.both_as_hold = True
        
    def get_targets(self, data, order_datetimes,):
        outcome_simulator = OutcomeSimulator()
        outcome_simulator = OutcomeSimulator()
        orders = pd.DataFrame()
        orders.loc[:, 'Opening_Value'] = data.loc[order_datetimes, 'Close']
        orders.loc[:, 'Ticker'] = self.ticker
        orders.loc[:, 'Take_Profit'] = self.take_profit
        orders.loc[:, 'Stop_Loss'] = self.stop_loss
        orders.loc[:, 'Time_Limit'] = self.time_limit
    
        # buys
        orders.loc[:, 'Direction'] = "BUY"
        buy_orders = outcome_simulator.limit_times(orders, data)
        is_tp_buy, _, _ = outcome_simulator.outcome_bool(buy_orders)
        # sells
        orders.loc[:, 'Direction'] = "SELL"
        sell_orders = outcome_simulator.limit_times(orders, data)
        is_tp_sell, _, _ = outcome_simulator.outcome_bool(sell_orders)
    
        is_buy = np.logical_and(
            is_tp_buy,
            np.logical_not(is_tp_sell)
        )
        is_sell = np.logical_and(
            is_tp_sell,
            np.logical_not(is_tp_buy)
        )
        is_both = np.logical_and(
            is_tp_buy,
            is_tp_sell
        )
        if self.both_as_hold:
            is_hold = np.logical_not(np.logical_xor(is_buy, is_sell))
        else:
            buy_faster = buy_orders['Take_Profit_Time'].to_numpy() \
                > sell_orders['Take_Profit_Time'].to_numpy()
            is_buy[np.logical_and(is_both, np.logical_not(buy_faster))] = False
            is_sell[np.logical_and(is_both, buy_faster)] = False
            is_hold = np.logical_not(np.logical_or(is_buy, is_sell))
        probs = np.stack((is_buy, is_sell, is_hold)).T
        return probs