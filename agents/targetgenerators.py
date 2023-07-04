#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:17:13 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd
from agenttesting.results import OutcomeSimulator

class VelocityBasedTargetGen(object):
    
    def __init__(self, up, down, time_limit):
        if np.sign(up) != 1:
            raise ValueError("Sign of argument up must be positive")
        if np.sign(down) != -1:
            raise ValueError("Sign of argument down must be negative")
        self.up = up
        self.down = down
        self.time_limit = time_limit
    
    def get_targets(self, data, order_datetimes):
        velocity = data['Close'] - data['Open']
        order_index = data.index.get_indexer(order_datetimes)
        indx_i = order_index[:,np.newaxis]
        indx_r = np.arange(self.time_limit)[np.newaxis,:]
        index = indx_i + indx_r
        v = velocity.to_numpy()
        v = v[index].squeeze()
        
        is_buy = np.logical_and(
            np.mean(v, axis=1) >= self.up,
            np.argmin(v.cumsum(axis=1), axis=1) == 0,
            )
        is_sell = np.logical_and( 
            np.mean(v, axis=1) <= self.down,
            np.argmax(v.cumsum(axis=1), axis=1) == 0,
            )
        is_hold = np.logical_not(np.logical_xor(is_buy, is_sell))
        targets = np.stack((is_buy, is_sell, is_hold)).T
        return targets

class TrendBasedTargetGen(object):
    
    def __init__(self, up, down, time_limit, up_down_ratio=0.6):
        if np.sign(up) != 1:
            raise ValueError("Sign of argument up must be positive")
        if np.sign(down) != -1:
            raise ValueError("Sign of argument down must be negative")
        self.up = up
        self.down = down
        self.time_limit = time_limit
        self.up_down_ratio = up_down_ratio
    
    def get_targets(self, data, order_datetimes):
        prices = data['Close']
        order_index = prices.index.get_indexer(order_datetimes)
        indx_i = order_index[:,np.newaxis]
        indx_r = np.arange(self.time_limit)[np.newaxis,:] + 1
        index = indx_i + indx_r
        p = prices.to_numpy()
        traces = p[index].squeeze()
        try:
            zeroing = traces[:,0][:,np.newaxis]
        except IndexError as ie:
            if traces.ndim == 1:
                traces = traces[:,np.newaxis]
                zeroing = traces[:,0][:,np.newaxis]
            else:
                raise ie
        traces = (traces - zeroing) / zeroing
        ups = self.up / zeroing
        downs = self.down / zeroing # NOTE: downs must be negative!
        deltas = traces[:,1:] - traces[:,:-1]
        tx = 10 * self.time_limit
        min_up_delta = (ups / tx)
        min_down_delta = (downs / tx)
        
        is_buy = np.logical_and(
            # First check that the final value is gt the up value
            traces[:,-1] > ups.squeeze(),
            # Second check the change over time is generally greater than the min change
            #np.mean(deltas >= 0, axis=1) >= 0.60,
            np.mean(deltas >= min_up_delta, axis=1) >= self.up_down_ratio, #0.50,
            )
        is_sell = np.logical_and( 
            traces[:,-1] < downs.squeeze(),
            #np.mean(deltas <= 0, axis=1) >= 0.60,
            np.mean(deltas <= min_down_delta, axis=1) >= self.up_down_ratio,
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

class WTrendBasedGenerator(object):
    def __init__(self, ticker, take_profit=50, time_limit=20, min_trend=1.0):
        self.ticker = ticker
        self.time_limit = time_limit
        self.min_trend = min_trend
        
    def get_targets(self, data, order_datetimes,):
        from agents.featurefunctions import WeightedTrend
        t_string = f"{self.time_limit}min"
        tgen = WeightedTrend([t_string])
        trend = tgen(data)
        valid_trend = np.zeros_like(trend.to_numpy())
        valid_trend[trend.to_numpy() > self.min_trend] = 1
        valid_trend[trend.to_numpy() < -self.min_trend] = -1
        # find changes in valid_trend
        changes = np.concatenate((np.array([True]), np.diff(valid_trend) != 0))
        
        
        trend['gt'] = abs(trend.to_numpy()) > self.min_trend
        trend['sign'] = np.sign(trend[self.ticker])
        trend['open'] = data['Open']
        trend['low'] = data['Low'].iloc[::-1].rolling(t_string,).min().iloc[::-1] - data['Open']
        trend['high'] = data['High'].iloc[::-1].rolling(t_string).max().iloc[::-1] - data['Open']
        trend['sell'] = trend['low'] < -self.take_profit and trend[self.ticker]
        self.trend = trend
        #data.index.get_indexer(order_datetimes)
        
        
    

class TargetGenerator(object):
    def __init__(self, conditions: dict):
        
        self.conditions = {}
        for condition, params in conditions.items():
            self.conditions[condition] = self._parse_condition_name(condition, params)
            
    def get_targets(self, data, order_datetimes):
        pass#is_
            
    def _parse_condition_name(self, condition, params):
        try:
            f = globals()[condition](params)
        except KeyError:
            raise KeyError(f'Unknown condition {condition}')    
        return f

class DeltaGT:
    def get_condition(self, values, comparators):
        return values[:,-1] > comparators.squeeze()
        