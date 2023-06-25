#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:16:08 2022

@author: mtolladay
"""
import string
import logging

import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpl
import pandas as pd

from agenttesting.outcomeprocessor import OutcomeSimulator
from utillities.timesanddates import get_ticker_time_zone, opening_and_closing_times

class Results(object):
    def __init__(self, predictions: np.array, order_datetimes: pd.DatetimeIndex,
                 ticker: string, take_profit: int, stop_loss: int, time_limit: int,
                 data: pd.DataFrame, cooldown: int = 0, max_orders: int = 1,
                 ):
        '''
        Class that calculates the results of a set of predictictions. It first 
        calculates the results of all passed orders at the provided datetimes
        (Results.signal_outcomes) and then applies the cooldown and max_orders 
        to provide a set of outcomes within these constraints 
        (Results.position_outcomes).

        Parameters
        ----------
        predictions : np.array
            A numpy array containg values [0,1,2] indicating BUY, SELL or HOLD
        order_datetimes : pd.DatetimeIndex
            The datetimes corresponding to the predictions
        ticker : string
            The ticker for the tradable financial product
        take_profit : int
            Take profit distance in points
        stop_loss : int
            Stop loss distance in points
        time_limit : int
            Time limit in minutes
        data : pd.DataFrame
            The market data containing OHLC data for the ticker covering the 
            required time 
        cooldown : int
            The number of minutes to wait after an order hits its stop loss before opening
            another order. Default is 0
        max_orders : int
            The maximum number of simultaneously open positions. Default is 1

        Returns
        -------
        None.

        '''
        self.ticker = ticker
        
        self.orders = self._construct_orders(
            predictions, order_datetimes, ticker, take_profit, stop_loss, time_limit, data)
        self._calculate_results(data, cooldown=cooldown, max_orders=max_orders)

    def _construct_orders(self, predictions: np.array, order_datetimes: pd.DatetimeIndex,
                         ticker: string,  take_profit: int, stop_loss: int, time_limit: int,
                         data: pd.DataFrame 
                        ) -> pd.DataFrame:
        directions = np.empty(len(predictions), dtype='<U4')
        directions[predictions == 0] = "BUY"
        directions[predictions == 1] = "SELL"
        order_datetimes = order_datetimes[predictions != 2] # HOLD
        directions = directions[predictions != 2]
        opening_value = data.loc[order_datetimes, 'Close'].to_numpy()
        orders = pd.DataFrame({
            'index': order_datetimes,
            'Direction': directions,
            'Opening_Value': opening_value.squeeze(),
        })
        orders['Ticker'] = ticker
        orders['Take_Profit'] = take_profit * OutcomeSimulator.multipliers[ticker]
        orders['Stop_Loss'] = stop_loss * OutcomeSimulator.multipliers[ticker]
        orders['Time_Limit'] = time_limit

        orders = orders.set_index('index')
        self.order = orders
        return orders

    def _calculate_results(self, data, cooldown, max_orders):
        simulator = OutcomeSimulator() 
        self.signal_outcomes = simulator.determine_outcomes(data, self.orders)
        self.position_outcomes = simulator.find_actioned_orders(
            self.signal_outcomes, cooldown=cooldown, max_orders=max_orders,
            )
        self.positions_results = simulator.calc_results(self.position_outcomes)

    def plot_position_profit(self):
        self._plot_profits('positions')

    def plot_all_signals_profits(self):
        self._plot_profits('signals')

    def _get_name_data_pair(self, orders_type):
        if orders_type.lower() == 'positions':
            title = "Positions"
            data = self.position_outcomes['Profit']
        elif orders_type.lower() == 'signals':
            title = "Signals"
            data = self.signal_outcomes['Profit']
        else:
            raise ValueError(f"Unknown order type: {orders_type}")
        return title, data
    
    def _plot_profits(self, orders_type):
        title, data = self._get_name_data_pair(orders_type)
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90)
        plt.plot(data.cumsum())
        plt.show()    

    def plot_position_distribution(self,):
        self._plot_distribution('Positions', 'distribution')

    def plot_position_profit_distribution(self):
        self._plot_distribution('Positions', 'total p/l distribution')

    def plot_position_profit_distribution_mean(self):
        self._plot_distribution('Positions', 'mean p/l distribution')

    def _plot_distribution(self, orders_type, dist_type):
        title, data = self._get_name_data_pair(orders_type)
        t = [c.hour + c.minute/60 for c in data.index.time]
        edges = self._calculate_edges()
        fig, ax = plt.subplots()
        ax.set_title(f'{orders_type} {dist_type}')
        ax.tick_params(axis='x', rotation=90)
        if dist_type.lower() == "distribution":
            _ = plt.hist(t, bins=edges)
        elif "p/l" in  dist_type.lower():
            if "total" in dist_type.lower():
                f = np.sum
            elif "mean" in dist_type.lower():
                f = np.mean
            else:
                raise ValueError(f"Unkown dist_type: {dist_type}")
            indx = np.digitize(t, bins=edges)
            p = np.zeros(len(edges))
            for i in np.unique(indx):
                p[i-1] = f(data.iloc[indx == i])
            _ = plt.bar(edges+0.25, p, width=0.5,
                        color=[['red', 'blue'][i] for i in (p > 0)])
        else:
            raise ValueError(f"Unkown dist_type: {dist_type}")
        plt.show()

    def _calculate_edges(self):
        tz = get_ticker_time_zone(self.ticker)
        opening_time, closing_time = opening_and_closing_times(self.ticker)
        opening_time = opening_time.tz_convert(tz)
        closing_time = closing_time.tz_convert(tz)
        ot = opening_time.time()
        ct = closing_time.time()
        edges = np.arange(ot.hour + ot.minute/60,
                          ct.hour + ct.minute/60,
                          0.5)
        return edges
        
    def plot_daily_profit(self,):
        d = self.position_outcomes['Profit'].cumsum().resample("1B").ohlc()
        mpl.plot(d, type='candle', style='yahoo', title='Daily P/L')
        plt.show()

    def plot_weekly_profit(self,):
        d = self.position_outcomes['Profit'].cumsum().resample("1W").ohlc()
        mpl.plot(d, type='candle', style='yahoo', title="Weekly P/L")
        plt.show()

    def _plot_data(self, data):
        o = data['Open'].resample('1B').first()
        h = data['High'].resample('1B').max()
        l = data['Low'].resample('1B').min()
        c = data['Close'].resample('1B').last()
        x = np.concatenate((o.to_numpy(), h.to_numpy(), l.to_numpy(), c.to_numpy()), axis=1)
        data_daily = pd.DataFrame(x, columns=['Open', 'High', 'Low', 'Close'], index=o.index)
        mpl.plot(data_daily, type='candle', style='yahoo', title=f'{self.ticker} Daily')

    def generate_report(self, path_to_files):
        # TODO
        # Create directory
        # Save figures
        # Write preample
        # Get outcome table(s)
        # write tables
        # write floats
        # Call latex compilation using sys
        # '/usr/local/texlive/2019/bin/x86_64-linux/pdflatex -synctex=1 -interaction=nonstopmode report.tex'
        pass

# Steering results methods using model to close orders additionally to standard exit conditions
class SteeringResults(Results):

    def _construct_orders(self, predictions, order_datetimes, ticker,
                          take_profit, stop_loss, time_limit,
                          data):
        directions = np.empty(len(predictions), dtype='<U4')
        directions[predictions == 0] = "BUY"
        directions[predictions == 1] = "SELL"
        directions[predictions == 2] = "HOLD"
        is_change = directions[:-1] != directions[1:]
        is_change = np.array([directions[0] != 'HOLD'] + is_change.tolist())
        
        old_d = 'HOLD'
        order = {}
        orders = {}
        for i, d, dt in zip(is_change, directions, order_datetimes):
            if i:
                new_d = d
                if new_d in ['BUY','SELL']:
                    if old_d != 'HOLD':
                        dt_limit = max(
                            order_datetimes[
                                order_datetimes.date == order['Opening_Datetime'].date()
                                ]
                            )
                        closing_datetime = min(dt, dt_limit)
                        if closing_datetime == order['Opening_Datetime']:
                            closing_datetime += pd.Timedelta(minutes=1)
                        order['Closing_Datetime'] = closing_datetime
                        orders[order['Opening_Datetime']] = order
                        order = {}
                        
                    order['Opening_Datetime'] = dt
                    order['Direction'] = d
                else:
                    if old_d != 'HOLD':
                        dt_limit = max(
                            order_datetimes[
                                order_datetimes.date == order['Opening_Datetime'].date()
                                ]
                            )
                        closing_datetime = min(dt, dt_limit)
                        if closing_datetime == order['Opening_Datetime']:
                            closing_datetime += pd.Timedelta(minutes=1)
                        order['Closing_Datetime'] = closing_datetime
                        orders[order['Opening_Datetime']] = order
                        order = {}
                old_d = new_d
            else:
                pass
            
        orders = pd.DataFrame(orders).T
        max_time = max(data.index.time)
        is_bad = pd.DatetimeIndex(orders['Closing_Datetime']).time > max_time
        orders = orders.iloc[np.logical_not(is_bad)]
        orders['Opening_Value'] = data.loc[orders.index,'Close'].to_numpy()
        orders['Closing_Value'] = data.loc[orders['Closing_Datetime'],'Close'].to_numpy()
        ds = orders['Direction'].to_numpy().copy()
        ds[ds=='BUY'] = 1
        ds[ds=='SELL'] = -1
        ds = ds.astype(float)
        orders['Profit'] = ds * (orders['Closing_Value'] - orders['Opening_Value'])
        orders['Ticker'] = ticker
        orders['Take_Profit'] = take_profit * OutcomeSimulator.multipliers[ticker]
        orders['Stop_Loss'] = stop_loss * OutcomeSimulator.multipliers[ticker]
        
        t = orders['Closing_Datetime'] - orders['Opening_Datetime']
        t = np.array([int(np.round(ti.seconds / 60)) for ti in t.to_list()])
        orders['Time_Limit'] = t[:,np.newaxis]
        self.orders = orders.copy()
        
        return orders
    



