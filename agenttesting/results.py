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

class OutcomeSimulator(object):
    
    spreads = {"^NDX" : 1,"^DJI" : 2.4, "^GSPC" : 0.4, "^FTSE" : 1,
               "^GDAXI" : 1.2, "^FCHI" : 1, "^AEX" : 0.3, "^N225" : 7, "^TWII" : 7,
               "GBPEUR=X" : 3e-4, "GBP=X" : 1.5e-4, "EUR=X" : 0.9e-4, "CHF=X" : 1.5e-4,
               "CHFEUR=X" : 3e-4, "CHFGBP=X" : 5e-4}
    multipliers = {"^NDX" : 1,"^DJI" : 1, "^GSPC" : 1, "^FTSE" : 1,
               "^GDAXI" : 1, "^FCHI" : 1, "^AEX" : 1, "^N225" : 1, "^TWII" : 1,
               "GBPEUR=X" : 1e4, "GBP=X" : 1e4, "EUR=X" : 1e4, "CHF=X" : 1e4,
               "CHFEUR=X" : 1e4, "CHFGBP=X" : 1e4}
    outcome_columns = ['Opening_Value','Direction','Ticker','Take_Profit','Stop_Loss','Time_Limit',
                       'Closing_Datetime','Closing_Value','Time_Limit_Hit','Take_Profit_Hit',
                       'Stop_Loss_Hit','Profit']
        
    def calc_results(self, orders: pd.DataFrame) -> pd.DataFrame:
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

    def find_actioned_orders(self, orders: pd.DataFrame, cooldown=3, max_orders=1) -> pd.DataFrame:
        '''
        Creates a dataframe of all positions given a maximum of one open position at a time

        Parameters
        ----------
        orders : pd.DataFrame
            An dataframe containing the orders indexed by their opening times with columns
        cooldown : int
            The number of minutes to wait after an order hits its stop loss before opening
            another order
        max_orders : int
            The maximum number of simultaneously open positions

        Returns
        -------
        positions -> pd.DataFrame
            Similar to orders but only containing the actioned orders based on cooldown 
            and max_orders

        '''
        if len(orders) == 0:
            return pd.DataFrame(columns=orders.columns)
        
        t = np.array([orders.index[0] for i in range(max_orders)]) ### ???
        triggered_orders = []
        for index, order in orders.iterrows():
            is_greater = index > t
            if is_greater.any():
                triggered_orders.append(index)
                new_t = order['Closing_Datetime']
                if order['Stop_Loss_Hit']:
                    new_t += pd.Timedelta(minutes=cooldown)
                    #t = np.array([new_t for i in range(max_orders)])
                #else:
                t[np.argmax(is_greater)] = new_t
                
                    
        
        if len(triggered_orders) == 0:
            positions = pd.DataFrame(columns=orders.columns)
        else:
            positions = orders.loc[triggered_orders,:]
            
        return positions

    def determine_outcomes(self, df: pd.DataFrame, orders_o: pd.DataFrame) -> pd.DataFrame :
        '''
        Calculates the outcomes of the orders given the data in df

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe containing OHLC market data covereing the tickers and timeframes 
            of the orders.
        orders_o : pd.DataFrame
            A dataframe containing orders indexed by their opening times and with
            columns containing ticker, direction, opening value, take profit and 
            stop loss distances and time limits

        Returns
        -------
        pd.DataFrame
            A dataframe containing the orders with additional columns indicating the 
            outcome (Take_Profit_Hit/Stop_Loss_Hit/Time_Limit_Hit), the closing
            datetime and value, and the profit

        '''
        
        if len(orders_o) == 0:
            return pd.DataFrame(columns=OutcomeSimulator.outcome_columns)
        
        orders = orders_o.copy(deep=True)
    
        orders = self.limit_times(orders, df)
        
        is_take_profit, is_stop_loss, is_timeout = self.outcome_bool(orders)
        
        to_indx = orders.index[is_timeout]
        tp_indx = orders.index[is_take_profit]
        sl_indx = orders.index[is_stop_loss]
        
        orders['Time_Limit_Hit'] = is_timeout
        orders['Take_Profit_Hit'] = is_take_profit
        orders['Stop_Loss_Hit'] = is_stop_loss
        
        orders.loc[to_indx,'Closing_Datetime'] = orders.loc[to_indx,'Timeout_Time']
        orders.loc[tp_indx,'Closing_Datetime'] = orders.loc[tp_indx,'Take_Profit_Time']
        orders.loc[sl_indx,'Closing_Datetime'] = orders.loc[sl_indx,'Stop_Loss_Time']
        
        closing_time = orders.loc[to_indx,'Timeout_Time']
        #print(df.index[-1])
        ct = []
        for i in closing_time:
            i_max = df.index[df.index.date == i.date()][-1]
            if i > i_max:
                i = i_max
            ct.append(i)
        
        closing_time = pd.DatetimeIndex(ct)
        to_closing = df.loc[closing_time,'Close'].to_numpy()
        
        buysell = np.zeros(len(orders))
        buysell[orders['Direction'] == 'BUY'] = 1.0
        buysell[orders['Direction'] == 'SELL'] = -1.0
        tp_closing = orders.loc[tp_indx,'Opening_Value'] + buysell[is_take_profit] * orders.loc[tp_indx,'Take_Profit']
        sl_closing = orders.loc[sl_indx,'Opening_Value'] - buysell[is_stop_loss] * orders.loc[sl_indx,'Stop_Loss']
        
       
        orders.loc[to_indx,'Closing_Value'] = to_closing
        orders.loc[tp_indx,'Closing_Value'] = tp_closing
        orders.loc[sl_indx,'Closing_Value'] = sl_closing

        s = [OutcomeSimulator.spreads[ticker] for ticker in orders['Ticker']]
        m = [OutcomeSimulator.multipliers[ticker] for ticker in orders['Ticker']]
        orders['Profit'] = m * (buysell * (orders['Closing_Value'] - orders['Opening_Value']) \
                            - s)

        return orders.loc[:,OutcomeSimulator.outcome_columns]
    
    def outcome_bool(self, orders):
        # Time limit is hit 
        is_timeout = np.logical_and(
            np.logical_or(orders['Take_Profit_Time'].isna(), 
                          orders['Take_Profit_Time'] > orders['Timeout_Time']),
            np.logical_or(orders['Stop_Loss_Time'].isna(), 
                          orders['Stop_Loss_Time'] > orders['Timeout_Time']))
        # Take profit is hit before stop loss (sl > tp)
        is_take_profit = np.logical_and(
            np.logical_or(orders['Stop_Loss_Time'].isna(),
                          orders['Stop_Loss_Time'] > orders['Take_Profit_Time']),
            np.logical_not(is_timeout))
        
        is_stop_loss = np.logical_and(
            np.logical_not(is_take_profit), np.logical_not(is_timeout))
        
        
        return is_take_profit, is_stop_loss, is_timeout
        
    def outcome_indx(self,orders):
        is_take_profit, is_stop_loss, is_timeout = self.outcome_bool(orders)
        
        to_indx = orders.index[is_timeout]
        tp_indx = orders.index[is_take_profit]
        sl_indx = orders.index[is_stop_loss]
        return to_indx, tp_indx, sl_indx
        
    def limit_times(self, orders, df):
        #orders = orders_o.copy(deep=True)
        time_limits = orders.loc[:,'Time_Limit'].to_numpy()
        tickers = orders.loc[:,'Ticker'].to_numpy()
        order_types = orders.loc[:,'Direction'].to_numpy()
        for ticker in np.unique(tickers):
            is_ticker = tickers == ticker
            for time_limit in np.unique(time_limits):
                is_time_limit = np.logical_and(is_ticker, time_limits == time_limit)
                for order_type in np.unique(order_types):
                    is_order_type = np.logical_and(is_time_limit, order_types == order_type)
                    if any(is_order_type):
                        if order_type == 'BUY':
                            tp_comparator_func = self.gt_func
                            tp_comparator_name = 'High'
                            sl_comparator_func = self.lt_func
                            sl_comparator_name = 'Low'
                        elif order_type =='SELL':
                            tp_comparator_func = self.lt_func
                            tp_comparator_name = 'Low'
                            sl_comparator_func = self.gt_func
                            sl_comparator_name = 'High'
                            
                        take_profit_times = self.price_delta_time(
                            orders=orders.loc[is_order_type,:], 
                            df=df, 
                            comparator_func=tp_comparator_func, 
                            comparator_name=tp_comparator_name,
                            delta_name='Take_Profit', 
                            time_limit=time_limit
                            )
                        stop_loss_times = self.price_delta_time(
                            orders.loc[is_order_type,:], 
                            df=df, 
                            comparator_func=sl_comparator_func, 
                            comparator_name=sl_comparator_name, 
                            delta_name='Stop_Loss', 
                            time_limit=time_limit
                            )
                        
                        orders.loc[orders.index[is_order_type],'Take_Profit_Time'] = take_profit_times
                        orders.loc[orders.index[is_order_type],'Stop_Loss_Time'] = stop_loss_times
                    
                orders.loc[orders.index[is_time_limit],'Timeout_Time'] = \
                    orders.index[is_time_limit] + pd.Timedelta(minutes=time_limit)
                
        return orders

    def gt_func(self, x, c, price_delta):
        return c >= (x + price_delta)
    
    def lt_func(self, x,c, price_delta):
        return c <= (x - price_delta)

    def price_delta_time(self, orders, df, comparator_func, comparator_name, delta_name, time_limit):
        order_indx = df.index.get_indexer(orders.index) # Int index of order start time in df
        x = orders['Opening_Value'].to_numpy().squeeze() # Opening value of order
        price_delta = orders[delta_name].to_numpy().squeeze() # Take_Profit/Stop_Loss
        r = np.arange(time_limit)[:,np.newaxis] # columnwise range 0 -> timelimit
        indx = order_indx + 1 + r # +1 as order starts at index close value 
        comparator = df[comparator_name].to_numpy().squeeze() # Comparison data
        indx[indx>=len(comparator)] = len(comparator) - 1 
        comparator = comparator[indx] # Comparison matrix (n_timesteps,max_orders)
        is_ineq = comparator_func(x, comparator, price_delta) # boolean array
        rel_indx = np.argmax(is_ineq, axis=0) # int index relative to order start
        none_found = np.logical_not(np.any(is_ineq, axis=0))
        # NB if rel_indx == 0 then the target is either
        #   a) not hit within the time limit (indx[0] is the start 
        #      time which starts at the 'close' value)
        #   b) hit on the first step
        abs_indx = rel_indx + order_indx + 1 # int index in df ### TODO +1 as the comparator does not include the order time!
        #abs_indx[rel_indx==0] = len(df.index) - 1
        time_indx = df.index[abs_indx].to_numpy() # the datetime when the price has moved by price_delta
        time_indx[none_found] = pd.NaT
        return pd.DatetimeIndex(time_indx, tz = orders.index.tz)


