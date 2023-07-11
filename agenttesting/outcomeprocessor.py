#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:09:48 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

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
        pt['Win Ratio'] = sum(orders['Profit'] > 0) / sum(orders['Profit'] < 0)
        pt['Profit'] = orders['Profit'].sum()
        pt['Profit (day)'] = orders['Profit'].resample('1B',closed='right',label='right').sum().mean()
        pt['Profit (week)'] = orders['Profit'].resample('1W',closed='right',label='right').sum().mean()
        pt['Avg. Profit'] = pt['Profit'] / pt['N_Trades']
        pt['Avg. Win'] = orders.Profit[orders.Profit.to_numpy() > 0].mean()
        #orders.iloc[orders.loc[:,'Take_Profit_Hit'].to_numpy()].loc[:,'Profit'].mean()
        pt['Avg. Lose'] = orders.Profit[orders.Profit.to_numpy() < 0].mean()
        #orders.iloc[orders.loc[:,'Stop_Loss_Hit'].to_numpy()].loc[:,'Profit'].mean()
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
            is_greater = index >= t
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

    def _determine_outcomes(self, df: pd.DataFrame, orders_o: pd.DataFrame) -> pd.DataFrame :
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
    
        # Get outcomes
        orders = self.limit_times(orders, df)
        
        is_take_profit, is_stop_loss, is_timeout = self.outcome_bool(orders)
        
        to_indx = orders.index[is_timeout]
        tp_indx = orders.index[is_take_profit]
        sl_indx = orders.index[is_stop_loss]
        
        orders['Time_Limit_Hit'] = is_timeout
        orders['Take_Profit_Hit'] = is_take_profit
        orders['Stop_Loss_Hit'] = is_stop_loss
        
        # Get closing times
        orders.loc[to_indx,'Closing_Datetime'] = orders.loc[to_indx,'Timeout_Time']
        orders.loc[tp_indx,'Closing_Datetime'] = orders.loc[tp_indx,'Take_Profit_Time']
        orders.loc[sl_indx,'Closing_Datetime'] = orders.loc[sl_indx,'Stop_Loss_Time']
        
        # Enforce closing at end of day
        closing_time = orders.loc[to_indx,'Timeout_Time']
        #print(df.index[-1])
        ct = []
        for i in closing_time:
            i_max = df.index[df.index.date == i.date()][-1]
            if i > i_max:
                i = i_max
                #print('Ooops')
            ct.append(i)
        
        #c = closing_time.to_numpy()
        #t_max = np.array([df.index[df.index.date == t][-1] for t in closing_time.date()])
        #f = np.min(np.stack((c, t_max)), axis=0)
        
        closing_time = pd.DatetimeIndex(ct)
        
        # Set closing values
        to_closing = df.loc[closing_time,'Close'].to_numpy()
        
        buysell = np.zeros(len(orders))
        buysell[orders['Direction'] == 'BUY'] = 1.0
        buysell[orders['Direction'] == 'SELL'] = -1.0
        tp_closing = orders.loc[tp_indx,'Opening_Value'] + buysell[is_take_profit] * orders.loc[tp_indx,'Take_Profit']
        sl_closing = orders.loc[sl_indx,'Opening_Value'] - buysell[is_stop_loss] * orders.loc[sl_indx,'Stop_Loss']
        
        orders.loc[to_indx,'Closing_Value'] = to_closing
        orders.loc[tp_indx,'Closing_Value'] = tp_closing
        orders.loc[sl_indx,'Closing_Value'] = sl_closing

        # Calculate profit/loss
        s = np.array([OutcomeSimulator.spreads[ticker] for ticker in orders['Ticker']])
        m = np.array([OutcomeSimulator.multipliers[ticker] for ticker in orders['Ticker']])
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
        # groupby???
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
        indx[indx>=len(comparator)] = len(comparator) - 1 # limit to the end of day
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
                   
    def determine_outcomes(self, data, orders, TSLI=20):
        orders['Trailing_Stop_Loss_Increment'] = TSLI
        time_limits = np.array([pd.Timedelta(minutes=tl) for tl in orders.Time_Limit])
        maxtime = orders.index.to_numpy() + time_limits
        end_of_day = [data.index[data.index.date == d].max() for d in np.unique(data.index.date)]
        end_of_day = dict(zip(pd.DatetimeIndex(end_of_day).date, end_of_day))
        maxtime = [min(end_of_day[t.date()], t + pd.Timedelta(minutes=order.Time_Limit)) \
                   for t, order in orders.iterrows()]
        orders['Max_Time'] = maxtime
        buy_orders = orders.iloc[orders.Direction.to_numpy() == 'BUY']
        sell_orders = orders.iloc[orders.Direction.to_numpy() == 'SELL']
        
        buy_orders['SL_Level'] = buy_orders.Opening_Value - buy_orders.Stop_Loss
        buy_orders['TP_Level'] = buy_orders.Opening_Value + buy_orders.Take_Profit
        #time_limits = np.array([pd.Timedelta(minutes=tl) for tl in buy_orders.Time_Limit])
        #buy_orders['Max_Time'] = buy_orders.index.to_numpy() + time_limits
        
        sell_orders['SL_Level'] = sell_orders.Opening_Value + sell_orders.Stop_Loss
        sell_orders['TP_Level'] = sell_orders.Opening_Value - sell_orders.Take_Profit
        #time_limits = np.array([pd.Timedelta(minutes=tl) for tl in sell_orders.Time_Limit])
        #sell_orders['Max_Time'] = sell_orders.index.to_numpy() + time_limits
        
        outcomes = pd.DataFrame(columns = [
            'Direction','Opening_Value','Ticker','Take_Profit','Stop_Loss','Time_Limit',
            'Trailing_Stop_Loss_Increment',
            'Stop_Loss_Hit', 'Take_Profit_Hit', 'Time_Limit_Hit', 'Profit','Closing_Datetime',
            ])
        open_buy_orders = pd.DataFrame(columns = buy_orders.columns)
        open_sell_orders = pd.DataFrame(columns = sell_orders.columns)
        
        for t, ohlc in data.iterrows():
            is_buy_sl = ohlc.Low[0] <= open_buy_orders.SL_Level.to_numpy()
            is_buy_tp = ohlc.High[0] > open_buy_orders.TP_Level.to_numpy()
            is_buy_tl = t >= open_buy_orders.Max_Time.to_numpy()
            
            is_sell_sl = ohlc.High[0] >= open_sell_orders.SL_Level.to_numpy()
            is_sell_tp = ohlc.Low[0] < open_sell_orders.TP_Level.to_numpy()
            is_sell_tl = t >= open_sell_orders.Max_Time.to_numpy()
            
            sl_buy = pd.DataFrame(columns=open_buy_orders.columns)
            sl_sell = pd.DataFrame(columns=open_buy_orders.columns)
            if is_buy_sl.any():
                sl_buy = open_buy_orders.iloc[is_buy_sl]
            if is_sell_sl.any():
                sl_sell = open_sell_orders.iloc[is_sell_sl]
            sl = pd.concat((sl_buy, sl_sell))
            
            tp_buy = pd.DataFrame(columns=open_buy_orders.columns)
            tp_sell = pd.DataFrame(columns=open_buy_orders.columns)
            if is_buy_tp.any():
                tp_buy = open_buy_orders.iloc[is_buy_tp]
            if is_sell_tp.any():
                tp_sell = open_sell_orders.iloc[is_sell_tp]
            tp = pd.concat((tp_buy, tp_sell))
            
            tl_buy = pd.DataFrame(columns=open_buy_orders.columns)
            tl_sell = pd.DataFrame(columns=open_buy_orders.columns)
            if is_buy_tl.any():
                tl_buy = open_buy_orders.iloc[is_buy_tl]
            if is_sell_tl.any():
                tl_sell = open_sell_orders.iloc[is_sell_tl]
            tl = pd.concat((tl_buy, tl_sell))
            
            if len(sl) > 0:
                sl['Stop_Loss_Hit'] = True
                sl['Take_Profit_Hit'] = False
                sl['Time_Limit_Hit'] = False
                directions = np.array([{'BUY': 1.0,'SELL' : -1.0}[d] for d in sl.Direction.to_numpy()])
                sl['Profit'] = directions * (sl.SL_Level - sl.Opening_Value)
                sl['Closing_Datetime'] = t
                
            if len(tp) > 0:
                tp['Stop_Loss_Hit']  = False
                tp['Take_Profit_Hit'] = True
                tp['Time_Limit_Hit'] = False
                tp['Profit'] = tp.Take_Profit
                tp['Closing_Datetime'] = t
                
            if len(tl) > 0:
                tl['Stop_Loss_Hit'] = False
                tl['Take_Profit_Hit'] = False
                tl['Time_Limit_Hit'] = True
                directions = np.array([{'BUY': 1.0,'SELL' : -1.0}[d] for d in tl.Direction.to_numpy()])
                tl['Profit'] = directions * (ohlc.Close[0] - tl.Opening_Value)
                tl['Closing_Datetime'] = t
            
            closed = pd.concat((sl, tp, tl))
            outcomes = pd.concat((outcomes, closed))
            # Remove closed orders
            if len(closed) > 0 :
                open_buy_orders = open_buy_orders.drop(closed.index[(closed.Direction=='BUY').to_numpy()])
                open_sell_orders = open_sell_orders.drop(closed.index[(closed.Direction=='SELL').to_numpy()])
            
            #Apply trailing sl
            is_buy_tsli = ohlc.High[0] > open_buy_orders.Opening_Value + open_buy_orders.Trailing_Stop_Loss_Increment
            if is_buy_tsli.any():
                open_buy_orders.SL_Level[is_buy_tsli] += open_buy_orders.Trailing_Stop_Loss_Increment[is_buy_tsli]
            is_sell_tsli = ohlc.High[0] > open_sell_orders.Opening_Value + open_sell_orders.Trailing_Stop_Loss_Increment
            if is_sell_tsli.any():
                open_sell_orders.SL_Level[is_sell_tsli] += open_sell_orders.Trailing_Stop_Loss_Increment[is_sell_tsli]
                
            # Check to see if an order is opened
            if t in buy_orders.index:
                orders_to_add = buy_orders.loc[t]
                open_buy_orders.loc[orders_to_add.name] = orders_to_add
            if t in sell_orders.index:
                orders_to_add = sell_orders.loc[t]
                open_sell_orders.loc[orders_to_add.name] = orders_to_add
        
        outcomes['Profit'] = outcomes['Profit'].to_numpy().astype(np.float32) 
        s = np.array([OutcomeSimulator.spreads[ticker] for ticker in outcomes['Ticker']])
        #m = np.array([OutcomeSimulator.multipliers[ticker] for ticker in orders['Ticker']])
        outcomes['Profit'] = outcomes['Profit'] - s
        outcomes = outcomes.sort_index()
        return outcomes
                