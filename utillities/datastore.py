#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 13:12:28 2022

@author: mtolladay
"""

import shelve
import pandas as pd

from utillities.timesanddates import opening_and_closing_times, holidays

class Market_Data_File_Handler(object):
    '''
    Provides access functions for the weekly tabulated market data held in a dbm file
    '''
    #_basename = '/home/mtolladay/Documents/finance/'
    #_min_intervals = [2,3,5,10,15,20,30,60,120,240]
    
    def __init__(self, dataset_name=None):
        if dataset_name is None:
            self.filename = '/home/mtolladay/Documents/finance/week_1m_training.dbm'
        elif dataset_name.lower() == 'training':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_training.dbm'
        elif dataset_name.lower() == 'validation':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_validation.dbm'
        elif dataset_name.lower() == 'training2':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_training2.dbm'
        elif dataset_name.lower() == 'validation2':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_validation2.dbm'
        elif dataset_name.lower() == 'meta_training':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_meta_train.dbm'
        elif dataset_name.lower() == 'meta_testing':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_meta_test.dbm'
        elif dataset_name.lower() == 'all':
            self.filename = '/home/mtolladay/Documents/finance/week_1m_all.dbm'
        else:
            raise ValueError(f'Unrecognised dataset_name "{dataset_name}". Use one of "training"/"validation"/"all"')
        self.keys = self._get_available_keys()
        self.keys.sort()
        self._starting_date = self.keys[0]
        self.tickers = self._get_available_tickers()
        self._get_available_dates()
    
    def get_available_dates(self, ticker: str=None) -> pd.DatetimeIndex:
        '''
        Returns the dates for which data exists in the dbm file

        Parameters
        ----------
        ticker : str
            Will only find dates containing ticker (default=None, returns all dates)
            
        Returns
        -------
        dates : pd.DateTimeindex

        '''
        if ticker is None:
            return self._all_dates
        else:
            if ticker in self._ticker_dates.keys():
                return self._ticker_dates[ticker]
            else:
                raise ValueError(f'Ticker {ticker} was not found in the data file {self.filename}')
            
    def _get_available_dates(self):
        all_dates = []
        available_dates = {}
        with shelve.open(self.filename) as file:
            tz = file[list(file.keys())[0]].index.tz
            for key in self.keys:
                df = file[key]
                tickers = df.columns.levels[1]
                for ticker in tickers:
                    dates = pd.bdate_range(df.index.min().date(), df.index.max().date(), freq='b', 
                                           tz=tz)
                    dates = dates.difference(holidays(ticker))
                    if ticker in available_dates.keys():
                        available_dates[ticker] = available_dates[ticker].union(dates)
                    else:
                        available_dates[ticker] = dates
                
                all_dates.append(pd.bdate_range(file[key].index.min().date(),
                                                file[key].index.max().date(),
                                                freq='b', tz=tz))
            d = all_dates[0]
            for date in all_dates[1:]:
                d = d.union(date)
            
        self._all_dates = d
        self._ticker_dates = available_dates
            
    def _get_available_keys(self):
        '''
        Returns the keys for the dbm file
        
        Returns
        -------
        keys : list(str)

        '''
        with shelve.open(self.filename) as file:
            keys = list(file.keys())
            keys.sort()
            return keys
    
    def _get_available_tickers(self):
        '''
        Returns the tickers contained in the dbm file records

        Returns
        -------
        list(str)

        '''
        with shelve.open(self.filename) as file:
            tickers = [set(file[key].columns.levels[1]) for key in self.keys]
            return list(set().union(*tickers))


    
    def get_ticker_data(self, ticker, start_date=None, end_date=None, use_opening_hours=True,
                        start_time=None, end_time=None, as_list=True):
        '''
        Gets data correspnding to a ticker or tickers from the dbm file

        Parameters
        ----------
        ticker : str or list or tuple of str
            The ticker(s) symbol requested
        start_date : pd.Timestamp or str, optional
            Start of date range. The default is None, data from all available dates will be returned.
        end_date :  pd.Timestamp or str, optional
            End of date range. The default is None, data from all available dates will be returned.
        start_time : pd.Timestamp or str, optional
            The start time of the first datapoint. The default is "00:00:00".
        end_time : pd.Timestamp or str, optional
            The time of the last datapoint. The default is "23:59:00".
        as_list : bool, optional
            When True the data for each ticker is returned as a list of dataframes, one for each 
            day, otherwise it is returned as a single dataframe. The default is True.

        Returns
        -------
        data : list[pd.DataFrame] or pd.DataFrame or dict(list[pd.DataFrame]) or dict(pd.DataFrame)
            If a single ticker is requested then the retrun value is a list of dataframes, one for 
            each day, or a single dataframe containing all data for the ticker. If multiple tickers
            are requested then a dict is returned with tickers as keys and the values like those
            from the single ticker.

        '''
        if use_opening_hours and (start_time is not None and end_time is not None): # CONFLICT
            raise ValueError('Can not use "use_opening_hours=True" and specify start and end times')
            
        if (not use_opening_hours) and start_time is None and end_time is None: # Use defaults
            start_time = "00:00:00"
            end_time = "23:59:00"
  
        if type(ticker) is str:
            data = self._get_ticker_data(ticker, start_date=start_date, end_date=end_date,
                                         use_opening_hours=use_opening_hours,
                                         start_time=start_time, end_time=end_time, as_list=as_list)
        elif type(ticker) in [list, tuple]:
            data = self._get_tickers_data(ticker, start_date=start_date, end_date=end_date,
                                          use_opening_hours=use_opening_hours,
                                          start_time=start_time, end_time=end_time, as_list=as_list)
            
        return data
    
    def _get_tickers_data(self, tickers, **kwargs):
        '''
        Helper function for dealing with collections of tickers
        '''
        data = {} 
        for ticker in tickers:
            df = self._get_ticker_data(ticker=ticker, **kwargs)
            data[ticker] = df
            '''
            if type(df) is pd.DataFrame:
                if not df.empty:
                    data[ticker] = df
            else:
                data[ticker] = df
            '''                                    
        return data    
        
    def _get_ticker_data(self, ticker, start_date=None, end_date=None, use_opening_hours=True,
                         start_time="00:00:00", end_time="23:59:00", as_list=True) -> list[pd.DataFrame]:
        available_dates = self.get_available_dates(ticker)
        if start_date is None:
            start_date = available_dates[0]
        if end_date is None:
            end_date = available_dates[-1]

        date_range = pd.bdate_range(start_date, end_date) # Get the business days
        date_range = date_range.difference(holidays(ticker)) # Remove the holidays
        date_range = date_range.intersection(available_dates) # Only availble dates
        
        dfs = []
        with shelve.open(self.filename) as file:
            for date in date_range:
                df = file[self._closest_previous_key(date)].copy()
                df = df.loc[:,pd.IndexSlice[:,ticker]]
                if use_opening_hours:
                    start_datetime, end_datetime = opening_and_closing_times(ticker, date=date.date())
                else:
                    start_datetime = pd.Timestamp.combine(date.date(), start_time.time())
                    end_datetime = pd.Timestamp.combine(date.date(), end_time.time())
                df = self._regularise_dataframe(df, start_datetime, end_datetime)
                #df.columns = df.columns.droplevel(level=1)
                dfs.append(df)
            
            if as_list:
                return dfs
            else:
                return pd.concat(dfs)


       
    def means_and_stds(self, tickers, price='Close', time_length=45):
        #if tickers is None:
        #    tickers = self.tickers
            
        time_str = str(time_length) + 'min'
        ms = {}
        ss = {}
        if type(tickers) is str:
            tickers = [tickers]
            
        for ticker in tickers:
            ticker_data = self.get_ticker_data(ticker, as_list=False)
            ticker_data.columns = ticker_data.columns.droplevel(1)
            ss[ticker] = ticker_data.loc[:,price].rolling(time_str).std().mean()
            ms[ticker] = ticker_data.loc[:,price].mean()
            
        means = pd.Series(ms)
        stds = pd.Series(ss)
        return means, stds
                      
    def _regularise_dataframe(self, df, start_datetime, end_datetime) -> pd.DataFrame:
        '''
        Set df.index to fill the datetime range using freq='min' and attempts to remove all NaN 
        values using forward fill first followed by a back fill
        '''
        #print(df.index)
        new_index = pd.date_range(start_datetime, end_datetime, freq='min',tz=df.index.tz)
        #df = df.asfreq('min').fillna(method='ffill')
        df = df.reindex(new_index, method='ffill')
        df = df.fillna(method='bfill') #
        return df
   
    def _closest_previous_key(self, dt) -> str:
        '''
        Finds the key whose data will contain the datetime given in dt
        '''
        if type(dt) is str: # convert str to pd.Timestamp
            dt = pd.Timestamp(dt)
            
        if dt.weekday() == 6: # Sundays are keys
            key_dt = dt
        else: # Otherwise find the closest previous Sunday
            key_dt = dt - pd.Timedelta(days=dt.weekday()+1)
        return str(key_dt.date())

