#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:33:52 2022

@author: mtolladay
"""
import numpy as np
import pandas as pd

def get_ticker_time_zone(ticker):
    tzs = {
        '^NDX': "US/Eastern",
        '^DJI': "US/Eastern",
        '^GSPC': "US/Eastern",
        '^GDAXI': "Europe/Berlin",
        '^AEX': "Europe/Berlin",
        '^FCHI': "Europe/Berlin",
        '^FTSE': "Europe/London",
        '^N225': "Asia/Tokyo",
        '^TWII': "Asia/Taipei",
        }
    if ticker in tzs.keys():
        tz = tzs[ticker]
    elif ticker in ['GBPEUR=X','GBP=X','EUR=X','CHF=X','CHFEUR=X','CHFGBP=X']:
        tz = "UTC"
    else:
        raise ValueError(f"No time zone available for ticker {ticker}")
    return tz

def opening_and_closing_times(ticker, date=None):
    if ticker in ['^NDX', '^DJI', '^GSPC']:
        opening_time = pd.Timestamp("09:30:00", tz="US/Eastern")
        closing_time = pd.Timestamp("16:00:00", tz="US/Eastern")
    elif ticker in ['^GDAXI','^FCHI','^AEX']:
        opening_time = pd.Timestamp("09:00:00", tz="Europe/Berlin")
        closing_time = pd.Timestamp("17:30:00", tz="Europe/Berlin")
    elif ticker == '^FTSE':
        opening_time = pd.Timestamp("08:00:00", tz="Europe/London")
        closing_time = pd.Timestamp("16:30:00", tz="Europe/London")
    elif ticker == '^N225':
        opening_time = pd.Timestamp("09:00:00", tz="Asia/Tokyo")
        closing_time = pd.Timestamp("15:00:00", tz="Asia/Tokyo")
    elif ticker == '^TWII':
        opening_time = pd.Timestamp("09:00:00", tz="Asia/Taipei")
        closing_time = pd.Timestamp("13:25:00", tz="Asia/Taipei")
    elif ticker in ['GBPEUR=X','GBP=X','EUR=X','CHF=X','CHFEUR=X','CHFGBP=X']:
        opening_time = pd.Timestamp("00:00:00", tz="UTC")
        closing_time = pd.Timestamp("23:59:00", tz="UTC")                 
    else:
        raise ValueError('No known opening times for ticker "' + {ticker} +'"')
    
    if date is not None:
        opening_time = pd.Timestamp.combine(date, opening_time.time()).tz_localize(opening_time.tz)
        closing_time = pd.Timestamp.combine(date, closing_time.time()).tz_localize(closing_time.tz)
        
    return opening_time.tz_convert("UTC"), closing_time.tz_convert("UTC")


def holidays(ticker):
    if ticker in ['^NDX', '^DJI', '^GSPC']:
        holidays = ["2022-01-17","2022-02-21","2022-04-15","2022-05-30","2022-06-20",
                    "2022-07-04","2022-09-05","2022-11-24","2022-11-25","2022-12-26",
                    "2023-01-02","2023-01-16","2023-02-20","2023-04-07","2023-05-29",
                    "2023-06-19","2023-07-03","2023-07-04","2023-09-04","2023-11-23",
                    "2023-11-24","2023-12-25"]
    elif ticker == '^GDAXI':
        holidays = ["2022-04-15", "2022-04-18","2022-06-06","2022-06-16","2022-10-03",
                    "2022-12-26",
                    "2023-04-07","2023-04-10","2023-05-01","2023-12-25","2023-12-26"]
    elif ticker == '^FTSE':
        holidays = ["2022-06-02","2022-06-03","2022-08-29","2022-09-19","2022-12-23",
                    "2022-12-26","2022-12-27","2022-12-30",
                    "2023-01-02","2023-04-07","2023-04-10","2023-05-01","2023-05-08",
                    "2023-05-29","2023-07-28","2023-12-22","2023-12-25","2023-12-26",
                    "2023-12-29"]
    elif ticker in ['^FCHI','^AEX']:
        holidays = ["2022-12-26",
                    "2023-04-07","2023-04-10","2023-05-01","2023-12-25","2023-12-26"]
    elif ticker == '^N225':
        holidays = ["2022-08-11","2022-09-19","2022-09-23","2022-10-10","2022-11-03",
                   "2022-11-23","2022-12-31",
                   "2023-01-01","2023-01-02","2023-01-03","2023-01-09","2023-02-11",
                   "2023-02-23","2023-03-21","2023-04-29","2023-05-03","2023-05-04",
                   "2023-05-05","2023-07-17","2023-08-11","2023-09-18","2023-09-23",
                   "2023-10-09","2023-11-03","2023-11-23","2023-12-31"]
    elif ticker == '^TWII':
        holidays = ["2022-09-09","2022-10-10",
                    "2023-01-02","2023-01-18","2023-01-19","2023-01-20","2023-01-23",
                    "2023-01-24","2023-01-25","2023-01-26","2023-01-27","2023-02-27",
                    "2023-02-28","2023-04-03","2023-04-04","2023-04-05","2023-05-01",
                    "2023-06-22","2023-06-23","2023-09-29","2023-10-09","2023-10-10"]
    else:
        holidays = []
        
    return pd.DatetimeIndex(holidays, tz='UTC')

def convert_offset_to_seconds(offset):
    if type(offset) == str:
        offset = pd.tseries.frequencies.to_offset(offset)
    t = pd.Timestamp.now()
    try:
        s = (t + offset - t).total_seconds()
    except TypeError:
        raise TypeError(f'Offset was of type {type(offset)}. Must be either offset string or offset')
    else:
        return s

class MarketDateTimeIndexGenerator(object):
    
    def __init__(self, ticker):
        self.ticker = ticker
        
        self.tz = get_ticker_time_zone(ticker)
        opening_time, closing_time = opening_and_closing_times(ticker)
        self.opening_time = opening_time.tz_convert(self.tz)
        self.closing_time = closing_time.tz_convert(self.tz)
        self.holidays = holidays(ticker)
        
    def make_index(self, start_datetime, end_datetime, freq):
        index = pd.bdate_range(start_datetime, end_datetime, freq=freq, tz=self.tz)
        index = self.remove_holidays_from_index(index)
        index = index[
            index.indexer_between_time(
                self.opening_time.time(), 
                self.closing_time.time()
                )
            ]
        return index
    
    def remove_holidays_from_index(self, datetimeindex):
        is_holiday = np.isin(datetimeindex.date, self.holidays.date)
        new_index = datetimeindex[np.logical_not(is_holiday)]
        return pd.DatetimeIndex(new_index)
        
if __name__ == "__main__":
    mdti = MarketDateTimeIndexGenerator('^NDX')
    index = mdti.make_index("2022-01-01","2023-01-01",freq='T')
        
