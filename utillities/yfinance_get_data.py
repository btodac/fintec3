#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:10:13 2022

@author: mtolladay
"""
import shelve
import pandas as pd

import yfinance as yf

n_validation_weeks = 4
n_validation_weeks2 = 8

filename_all = '/home/mtolladay/Documents/finance/week_1m_all.dbm'
filename_training = '/home/mtolladay/Documents/finance/week_1m_training.dbm'
filename_training2 = '/home/mtolladay/Documents/finance/week_1m_training2.dbm'
filename_validation = '/home/mtolladay/Documents/finance/week_1m_validation.dbm'
filename_validation2 = '/home/mtolladay/Documents/finance/week_1m_validation2.dbm'
filename_metatrain = '/home/mtolladay/Documents/finance/week_1m_meta_train.dbm'
filename_metatest = '/home/mtolladay/Documents/finance/week_1m_meta_test.dbm'


# Check these dates!!!!!!!!! 1m data is only available from the last 30 days
with shelve.open(filename_all) as data:
    current_keys = list(data.keys())

n_previous_weeks = 4
alldates = pd.date_range(pd.Timestamp.today() - pd.Timedelta(days=n_previous_weeks*7+1),
                         pd.Timestamp.today())
all_available_sundays = alldates[alldates.day_of_week == 6]
all_available_sundays = [str(s.date()) for s in all_available_sundays]
all_available_sundays = all_available_sundays[:-1]

# The full list of tickers to collect
# Nasdaq, dow jones, sp500, ftse100, dax40, CAC40, AEX25, Nikkei, Taiwan, Volatility
indices = "^NDX ^DJI ^GSPC ^FTSE ^GDAXI ^FCHI ^AEX ^N225 ^TWII ^VIX"
# US Oil, US GOLD, London Oil, london Gold
commodities = "CL=F GC=F BZ=F BRNT.L SGLN.L"
currencies = "GBPEUR=X GBP=X EUR=X CHF=X CHFEUR=X CHFGBP=X"
# Us treasury bonds: TIPS, 5 year, 10 year, 30 year
bonds = "TIP ^FVX ^TNX ^TYX"

is_first = True
# Loop over weeks
for start in [s for s in all_available_sundays if s not in current_keys]:
    start = pd.Timestamp(start)
    end = start + pd.Timedelta(days=6)
    print(f'{start.date()} {end.date()}')
    df = yf.download( tickers = " ".join([indices, commodities, currencies, bonds]),       
                      start=start.date(), end=end.date(),
                      interval = "1m",
                      group_by = 'column',
                      auto_adjust = False,
                      prepost = True,
                      threads = True,
                      proxy = None
                      )
    # Store in shelve using the start date (as str) as key
    with shelve.open(filename_all, writeback=True) as data:
        if is_first:
            # We must accept some Nans
            df = df.fillna(method='ffill')
            last_data = df.iloc[-1].copy()
            is_first = False         
        else:
            # Use the final data from the last week
            df.loc[last_data.name] = last_data
            df = df.sort_index()
            df = df.fillna(method='ffill')
            df = df.iloc[1:]
            last_data = df.iloc[-1].copy()
        data[str(start.date())] = df

# Split the data into multiple files
with shelve.open(filename_all) as data:
    all_keys = list(data.keys())
    all_keys.sort()
    
    with shelve.open(filename_training) as training_data:
        training_data.clear()
        for key in all_keys[:-n_validation_weeks]:
            if key not in list(training_data.keys()):
                training_data[key] = data[key]
            
    with shelve.open(filename_validation) as validation_data:
        validation_data.clear()
        for key in all_keys[-n_validation_weeks:]:
            if key not in list(validation_data.keys()):
                validation_data[key] = data[key]
                
    with shelve.open(filename_training2) as training_data:
        training_data.clear()
        for key in all_keys[:-n_validation_weeks2]:
            if key not in list(training_data.keys()):
                training_data[key] = data[key]
            
    with shelve.open(filename_validation2) as validation_data:
        validation_data.clear()
        for key in all_keys[-n_validation_weeks2:]:
            if key not in list(validation_data.keys()):
                validation_data[key] = data[key]
                
    with shelve.open(filename_metatrain) as metatrain_data:
        metatrain_data.clear()
        for key in all_keys[-4:-2]:
            metatrain_data[key] = data[key]
    
    with shelve.open(filename_metatest) as metatest_data:
        metatest_data.clear()
        for key in all_keys[-2:]:
            metatest_data[key] = data[key]
    
        
