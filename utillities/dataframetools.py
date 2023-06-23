#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:57:00 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd

def convert_to_ohlc(data, freq):
    data = data.resample(freq).ohlc().ffill()
    data.columns = [c.capitalize() for c in data.columns.droplevel(0)]
    c = data['Close']
    c.index = c.index + pd.tseries.frequencies.to_offset(freq)
    index = c.index.intersection(data.index)
    data.loc[index,'Open'] = c.loc[index]
    return data


if __name__ == "__main__":
    import mplfinance as mpl
    initial_value = 1e4
    data = initial_value * np.random.normal(loc=1, scale=0.001, size=1000)
    index = pd.date_range(
        pd.Timestamp("09:00"), periods=len(data), freq='14S'
        )
    df = pd.DataFrame(data, index=index)
    
    df_ohlc = convert_to_ohlc(df, 'T')
    
    mpl.plot(df_ohlc.iloc[:30], type='candle', style='yahoo',)