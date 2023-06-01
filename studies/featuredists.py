#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:46:21 2023

@author: mtolladay
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

from agents.preprocessing import ObservationBuilder
from utillities.datastore import Market_Data_File_Handler

ticker = "^NDX"
features = ['High','Kurt','Low','Mean','MeanDist','MeanDiff','Mom',
            'Range','Skew','Std','StochOsc','Trend']
columns = []
for f in features:
    if f == 'MeanDiff':
        columns.append('4min_8min_' + f)
        columns.append('16min_64min_' + f)
    elif f == 'Range':
        columns.append(f)
    else:
        columns.append('4min_' + f)
        columns.append('16min_' + f)
        columns.append('64min_' + f)


data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)

observer = ObservationBuilder(columns, all_data.index[0], all_data.index[-1], all_data.index.tz)

observations, order_datetime = observer.make_observations(all_data)

df = pd.DataFrame(observations, columns=columns, index=order_datetime)

for c in [c for c in columns if c not in  ['4min_skew','4min_Kurt']]:
    
    o = df[c].to_numpy().copy()
    o = o[o!=0]
    f = c.split('_')[-1].lower()
    if f in ['kurt','mean','meandist','meandiff',
                     'mom','skew','trend',]:
        p = stats.nct.fit(o)
        d = stats.nct(*p)
    elif f == 'high':
        p = stats.truncexpon.fit(o, f0=o.max())
        d = stats.truncexpon(*p)
    elif f == 'low':
        o = -o
        p = stats.truncexpon.fit(o, f0=o.max())
        d = stats.truncexpon(*p)
    elif f in ['range','std']:
        p = stats.invweibull.fit(o)
        d = stats.invweibull(*p)
    elif f == 'stochosc':
        p = stats.laplace_asymmetric.fit(o)
        d = stats.laplace_asymmetric(*p)

    fig, ax = plt.subplots(1, 1)
    ax.set_title(c)
    ax.hist(o, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
    m = o.mean()
    s = o.std()
    x = np.linspace(m-6*s, m+6*s, 1000)
    ax.plot(x, d.pdf(x), 'k-', lw=2, label='frozen pdf')
    ax.set_xlim(m-2*s, m+2*s)
    plt.show()
