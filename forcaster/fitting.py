#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:25:48 2023

@author: mtolladay
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from agenttesting.datagenerator import GBMDataGen
from utillities.datastore import Market_Data_File_Handler
from utillities.timesanddates import get_ticker_time_zone

ticker = "^GDAXI"

# Load data
data_file = Market_Data_File_Handler(dataset_name="all")
all_data = data_file.get_ticker_data(ticker, as_list=False)
split_time = pd.Timestamp("2023-03-01", tz='UTC')
training_data = all_data.iloc[ all_data.index.to_numpy() < split_time ]
validation_data = all_data.iloc[all_data.index.to_numpy() >= split_time]

tz = get_ticker_time_zone(ticker) #'^GDAXI'
training_data = training_data.tz_convert(tz)
validation_data = validation_data.tz_convert(tz)
training_data = training_data.between_time(
    #"10:00", "16:00" # NDX
    "09:30", '17:30' # GDAXI
    )
# Convert to numpy array (close, range)
y = training_data[['Open','High','Low','Close']].to_numpy()


# Starting values
drift = 0
kappa = 1 # rate of return to mean volatility
theta = 0.0000001 # Leang term average voaltility
v0 = 0.000001 # initial volatility
chi = 0.00000001 # volatility of volatility
rho = 0 # coefficient of covariance
# fun = mean( ( Y - Y_pred) ^2 )
stoch_gen = GBMDataGen(
    training_data.index[0], training_data.index[50],
    initial_value=float(training_data['Open'].iloc[0].iloc[0]),
    drift=0, # points per second
    volatility=v0, 
    start_time=training_data.index[0],
    end_time=training_data.index[50],
    tz=training_data.index.tz,
    rho=rho,
    kappa=kappa,
    theta=theta,
    chi=chi,
    )
def objective(x,y):
    #drift, 
    kappa, theta, chi = x.tolist()
    #stoch_gen.drift = drift
    stoch_gen.kappa = kappa
    stoch_gen.theta = theta
    stoch_gen.chi = chi
    #stoch_gen.rho = rho
    
    batch_size = 16
    
    err = 0.0
    for i in range(batch_size):
        y_i = stoch_gen.generate()
        err += np.mean((y - y_i.to_numpy()[:-1,3]) ** 2)
        
    err /= batch_size

    return err    

x0 = [kappa, theta, chi]
bounds = ([0.0,5.0], [0.0,1.0], [0.0,0.01],)
#constraint = {'type' : 'ineq', 'fun' : lambda x: 2 * x[1] * x[2] - 10 * x[3] ** 2}
x = minimize(
    objective, 
    x0, 
    args=(y[:50,3]), 
    method="Nelder-Mead", 
    bounds=bounds, 
    #constraints=constraint,
    tol=1e-3,
    options = {
        "maxfev" : 1e4,
        }
    )