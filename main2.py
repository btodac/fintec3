#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:33:52 2022

@author: mtolladay
"""
import time
import traceback
import threading
import logging
import pickle

import pandas as pd

from brokerinterface import IGManager, PositionManager, MarketAgent, SessionManager
from utillities.user_data import Account
####################################################################################################
account_type = 'live'
tickers = ['^GDAXI','^NDX']
####################################################################################################
log_path = "/home/mtolladay/Documents/finance/logs/"
log_suffix = str(pd.Timestamp.now()).replace(' ','_').replace(':','-').split('.')[0] \
    +".log"
logname = "/home/mtolladay/Documents/finance/logs/debug_nb_" \
    + str(pd.Timestamp.now()).replace(' ','_').replace(':','-').split('.')[0] \
    +".log"
logging.basicConfig(encoding='utf-8')
l = logging.getLogger()
l.setLevel(logging.DEBUG)
l.handlers.clear()
# Handler for broker interface file log
filename = log_path + "broker_interface_" + log_suffix
broker_interface_handler = logging.FileHandler(log_path + "brokerinterface_" + log_suffix)
#broker_interface_handler.addFilter(logging.Filter("brokerinterface"))
broker_interface_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
broker_interface_handler.setLevel(logging.INFO)
# Handler for trading_ig
filename = log_path + "trading_ig_" + log_suffix
ig_handler = logging.FileHandler(log_path + "trading_ig_" + log_suffix)
ig_handler.addFilter(logging.Filter("trading_ig"))
ig_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
ig_handler.setLevel(logging.DEBUG)
# Handler for stream log
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
stream_handler.setLevel(logging.INFO)
# Add handlers
l.addHandler(broker_interface_handler)
l.addHandler(stream_handler)
l.addHandler(ig_handler)

account = Account(account_name=account_type)

ig_manager = IGManager(account)
position_manager = PositionManager(account.acc_number, ig_manager.open_fcn, ig_manager.close_fcn)

agents = []
for ticker in tickers:
    if ticker == "^GDAXI":
        filename = '/home/mtolladay/Documents/finance/NBmodels/NB_GDAXI_HD1KR5/model.pkl'
    elif ticker == '^NDX':
        filename = '/home/mtolladay/Documents/finance/NBmodels/NB_NDX_RLDTC0/model.pkl' #NB_NDX_ABDRVK
        
    with open(filename,'rb') as f:
        model = pickle.load(f)

    model._params['stop_loss'] = 5
    #model._params['time_limit'] = 20
    agent = MarketAgent(model, position_manager, account_type, ig_manager.backfill_fcn,
                  ig_manager.get_details_fcn,
                  size={'live' : 0.5, 'demo' : 1.0}[account_type], cooldown=0, frequency=60)
    print(agent.details._dict)
    agents.append(agent)

session_manager = SessionManager(agents, position_manager, ig_manager)
logging.info("Session starting...")


try:
    time.sleep(10)
    print(str(threading.enumerate()).replace(", <","\n <"))
    while session_manager.session_active:
        time.sleep(5)
except KeyboardInterrupt:
    #print(traceback.print_exc())
    if session_manager.session_active:
        session_manager.stop()
except Exception:
    print(traceback.print_exc())
    if session_manager.session_active:
        session_manager.stop()
logging.info("Session ended. Have a nice day :)")
print(str(threading.enumerate()).replace(", <","\n <"))