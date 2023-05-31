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

from agents.testagent import TestAgent
from brokerinterface import IGManager, PositionManager, MarketAgent, SessionManager
from utillities.user_data import Account
####################################################################################################
account_type = 'demo'
tickers = ['^GDAXI',]#'^NDX']
####################################################################################################
logname = "/home/mtolladay/Documents/finance/logs/debug_nb_" \
    + str(pd.Timestamp.now()).replace(' ','_').replace(':','-').split('.')[0] \
    +".log"
logging.basicConfig(encoding='utf-8')
l = logging.getLogger()
l.setLevel(logging.DEBUG)
l.handlers.clear()
file_handler = logging.FileHandler(logname)
file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
l.addHandler(file_handler)
l.addHandler(stream_handler)

account = Account(account_name=account_type)

ig_manager = IGManager(account)
position_manager = PositionManager(account.acc_number, ig_manager.open_fcn, ig_manager.close_fcn)

agents = []
for ticker in tickers:
    model = TestAgent(ticker,['2min_Trend','4min_Trend'],)

    agent = MarketAgent(model, position_manager, account_type, ig_manager.backfill_fcn,
                  ig_manager.get_details_fcn,
                  size={'live' : 0.5, 'demo' : 1.0}[account_type], cooldown=0, frequency=60)
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