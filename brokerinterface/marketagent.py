#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:14:09 2023

@author: mtolladay
"""
from collections import deque
import time
import pickle
from threading import Thread, Event
import logging

import numpy as np
import pandas as pd
from trading_ig.lightstreamer import Subscription

from utillities.timesanddates import opening_and_closing_times
from .positionmanager import IGinputError

log = logging.getLogger(__name__)


class MarketDetails(object):
    def __init__(self, details, get_details_fcn):
        self._dict = details
        self.get_details_fcn = get_details_fcn
        self.update_details()        
        
    def __getitem__(self, name):
        return self._dict[name]
    
    def update(self, new_dict):
        self._dict.update(new_dict)
    
    def update_details(self):
        d = self.get_details_fcn(self._dict['epic'])
        details = {'currency_code' : d['instrument']['currencies'][0]['code'],
                   'min_distance' : d['dealingRules']['minNormalStopOrLimitDistance']['value'],
                   'expiry' : d['instrument']['expiry'],
                   }
        
        self._dict.update(details)

class MarketDataStore(object):
    def __init__(self, epic, backfill_fcn, frequency=60):
        self.epic = epic
        self.backfill_fcn = backfill_fcn
        self.observation_length = str(frequency) + "S"
        
        self.back_filled_data = None
        self.data = deque() # thread safe: No need for lock
        self.data_subscription = Subscription(mode="MERGE", 
                                     items=["MARKET:" + epic],
                                     fields=['BID','OFFER','UPDATE_TIME'])  
    
        self.data_subscription.addlistener(self.add_data)
    
    def save(self,):
        filename = '/home/mtolladay/Documents/finance/streamdata/' \
            + self.epic.replace('.','') + '_' + pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d_%X')
            
        with open(filename,'wb') as f:
            pickle.dump(pd.concat(self.data, axis=1).T, f)
    
    def add_data(self, data):
        self.data.append(
            pd.Series(
                data={'Price' : 0.5 * (float(data["values"]['BID']) + float(data["values"]['OFFER']))}, 
                name=pd.Timestamp(data["values"]['UPDATE_TIME'], tz='Europe/London').tz_convert('UTC')
                )
            )
    
    def get_data(self,):
        if len(self.data) != 0:
            data = pd.concat(self.data, axis=1).T.resample(self.observation_length).ohlc().ffill()
            data.columns = [c.capitalize() for c in data.columns.droplevel(0)]
        else:
            data = pd.DataFrame(columns=['Open','High','Low','Close'])
            
        if self.back_filled_data is not None:
            ### TODO: This data needs to combine partial data at the joining point!
            data = pd.concat((self.back_filled_data, data),axis=0)
        
        log.debug(f"Data shape = {data.shape}")
        return data
    
    def back_fill_data(self, t_req):
        if t_req > 0:
            data = self.backfill_fcn(
                epic=self.epic, resolution='1Min', numpoints=t_req
                )
            
            dfa = data['prices'].loc[:,pd.IndexSlice['ask',:]] # OHLC for ask price
            dfb = data['prices'].loc[:,pd.IndexSlice['bid',:]] # OHLC for bid prive
            dfa.columns = dfa.columns.droplevel(0)
            dfb.columns = dfb.columns.droplevel(0)
            spreads = dfa - dfb
            mid_values = dfa - 0.5 * spreads
            mid_values.index = mid_values.index.tz_localize('Europe/London').tz_convert('UTC')### Hack
            
            log.debug(mid_values.to_string())
            self.back_filled_data = mid_values[:-1].resample(self.observation_length).ffill().bfill()

class MarketAgent(object):
    
    available_epics = {'LIVE' : {'^GDAXI' : "IX.D.DAX.IFMM.IP", "^NDX" : "IX.D.NASDAQ.IFS.IP"},
                       'DEMO' : {'^GDAXI' : "IX.D.DAX.IFS.IP", "^NDX" : "IX.D.NASDAQ.IFS.IP"}}
    
    def __init__(self, model, position_manager, account_type, backfill_fcn, get_details_fcn, 
                 size=0.5, cooldown=5, frequency=60):
        log.debug('Initializing OrderBot9000 for ' + model._params['ticker'])
        self.model = model
        self.position_manager = position_manager
        self.frequency = frequency # seconds
               
        self.details = MarketDetails(
            {'epic' : MarketAgent.available_epics[account_type.upper()][model._params['ticker']], 
             'ticker' : model._params['ticker'],
             'stop_loss' : model._params['live_sl'],
             'take_profit' : model._params['live_tp'],
             'time_limit' : model._params['live_tl'],
             'cooldown' : cooldown,
             'size' : size,
             'trailing_stop': True,
             'trailing_stop_increment' : 5, # if above is false use None
            },
            get_details_fcn=get_details_fcn
            )
       
        self.opening_time, self.closing_time = opening_and_closing_times(model._params['ticker'])
        self.market_data_store = MarketDataStore(self.details['epic'], backfill_fcn, self.frequency)
        self._quit_event = Event()
        self.signal_generation_thread = Thread(target=self.signal_generation_loop, 
                                               name="Agent signal generator")
        self.sub_id = None
        
    @property
    def is_active(self):
        return self.signal_generation_thread.is_alive()
    
    @property
    def is_in_trading_period(self):
        t = pd.Timestamp.now(tz='UTC')
        return t.time() >= self.opening_time.time() and t.time() < self.closing_time.time()
    
    def __del__(self):
        pass
        '''
        self.stop()
        self.market_data_store = None
        self.position_manager = None
        self.get_details_fcn = None
        '''
    
    def start(self):
        # Back fill agent data
        t = pd.Timestamp.now(tz='UTC')
        t_req = min(int(np.floor((t - self.opening_time).total_seconds()/60)), 512)
        self.market_data_store.back_fill_data(t_req)
        # Start the signal generation loop
        self._quit_event.clear()
        if not self.signal_generation_thread.is_alive():
            log.info(f"Starting signal generator thread for {self.details['ticker']}")
            try:
                self.signal_generation_thread.start()
            except RuntimeError:
                self.signal_generation_thread = Thread(target=self.signal_generation_loop, 
                                                       name="Agent signal generator")
                self.signal_generation_thread.start()
                
        return self.market_data_store.data_subscription
   
    def stop(self):
        # Clear positions
        log.info(f"Stopping agent for {self.details['ticker']}")
        self.position_manager.clear(epic=self.details['epic'])
        if self.is_active:
            self._quit_event.set()
            self.signal_generation_thread.join()
        self.market_data_store.save()
        
    def signal_generation_loop(self,):
        while not self._quit_event.wait(1): #self.frequency): ### TODO: Pandas resamples 60s at whole minute intervals
            t = pd.Timestamp.now('UTC')
            if t.second >= 58:
                log.debug("Generating signal...")
                action = self.model(self.market_data_store.get_data())
                if action != "HOLD":
                    try:
                        self.position_manager.open_position(action, self.details)
                    except IGinputError:
                        self.details.update_details()
                        self.position_manager.open_position(action, self.details)
                log.info(f"{str(t).split('.')[0]} {self.details['ticker']}: {action}")
                time.sleep(5) # Wait long enough to ensure only one call is made per minute
                    
