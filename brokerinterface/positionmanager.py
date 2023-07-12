#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:39:36 2023

@author: mtolladay
"""
import time
from threading import Thread, Event, RLock
import json
import logging
import pandas as pd
from trading_ig.lightstreamer import Subscription
from trading_ig.rest import IGException

log = logging.getLogger(__name__)

RED = "\x1b[31;1m"
GREEN = "\x1b[32;1m"
YELLOW = "\x1b[33;1m"
BLUE = "\x1b[34;1m"
PINK = "\x1b[35;1m"
CYAN = "\x1b[36;1m"
WHITE = "\x1b[37;1m"

def direction_to_int(direction):
    return {"BUY" : 1, "SELL" : -1}[direction]

def direction_invert(direction):
    return {"BUY" : "SELL", "SELL" : "BUY"}[direction]

class IGinputError(ValueError):
    def __init__(self):
        pass
    
class PositionManager(object):
    '''
    Manages all positions for a session. Uses a LS connection to monitor 
    and mirror changes to OTC positions. 
    '''
    
    def __init__(self, n_max_positions=1):
        self.n_max_positions = n_max_positions
        
        self._positions_lock = RLock()
        self.positions = {}
        self.closed_positions = {}
        
        self._cooldown_end_time = pd.Timestamp.now('UTC')
        self._is_in_cooldown = False
        
        self.sub_id = None
        self._stop_updating = Event()
        
        self.update_thread = Thread(target=self.update_loop, name="Position manager updater")
    
    def __del__(self):
        pass
        '''
        self.stop()
        self.open_fcn = None
        self.close_fcn = None
        '''       
    
    @property
    def is_active(self):
        return self.update_thread.is_alive()
    
    def start(self, ig_manager):
        self.open_fcn = ig_manager.open_fcn
        self.close_fcn = ig_manager.close_fcn     
        self.subscription = Subscription(
            mode="DISTINCT", 
            items=["TRADE:" + ig_manager.account.acc_number], 
            fields=["OPU"]
            )
        self.subscription.addlistener(self.position_watcher)
        ig_manager.add_subscription(
            self.subscription,
            self
            )
        
        self._stop_updating.clear()
        if not self.update_thread.is_alive():
            log.info('Starting Position Manager update thread')
            try:
                self.update_thread.start()
            except RuntimeError:
                self.update_thread = Thread(
                    target=self.update_loop, name="Position manager updater",
                    )
                self.update_thread.start()
        
    def stop(self,):
        log.info('Stopping Position Manager update thread')
        if self.update_thread.is_alive():
            self._stop_updating.set()
            self.update_thread.join()
        
        self.clear()
        
        filename = '/home/mtolladay/Documents/finance/trades/' \
            + 'PosManTrades_' + pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d_%X')
            
        with open(filename,'w') as f:
            if len(self.closed_positions) > 0:
                df = pd.DataFrame(self.closed_positions).T
                df = df.set_index('date')
                print(df.to_string(), file=f)
        
    def update_loop(self):
        while not self._stop_updating.wait(1):
            self._is_in_cooldown = pd.Timestamp.now('UTC') < self._cooldown_end_time
            
            with self._positions_lock:
                now_time = pd.Timestamp.now('UTC')
                for dealId in list(self.positions.keys()):
                    if now_time > self.positions[dealId]['endtime']:
                        log.info(f"Closing {dealId} due to timeout")
                        self.close_position(dealId, outcome="timeout/external")
                    
    def position_watcher(self, stream_data):
        data = stream_data['values']['OPU']
        if data is not None:
            opu = json.loads(data)
            direction = direction_to_int(opu['direction'])
            dealId = opu['dealId']
            status = opu['status']
            level = opu['level']
            dlevel = direction * level
            stopLevel = direction * opu['stopLevel']
            limitLevel = direction * opu['limitLevel']
            if status == "OPEN":
                log.info(GREEN + f'Deal opened {dealId} at {level}' + WHITE)
            elif status == "DELETED":
                reason = None
                if limitLevel is not None:
                    if dlevel >= limitLevel:
                        reason = "take profit hit"
                        color = BLUE

                if stopLevel is not None:
                    if dlevel <= stopLevel:
                        reason = "stop loss hit"
                        color = RED
                        
                if reason is None:
                    reason = 'timeout/external'
                    color = YELLOW
                    
                self.close_position(dealId, reason, data=opu)
                log.info(color + f'Deal {dealId} closed due to {reason} price = {level}' + WHITE)
            elif status == "UPDATED":
                log.info(CYAN + f'Deal {dealId} {status} current = {level}' + WHITE)
                
    def open_position(self, direction, details):
        with self._positions_lock:
            if len(self.positions) >= self.n_max_positions:
                text = PINK + "Unable to open position max positions are active"
            elif  pd.Timestamp.now('UTC') < self._cooldown_end_time:
                text = PINK + "Unable to open position, bot in cooldown"
            else:
                self._open_position(direction, details)
                text = GREEN + "Position opened"
            log.info(text + WHITE)
        
    def close_position(self, dealId, outcome, data=None):
        log.debug(f'Attempting to close position {dealId} due to {outcome}')
        is_closed = True
        with self._positions_lock:
            if dealId in self.positions.keys():
                if outcome.lower() in ["timeout/external","reverse posiiton","clearing"]:
                    is_closed = self._close_position(dealId)
                elif outcome.lower() == "stop loss hit":
                    #self._is_in_cooldown = True
                    self._cooldown_end_time = pd.Timestamp.now('UTC') \
                        + pd.Timedelta(minutes=self.positions[dealId]['cooldown'])
                elif outcome.lower() == "take profit hit":
                    pass
    
                if is_closed:
                    deal = self.positions.pop(dealId)
                    if data is not None:
                        deal.update(data)
                    self.closed_positions[dealId] = deal
                else:
                    print(f'ERROR failed to close position {dealId}!')
            else:
                log.info(f"Deal {dealId} is already deleted!")
            
    def clear(self, epic=None):
        with self._positions_lock:
            if epic is None:
                dealIds = list(self.positions.keys())
            else:
                dealIds = [dealId for dealId, pos_data in self.positions.items() if pos_data['epic'] == epic]
                
            for dealId in dealIds:
                self.close_position(dealId, "CLEARING")
    
    def _open_position(self, direction, details):
        i = 0
        pos_data = None
        while True and i < 5:
            try:
                pos_data = self.open_fcn(
                    currency_code=details['currency_code'],
                    direction=direction,
                    epic=details['epic'],
                    order_type='MARKET',
                    expiry=details['expiry'],
                    force_open='true',
                    guaranteed_stop='false',
                    size=details['size'], 
                    level=None,
                    limit_distance=max(details['min_distance'], details['take_profit']),
                    limit_level=None,
                    quote_id=None,
                    stop_level=None,
                    stop_distance=max(details['min_distance'], details['stop_loss']), 
                    trailing_stop='true',#str(details['trailing_stop']).lower(),#'false'/'true',
                    trailing_stop_increment=20,#None,#details['trailing_stop_increment'],
                    )
            
                #log.info(print(pos_data))

            except:
                i += 1
                time.sleep(0.1)
                continue
            else:
                if pos_data['dealStatus'] == "REJECTED":
                    log.info(print(pos_data))
                    raise IGinputError
                pos_data.update(details._dict)
                t_start = pd.Timestamp(pos_data['date']).tz_localize('UTC')
                pos_data['endtime'] = t_start + pd.Timedelta(minutes=details['time_limit'])
                self.positions[pos_data['dealId']] = pos_data
            break
        
        log.debug(pos_data)
            
    def _close_position(self, dealId):
        log.info('Closing from bot')
        success = False
        i = 0
        while True and i < 5:
            try:
                pos_data = self.close_fcn(
                    deal_id=dealId,
                    direction=direction_invert(self.positions[dealId]['direction']),
                    epic=None,
                    expiry=None,
                    level=None,
                    order_type='MARKET',
                    quote_id=None,
                    size=self.positions[dealId]['size'],
                    session=None,
                )
                #print(pos_data)
            except Exception as e:
                if str(e).find("invalid.url") != -1:
                    success = True
                    log.info(f'Position {dealId} was not found!')
                    break
                #print(traceback.print_exc())
                i += 1
                time.sleep(0.2)
                continue
            else:
                success = True
                log.info(f'Position {dealId} closed')
            break
        return success