#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:36:11 2023

@author: mtolladay
"""
import time
import traceback
from threading import Thread, Event
import logging
from requests.exceptions import ConnectionError

from trading_ig import IGService, IGStreamService

log = logging.getLogger(__name__)

class IGManager(object):
    
    def __init__(self, account):
        self.account = account
        
        self.ig_service = None
        self.ig_stream_service = None
        
        self._subscriptions = {}      
        self._stop_caretaker_thread = Event()
    
    def __del__(self,):
        pass
        #self.stop_stream_service()
        #self.stop_service()
        
    def start(self):
        self.start_service()
        self.start_stream_service()
        
    def stop(self):
        self.stop_stream_service()
        self.stop_service()
           
    def start_service(self):
        if self.ig_service is None:
            ig_service = IGService(
                self.account.username, self.account.password, self.account.api_key, 
                self.account.acc_type, acc_number=self.account.acc_number
            )
            ig_service.create_session()
            self.ig_service = ig_service
    
    def stop_service(self):
        if self.ig_service is not None:
            try:
                self.ig_service.logout()
            except ConnectionError:
                pass
            except:
                print(traceback.print_exc())
            finally:
                self.ig_service = None
            
    def start_stream_service(self):
        self.start_service()
        if self.ig_stream_service is None:
            ig_stream_service = IGStreamService(self.ig_service)
            ig_stream_service.create_session()
            self.ig_stream_service = ig_stream_service
    
    def stop_stream_service(self):
        if self.ig_stream_service is not None:
            if hasattr(self, 'caretaker_thread'):
                self._stop_caretaker_thread.set()
                if self.caretaker_thread.is_alive():
                    self.caretaker_thread.join()
                    del self.caretaker_thread
            #try:
            #    self.ig_stream_service.unsubscribe_all()
            #except:
            #    print(traceback.print_exc())
            try:
                self.ig_stream_service.disconnect()
            except:
                print(traceback.print_exc())
            self.ig_stream_service = None

    def add_subscription(self, subscription, listener_obj):
        subscription.addlistener(self._monitor)
        if self.ig_stream_service is None:
            self.start_stream_service()
            
        sub_id = self.ig_stream_service.ls_client.subscribe(subscription)
        
        if not hasattr(self, 'caretaker_thread'):
            self.caretaker_thread = Thread(target=self._caretaker_loop,
                                           name="IGManager caretaker")
            self.caretaker_thread.start()

        listener_obj.sub_id = sub_id
        self._subscriptions[sub_id] = {'subscription' : subscription,
                                       'listener' : listener_obj,
                                       }
    
    def remove_subscription(self, sub_id):
        if self.ig_stream_service.ls_client is not None:
            self.ig_stream_service.ls_client.unsubscribe(sub_id)
        del self._subscriptions[sub_id]
        if len(self._subscriptions) == 0:
            self.stop_stream_service()
       
    def _monitor(self, data):
        self._is_healthy = True
    
    def _restart_stream_service(self):
        if self.ig_stream_service is not None:
            #try:
            #    self.ig_stream_service.unsubscribe_all()
            #except:
            #    print(traceback.print_exc())
                
            try:
                self.ig_stream_service.disconnect()
            except:
                print(traceback.print_exc())
            finally:
                self.ig_stream_service = None
            
            if self.ig_service is None:
                self.start_service()
                   
            ig_stream_service = IGStreamService(self.ig_service)
            i = 0
            while i < 5:
                try:
                    ig_stream_service.create_session()
                except:
                    i += 1
                    time.sleep(0.1)
                    continue
                else:
                    break
            
            subscriptions = {}
            for sub_id_old in list(self._subscriptions.keys()):
                #subscription =  self._subscriptions.pop(sub_id_old)
                subscription = self._subscriptions[sub_id_old]
                sub_id = ig_stream_service.ls_client.subscribe(subscription['subscription'])
                subscription['listener'].sub_id = sub_id
                subscriptions[sub_id] = subscription             
            self._subscriptions = subscriptions
            self.ig_stream_service = ig_stream_service
        
    def _caretaker_loop(self,):
        log.info('Caretaker loop starting')
        self._is_healthy = True
        while not self._stop_caretaker_thread.wait(45): 
            if not self._is_healthy:
                log.info('Restarting stream!')
                self._restart_stream_service() 
            self._is_healthy = False # This resets the flag, which is set to true every time data is recieved
        log.info('Caretaker loop stopping')
        return None
    
    def open_fcn(self, *args, **kwargs):
        if self.ig_service is None:
            self.start_service()
            
        return self.ig_service.create_open_position(*args, **kwargs)
      
    def close_fcn(self, *args, **kwargs):
        if self.ig_service is None:
            self.start_service()
            
        return self.ig_service.close_open_position(*args, **kwargs)
    
    def backfill_fcn(self, *args, **kwargs):
        if self.ig_service is None:
            self.start_service()
            
        return self.ig_service.fetch_historical_prices_by_epic_and_num_points(*args, **kwargs)
    
    def get_details_fcn(self, *args, **kwargs):
        if self.ig_service is None:
            self.start_service()
            
        return self.ig_service.fetch_market_by_epic(*args, **kwargs)