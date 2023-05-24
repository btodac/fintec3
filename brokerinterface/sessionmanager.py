#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:20:51 2023

@author: mtolladay
"""
from threading import Thread, Event
import logging

log = logging.getLogger(__name__)

class SessionManager(object):

    def __init__(self, agents, position_manager, ig_manager):
        log.debug('Starting session manager')
        self.agents = agents
        self.position_manager = position_manager
        self.ig_manager = ig_manager
        
        self.start()
    
    def __del__(self):
        self.stop()
        self.position_manager = None
        self.ig_manager = None
    
    def start(self,):
        self.session_active = True
        self._stop_session = Event()
        self._main_thread = Thread(target=self.main_loop, name="Session manager main loop")
        self._main_thread.start()
        
    def stop(self,):
        self._stop_session.set()
        self._main_thread.join()
        self.session_active = False
        
    def main_loop(self,):
        # Loops over agents and starts/stops them if in trading hours
        while not self._stop_session.wait(1):
            agents_active = False
            for agent in self.agents:
                if not agent.is_active and agent.is_in_trading_period:
                    self.ig_manager.start_service()                       
                    sub = agent.start()
                    self.ig_manager.add_subscription(sub, agent)  
                elif agent.is_active and not agent.is_in_trading_period:
                    if agent.sub_id is not None:
                        self.ig_manager.remove_subscription(agent.sub_id) 
                        
                agents_active = agents_active or agent.is_active
            
            if agents_active: 
                if self.position_manager.sub_id is None:
                    self.ig_manager.add_subscription(self.position_manager.subscription,
                                                     self.position_manager)
            else:
                if self.position_manager.sub_id is not None:
                    self.ig_manager.remove_subscription(self.position_manager.sub_id)
                    self.position_manager.sub_id = None
                    
                self.ig_manager.stop_stream_service()
                self.ig_manager.stop_service()
        # Close all agents
        for agent in self.agents:
            if agent.is_active:
                agent.stop()
                if agent.sub_id is not None:
                    self.ig_manager.remove_subscription(agent.sub_id)
                    
        self.position_manager.stop()
        if self.position_manager.sub_id is not None:
            self.ig_manager.remove_subscription(self.position_manager.sub_id)
            
        self.ig_manager.stop_stream_service()
        self.ig_manager.stop_service()