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
            agents_active = self.update_agents()
            # Start/Stop position manager
            if not agents_active:
                self.stop_position_manager()
                self.ig_manager.stop()
        
        self.stop_agents()
        self.stop_position_manager() 
        self.ig_manager.stop()
        
    def update_agents(self):
        agents_active = False
        for agent in self.agents:
            if not agent.is_active and agent.is_in_trading_period:                      
                agent.start()
                self.ig_manager.add_subscription(
                    agent.subscription, 
                    agent
                    )
                if self.position_manager.sub_id is None:
                    self.ig_manager.add_subscription(
                        self.position_manager.subscription,
                        self.position_manager
                        )
            elif agent.is_active and not agent.is_in_trading_period:
                agent.stop()
                if agent.sub_id is not None:
                    self.ig_manager.remove_subscription(agent.sub_id)
                    
            agents_active = agents_active or agent.is_active
        return agents_active
        
    def stop_position_manager(self):
        self.position_manager.stop()
        if self.position_manager.sub_id is not None:
            self.ig_manager.remove_subscription(self.position_manager.sub_id)
            self.position_manager.sub_id = None
    
    def stop_agents(self):
        for agent in self.agents:
            if agent.is_active:
                agent.stop()
                if agent.sub_id is not None:
                    self.ig_manager.remove_subscription(agent.sub_id)