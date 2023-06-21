#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:38:50 2023

@author: mtolladay
"""
import logging

log = logging.getLogger(__name__)

SPREAD = 0.2#1.2 #2


class Position(object):
    
    def __init__(self, opening_price, direction):
        self.opening_price = opening_price
        self.direction = direction
        self.max_profit = -SPREAD
        
    def current_profit_loss(self, price):
        return self.direction * (price - self.opening_price) * 1e4 - SPREAD
    
    def profit_delta(self, price):
        profit = self.current_profit_loss(price)
        return profit
    
class Broker(object):
    def __init__(self, market_data):
        self.market_data = market_data
        self.positions = []
        self.funds = 0.0
    
    def open_position(self, direction):
        self.positions.append(Position(self.market_data.current_price))
    
    def close_position(self,):
        position = self.positions.pop()
        return position.current_profit_loss(self.market_data.current_price)
        
    @property
    def current_profit_loss(self,):
        price = self.market_data.current_price
        pl = 0.0
        for position in self.positions:
            pl += position.current_profit_loss(price)
        
        return pl
    
    def reset(self,):
        self.positions = []
        self.funds = 0.0
        