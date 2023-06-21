#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 21:38:06 2023

@author: mtolladay
"""
import numpy as np
import tensorflow as tf



SPREAD = 0.9

class Position(object):
    
    def __init__(self, opening_price, direction):
        self.opening_price = opening_price
        self.direction = direction
        
    def current_profit_loss(self, price):
        return self.direction * (price - self.opening_price) * 1e4 - SPREAD
    
class Tester(object):
    def __init__(self, env, model):
        self.env = env
        self.model = model
        
        self._last_action = 0
        self.position = None
        self.funds = 0
        self.trade_history = []
        self.funds_history = []
                
    def test(self, ):
        self.funds = 0
        self.trade_history = []
        action_probs_list = []
        while self.env._observer.tick_count < len(self.env._observer)-1:
            state, _ = self.env.make_state()
            action, action_probs = self.take_bot_action(state)
            profit, current_profit = self.process_action(action)
            self.trade_history.append(profit)
            self.funds_history.append((current_profit))
            action_probs_list.append(action_probs)
            #print(f'{state[0,0]} {action_probs},  ${profit}')
        
        action_probs = np.array(action_probs_list)
        print(f"Action probs: {action_probs.mean(axis=0)} +/- {action_probs.std(axis=0)}")
        return self.funds, self.trade_history, self.funds_history
    
    def test2(self,):
        observations = self.observer.make_observations()
        observations = tf.convert_to_tensor(observations)
        #prices = self.observer.closing_prices
        self.funds = 0
        self.trade_history = []
        #action_probs_list = []
        
        predictions = self.model.predict(x=observations, batch_size=512, verbose=0,)
        actions = tf.argmax(predictions, axis=1).numpy()
        delta_action = np.where(actions[1:] != actions[:-1])[0]
        #old_actions = actions[delta_action]
        new_actions = actions[delta_action + 1]
        directions = np.zeros_like(new_actions)
        directions[new_actions==1] = 1
        directions[new_actions==2] = -1
        opening_prices = self.observer.closing_prices[delta_action[:-1]]
        closing_prices = self.observer.closing_prices[delta_action[1:]]
        profits = directions * (closing_prices - opening_prices) * 1e4 - SPREAD
        profits = profits[directions != 0]
        self.funds = sum(profits)
        self.trade_history = profits
        return self.funds, self.trade_history
            
        
        

        while self.observer.tick_count < len(self.observer)-1:
            state = next(self.observer)
            action, action_probs = self.take_bot_action(state)
            profit = self.process_action(action)
            self.trade_history.append(profit)
            action_probs_list.append(action_probs)
            #print(f'{state[0,0]} {action_probs},  ${profit}')
        
        action_probs = np.array(action_probs_list)
        print(f"Action probs: {action_probs.mean(axis=0)} +/- {action_probs.std(axis=0)}")
        return self.funds, self.trade_history
            
    def take_bot_action(self,state):
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        # Take best action
        return tf.argmax(action_probs[0]).numpy(), action_probs
    
    def process_action(self, action):
        profit = 0
        price = self.env._observer.current_price
        if action == 0: # Hold
            if self._last_action == 0: #hold -> hold
                pass
            else:
                profit = self.close_position(price)   
        elif action == 1: # Buy
            if self._last_action == 0: # hold -> buy
                self.open_position(price, 1)
            elif self._last_action == 1: # buy -> buy
                pass
            elif self._last_action == 2:# sell -> buy
                profit = self.close_position(price)
                self.open_position(price, 1)   
        elif action == 2: # Sell
            if self._last_action == 0: # hold -> sell
                self.open_position(price, -1)
            elif self._last_action == 1:# buy -> sell
                profit = self.close_position(price)
                self.open_position(price, -1)
            elif self._last_action == 2: # sell -> sell
                pass
                #    profit = self.position.current_profit_loss(price)
        self._last_action = action
        
        if self.position is not None:
            current_profit = self.funds + self.position.current_profit_loss(price)
        else:
            current_profit = self.funds
        return profit, current_profit 
     
    def open_position(self, price, direction):
        self.position = Position(opening_price=price, direction=direction)
        
    def close_position(self, price):
        profit = self.position.current_profit_loss(price) #- SPREAD
        self.funds += profit
        self.position = None
        return profit           