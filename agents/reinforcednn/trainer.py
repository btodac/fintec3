#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:33:20 2023

@author: mtolladay
"""
import os
#import sys
import pickle
#import string
#import random
import gc
import logging
import traceback

import numpy as np
#import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

class TrainingParams(object):
    
    def __init__(self):
        #self.alpha = 0.5 # Q-learning, learn rate
        self.gamma = 0.1  # Discount factor for past rewards
        self.gamma_max = 0.9
        self.gamma_interval = (self.gamma_max - self.gamma)
        self.epsilon = 1.0  # Epsilon greedy parameter
        self.epsilon_min = 0.05  # Minimum epsilon greedy parameter
        self.epsilon_max = self.epsilon  # Maximum epsilon greedy parameter
        self.epsilon_interval = (
            self.epsilon_max - self.epsilon_min
            )  # Rate at which to reduce chance of random action being taken
        self.batch_size = 32  # Size of batch taken from replay buffer
        self.num_actions = 3 # (BUY/SELL/HOLD)
        self.max_frames = 9e5
        # Number of frames to take random action and observe output
        self.epsilon_random_frames = 5e4
        # Number of frames for exploration
        self.epsilon_greedy_frames = 2e5
        # Maximum replay length
        self.max_memory_length = 1e5
        # Train the model after 4 actions
        self.update_after_actions = 8
        # How often to update the target network
        self.update_target_network = 5e4 # 100 * self.update_after_actions
        # Save every n frames
        self.n_frames_save = 5e4
        
    def decay_epsilon(self):
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)
        self.gamma += self.gamma_interval / self.epsilon_greedy_frames
        self.gamma = min(self.gamma, self.gamma_max)

class History(object):
    def __init__(self, max_memory_length):
        self.max_memory_length = max_memory_length
        
        self.rewards_history = []
        self.state_history = []
        self.state_next_history = []
        self.action_history = []
        self.done_history = []
        #self.profit_hitsory = []
        
    def __len__(self,):
        return len(self.action_history)
        
    def update(self, state, state_next, reward, action, done,):
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)
        if len(self.action_history) > self.max_memory_length:
            self._apply_limit()
    
    def get_batch(self, batch_size):
        indices = np.random.choice(np.arange(len(self.done_history)), 
                                   size=batch_size, 
                                   replace=False)

        states = [self.state_history[i] for i in indices]
        states_next = [self.state_next_history[i] for i in indices]        
        state_sample = np.stack(states)
        state_next_sample = np.stack(states_next)    
        rewards_sample = [self.rewards_history[i] for i in indices]
        action_sample = [self.action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor(
            [float(self.done_history[i]) for i in indices]
        )
        return state_sample, state_next_sample, rewards_sample, action_sample, done_sample
    
    def _apply_limit(self):
        del self.rewards_history[:1]
        del self.state_history[:1]
        del self.state_next_history[:1]
        del self.action_history[:1]
        del self.done_history[:1]

class TrainingState(object):
    def __init__(self, 
                 state = None,
                 frame_count = 0,
                 running_reward = 0,
                 episode_count = 0,
                 wins = 0,
                 avg_t = 0,
                 t_to_end = 0,
                 outcomes = [],
                 episode_reward_history = [],
                     ):
        self.frame_count = frame_count
        self.running_reward = running_reward
        self.episode_count = episode_count
        self.wins = wins
        self.avg_t = avg_t
        self.t_to_end = t_to_end
        self.outcomes = outcomes
        self.episode_reward_history = episode_reward_history

class EarlyStopping(object):
    def __init__(self, wait_limit=100000):
        self.wait_limit = wait_limit
        
        self.old_wins = 0
        self.count = 0
        self.model_weights = None
        
    def stop(self, wins):
        # early stopping
        store_model_weights = wins > self.old_wins
        if store_model_weights:
            self.count = 0
            self.old_wins = wins
        else:
            self.count += 1
            
        stop_training = self.count >= self.wait_limit
        
        return store_model_weights, stop_training           

class Trainer(object):
    
    def __init__(self, rl_agent=None, env=None, model_dir=None,
                 optimizer = None, loss_function = None, training_params = None,
                 restart=False
                 ):
        if restart:
            try:
                self.load_optimizer_state(model_dir)
            except:
                raise ValueError(f"If restart is set to True then model_dir must be supplied")
            #else:
            #    self.train(**self.training_state.__dict__)
        else:
            self.rl_agent = rl_agent
            self.model_dir = model_dir
            self.env = env
            self.optimizer = optimizer
            self.loss_function = loss_function
            self.params = training_params  
            
            self.early_stopping = EarlyStopping()
            self.history = History(training_params.max_memory_length)
            self.training_state = TrainingState()
                  
    def run(self):
        self.train(**self.training_state.__dict__)
        
    def train(self, state=None, frame_count=0, running_reward=0, episode_count=0, wins=0, avg_t=0,
              t_to_end=0, outcomes = [], episode_reward_history=[], episode_reward = 0):
        
        while True:  # Run until solved
            if state is None:
                state, done  = self.env.reset()

            while True:
                frame_count += 1
                action, action_probs = self.make_action(frame_count, state)

                # Apply the sampled action in our environment
                state_next, reward, done, = self.env.step(action)
                #'''
                #sys.stdout.write("\033[K")
                #sys.stdout.flush()
                running_str = f'Running reward {running_reward:6.2f} E: {episode_count} '\
                      f'W/L:{wins:>6.3f} Avg time: {avg_t:5.0f} '\
                      f'frames: {frame_count:<9d}, P/L: {self.env.broker.funds:9.2f}, action: {action}, '\
                      f'reward: {reward:6.2f} ' \
                      f'{action_probs[0]}'
                print(f'\r{running_str:190}', end='\r', flush=True)
                #'''
                episode_reward += reward

                self.history.update(state, state_next, reward, action, done)

                state = state_next

                # Update every fourth frame and once batch size is over 32
                if frame_count % self.params.update_after_actions == 0 and \
                        len(self.history) > self.params.batch_size:                       
                    self.rl_agent.update_model(
                        *self.history.get_batch(self.params.batch_size), 
                        self.params.gamma
                        )
                
                if frame_count % self.params.update_target_network == 0:
                    # update the the target network with new weights
                    self.rl_agent.update_target_network()
                
                if frame_count % self.params.n_frames_save == 0:
                    training_state = TrainingState(
                        state, frame_count, running_reward , episode_count,
                        wins, avg_t, t_to_end, outcomes, episode_reward_history,)
                    self.save_optimizer_state(training_state)
                    
                if done:
                    aw = 0.05
                    #wins = wins * (1-aw) + (reward>100) *aw # sum(outcomes) / len(outcomes)
                    wins = wins * (1-aw) + (self.env.broker.funds > 0) *aw # sum(outcomes) / len(outcomes)
                    outcomes.append(wins)
                    if len(outcomes) > 100000:
                        del outcomes[:1]
                    #avg_t = (avg_t * episode_count + (frame_count - t_to_end)) / (episode_count + 1)
                    avg_t = avg_t * (1-aw) + (frame_count - t_to_end) * aw
                    t_to_end = frame_count                    
                    print("", end="\n",flush=True)# Use newline to keep last line from running output
                    break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            episode_count += 1
            '''
            if frame_count > self.params.epsilon_greedy_frames:
                save_model_weights, stop_training = self.early_stopping.stop(wins)
                if save_model_weights:
                    self.early_stopping.model_weights = self.model.get_weights()
            else:
                save_model_weights, stop_training = False, False
            '''
            save_model_weights, stop_training = False, False

            if (running_reward > 700 and frame_count > self.params.epsilon_greedy_frames) or\
                    (wins > 0.66 and frame_count > self.params.epsilon_greedy_frames) or\
                    stop_training or \
                    frame_count > self.params.max_frames:  # Condition to consider the task solved

                #if stop_training:
                #    self.model.set_weights(self.early_stopping.model_weights)
                
                self.save_optimizer_state(
                    TrainingState(state, frame_count, running_reward , episode_count,
                                  wins, avg_t, t_to_end, outcomes, episode_reward_history,)
                    )
                print(f"Exit at {frame_count}!")
                break
            state, done = self.env.reset()
            episode_reward = 0
            
            gc.collect()
    
    def make_action(self, frame_count, observation):
        # Use epsilon-greedy for exploration
        if frame_count < self.params.epsilon_random_frames or\
            self.params.epsilon > np.random.rand(1)[0]:
            action, action_probs = self._take_random_action()
        else:
            action, action_probs = self.rl_agent.predict(observation)
        
        if frame_count > self.params.epsilon_random_frames:
            self.params.decay_epsilon()
            
        return action, action_probs
    
    
    def _take_random_action(self):
        action = np.random.choice(self.params.num_actions)
        action_probs = np.zeros(self.params.num_actions)
        action_probs[action] = 1
        return action, action_probs[np.newaxis,:]
         
    def save_optimizer_state(self, training_state):
        #self.model.save(self.model_dir,)
        try:
            os.makedirs(self.model_dir)
        except OSError:
            pass
        data = {'rl_agent' : self.rl_agent,
                'env' : self.env,
                'params' : self.params,
                #early_stopping' : self.early_stopping,
                'history' : self.history,
                'training_state' : training_state,
                'model_dir' : self.model_dir,
                }
        
        run_state_file = self.model_dir + '/run_state.pkl'
        if os.path.isfile(run_state_file): 
            os.system("cp " + run_state_file + " " \
                  + self.model_dir + '/run_state_bkup.pkl')
                
        with open(run_state_file, 'wb') as f:            
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_optimizer_state(self, model_dir):
        try:
            with open(model_dir + '/run_state.pkl', 'rb') as f:        
                state = pickle.load(f)
        except:
            print('Last backup is corrupted loading previous state')
            with open(model_dir + '/run_state_bkup.pkl', 'rb') as f:        
                state = pickle.load(f)
            
        self.rl_agent = state['rl_agent']
        self.model_dir = state['model_dir']
        self.env = state['env']
        self.params = state['params']
        #self.early_stopping = state['early_stopping']
        self.history = state['history']
        self.training_state = state['training_state']
            
    

    

