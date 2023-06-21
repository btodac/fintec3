#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 10:54:01 2023

@author: mtolladay
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class RLAgent(object):
    def __init__(self,input_shape):
        
        self.optimizer = keras.optimizers.Adam(learning_rate=0.0001, 
                                          clipnorm=1.0)
        # Using huber loss for stability
        self.loss_function = keras.losses.Huber()
        
        self.model = self.create_q_model(input_shape,)
        self.model_target = self.create_q_model(input_shape)
    
    def create_q_model(self, input_data_size,):
        model = keras.Sequential()
        model.add(layers.Input(shape=input_data_size))
        model.add(layers.Flatten())
        for i in range(5):
            model.add(layers.Dense(
                128, activation='relu',
            #                       kernel_constraint=keras.constraints.MaxNorm(max_norm),
                ))
        model.add(layers.Dense(3, activation='softmax', ))  
        
        model.compile()
        return model
    
    def predict(self, observation):
        state_tensor = tf.convert_to_tensor(observation)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_probs = self.model(state_tensor, training=False)
        # Take best action
        return tf.argmax(action_probs[0]).numpy(), action_probs
    
    def update_model(self, state_sample, state_next_sample, rewards_sample, 
                     action_sample, done_sample, gamma):

        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        state_sample = tf.convert_to_tensor(state_sample)
        state_next_sample = tf.convert_to_tensor(state_next_sample)
        future_rewards = self.model_target.predict(state_next_sample, verbose=0)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + \
                            gamma * tf.reduce_max(future_rewards, axis=1) #+\
                            #(1 - self.alpha) * self.model(state_sample) # TODO This line ay be balls!

        # If final frame set the last value to -1
        updated_q_values = updated_q_values * (1 - done_sample) - done_sample

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, 3)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            # state_sample = tf.convert_to_tensor(state_sample)
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
        return None

    def update_target_network(self):
        self.model_target.set_weights(self.model.get_weights())