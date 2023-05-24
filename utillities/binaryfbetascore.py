#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:18:11 2022

@author: mtolladay
"""
import tensorflow  as tf

class BinaryFBetaScore(tf.keras.metrics.Metric):
    # ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
    def __init__(self, beta=0.5, threshold=0.5, name='binary_f_Beta', **kwargs):
        super(BinaryFBetaScore, self).__init__(name=name, **kwargs)
        self.beta = beta
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_negatives = self.add_weight(name='tp', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.math.greater(y_pred, self.threshold) # tf.cast(y_pred, tf.bool)

        def _to_values(values, sample_weight=None):
            values = tf.cast(values, self.dtype)
            if sample_weight is not None:
                sample_weight = tf.cast(sample_weight, self.dtype)
                values = tf.multiply(values, sample_weight)
            return values

        values = _to_values(tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True)))
        self.true_positives.assign_add(tf.reduce_sum(values))
        values = _to_values(tf.logical_and(tf.equal(tf.logical_not(y_true), True), tf.equal(y_pred, True)))
        self.false_positives.assign_add(tf.reduce_sum(values))
        values = _to_values(tf.logical_and(tf.equal(y_true, True), tf.equal(tf.logical_not(y_pred), True)))
        self.false_negatives.assign_add(tf.reduce_sum(values))

    def result(self):
        p = tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_positives),
                )
        r = tf.math.divide_no_nan(
                self.true_positives,
                tf.math.add(self.true_positives, self.false_negatives),
                )
        beta2 = self.beta**2
        one_beta2 = 1 + beta2
        fb = tf.math.divide_no_nan(
            tf.math.multiply_no_nan(one_beta2, tf.math.multiply_no_nan(p, r)),
            tf.math.add(tf.math.multiply_no_nan(beta2, p), r)
            )

        return fb

    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)
        
    def get_config(self):
        config = super().get_config()#.copy()
        config.update({
            'beta' : self.beta,
            'threshold': self.threshold
        })
        return config