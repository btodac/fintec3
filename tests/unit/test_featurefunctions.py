#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 21:31:19 2023

@author: mtolladay
"""
import unittest

import pandas as pd
from agents import featurefunctions

t = pd.Timestamp.now().normalize()
test_data = pd.DataFrame(
    {'Open' : [1,1,1,1,2,3,3,4],
     'High' : [2,2,2,3,4,4,5,4],
     'Low' :  [0,0,0,1,2,2,3,2],
     'Close': [1,1,1,2,3,3,4,3],},
    index = pd.date_range(t, t + pd.Timedelta(minutes=7), freq='min')
    )

class TestAvgTrueRange(unittest.TestCase):
    
    def setUp(self,):
        self.data = test_data
        self.atr = featurefunctions.AvgTrueRange()
    
    def test_avgtruerange(self,):
        pass

if __name__ == '__main__':
    unittest.main()