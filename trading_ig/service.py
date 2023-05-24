#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:59:55 2022

@author: mtolladay
"""

import json
import logging
import time
from base64 import b64encode, b64decode

from Crypto.Cipher import PKCS1_v1_5
from Crypto.PublicKey import RSA
from requests import Session
from urllib.parse import urlparse, parse_qs


    
class IGSException(Exception):
    pass

class IGSession(object):
    D_BASE_URL = {
        "live": "https://api.ig.com/gateway/deal",
        "demo": "https://demo-api.ig.com/gateway/deal",
    }
    
    def __init__(self, login_details):
        self.login_details = login_details
        self.session = None
        self.base_url = self.D_BASE_URL[login_details[login_details['acc_type'].lower()]]
        self._start()
    
    def __del__(self,):
        self._close()
    
    def _call_request_type(self, request_type):
        if request_type == "GET":
            return self.session.get
        elif request_type == "PUT":
            return self.session.put
        elif request_type == "POST":
            return self.session.post 
        elif request_type == "DELETE":
            return self.session.delete
        
        
    def _request(self, endpoint, request_type, body, version):
        url = self.base_url + endpoint
        request = self._call_request_type(request_type)
        self.session.headers.update({'VERSION': version})
        response = request(url, data=body)
        logging.info(f"{request_type} '{endpoint}', resp {response.status_code}")
        if response.status_code != 200:
            raise IGSException(f"Error: {response.status_code} {response.text}")
        return response
        
    def _start(self,):
        self.session = Session()
        login = {"encryptedPassword" : False, #(Boolean) 	Whether the password has been sent encrypted.
                 "identifier" : self.login_details.identifier, #(String) 	Client login identifier
                 "password" : self.login_details.password} # (String) 	Client login password
        response = self._request("/session", "POST", body=login, version=1)
        if "CST" in response.headers:
            self.session.headers.update({'CST': response.headers['CST']})
        if "X-SECURITY-TOKEN" in response.headers:
            self.session.headers.update({'X-SECURITY-TOKEN': response.headers['X-SECURITY-TOKEN']})

        return response

    def _close(self,):
        _ = self._request("/session", "DELETE", body=None, version=1)
        self.session.close()
        self.session = None
        
    def _check(self,):
        response = self._request("/session", "GET", body=None, version=1)
        return response

class IG
        
def parse_response(*args, **kwargs):
    """Parses JSON response
    returns dict
    exception raised when error occurs"""
    response = json.loads(*args, **kwargs)
    if "errorCode" in response:
        raise (Exception(response["errorCode"]))
    return response