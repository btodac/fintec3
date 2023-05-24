#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:44:27 2023

@author: mtolladay
"""
import string
import os
import pickle
import shelve
import getpass

from utillities.pass_encryptor import password_decrypt, password_encrypt

class AccountDetails(object):
    def __init__(self, username=None, password=None, api_key=None, acc_type=None, acc_number=None):
        self.username = username
        self.password = password
        self.api_key = api_key
        self.acc_type = acc_type
        self.acc_number = acc_number
        
    def _print_details(self):
        print(f'Name: {self.username}')
        print(f'Account type: {self.acc_type}')
        print(f'API key: {self.api_key}')        
        print(f'Account number: {self.acc_number}')
        
    def as_dict(self):
        return self.__dict__
       
class Account(object):
    
    _path_to_store = os.environ['HOME'] +'/finance/OrderBot/'
    _account_file_store = _path_to_store + 'Accounts.dbm'
    
    def __init__(self, account_name=None, show_available=False):
        
        if not os.path.exists(Account._path_to_store):
            os.makedirs(Account._path_to_store)
        
        if account_name is None:
            if show_available:
                self._account = self._choose_account()
            else:
                self._account = self._get_new_account()
        else:
            self._account = self._load_account(account_name)
    
    @property
    def name(self):
        return self._name
    
    @property
    def username(self):
        return self._account.username
    
    @property
    def password(self):
        return self._account.password
    
    @property
    def api_key(self):
        return self._account.api_key
    
    @property
    def acc_type(self):
        return self._account.acc_type
    
    @property
    def acc_number(self):
        return self._account.acc_number
    
    def as_dict(self):
        return self._account.as_dict()
    
    
            
    def _get_new_account(self,):
        print('Creating a new IG access account')
        
        name = input('Enter the name for the new account: ')
        password = self._get_new_password()
        
        is_good = False
        while not is_good:
            acc_username = input('Enter account username: ')
            acc_type = self._get_acc_type_()
            acc_api_key = input('Enter the api_key: ')
            acc_number = input('Enter the account number: ')
            acc_password = input('Enter the account password: ')
            
            print('Please check details: ')
            print(f'Account username: {acc_username}')
            print(f'Account type: {acc_type}')
            print(f'API key: {acc_api_key}')        
            print(f'Account number: {acc_number}')
            print(f'Account password: {acc_password}')
            
            if input('Is this correct? (y/n): ').lower() == 'y':
                is_good = True
        
        account = AccountDetails(username=acc_username, 
                        password=acc_password, 
                        api_key=acc_api_key, 
                        acc_type=acc_type,
                        acc_number=acc_number)
        self._save_account(account_name=name, account=account, password=password)
        
        self._name = name
        return account
        
               
    def _get_new_password(self):
        print('You need to provide a password for this account. Store this securely!')
        is_correct = False
        while not is_correct:
            password = getpass.getpass('Enter a password: ')
            if self._is_pass_valid(password):
                pass_check = getpass.getpass('Re-enter the password: ')
                if pass_check != password:
                    is_correct = False
                else:
                    is_correct = True
            else: 
                print('Password is invalid pasword!')
        return password
                
    def _is_pass_valid(self, password):
        is_long = len(password) > 10
        has_letters = sum([s in password for s in set(string.ascii_letters)]) > 5
        return is_long and has_letters
     
    def _get_acc_type_(self,):
        is_valid = False
        while not is_valid:
            acc_type = input('Enter the account type (DEMO/LIVE): ')
            if acc_type.upper() != 'DEMO' and acc_type.upper() != 'LIVE':
                print('Invalid account type. Must be one of "DEMO" or "LIVE"')
            else:
                is_valid = True
        return acc_type.upper()

    def _save_account(self, account_name, account, password):
        with shelve.open(Account._account_file_store) as file:
            if account_name not in list(file.keys()):
                # Encrypt account
                b_account = pickle.dumps(account)
                e_account = password_encrypt(b_account, password)
                file[account_name] = e_account
            else:
                raise ValueError(f'Account with name "{account_name}" found!')
        return None
    
    def _load_account(self, account_name):
        with shelve.open(Account._account_file_store) as file:
            if account_name in list(file.keys()):
                e_account = file[account_name]
            else:
                raise ValueError(f'No account with name "{account_name}" found!')
        
        password = getpass.getpass("Enter password: ")
        try:
            b_account = password_decrypt(e_account, password)
        except:
            raise ValueError('Password is incorrect! Please try again')
        account = pickle.loads(b_account)
        self._name = account_name
        return account
    
    def delete_account(self, password):
        try:
            self._load_account(self.name)
        except:
            raise ValueError('Unable to delete account. Account and password do not match!')
        else:
            with shelve.open(Account._account_file_store) as file:
                del file[self.name]
    
    def _choose_account(self):
        with shelve.open(Account._account_file_store) as file:
            keys = list(file.keys())
        print("\n".join([f"{i+1}) {key}" for i, key in enumerate(keys)]))
        i = input("Entry: ")
        return self._load_account(keys[int(i)-1])
        