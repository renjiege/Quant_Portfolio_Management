#!/usr/bin/env python
# coding: utf-8

import os
import args
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta  
from functools import reduce

def line_plot(df, var, title=False):
    fig, ax = plt.subplots(figsize=(10,5), dpi=100)
    plt.style.use('seaborn-ticks') # print(plt.style.available)
    if title:
        plt.title(title, fontsize=22)

    df.reset_index().plot.line(x= 'date', y = var, ax=ax, color="black").grid(alpha=.4)
    ax.get_legend().remove()
#     fig.savefig(_charts + "\\" + segname + "_MON.png", bbox_inches = 'tight', pad_inches = 0)

def get_return(df, var):
    df['G1'] = df[var].pct_change(1)
    df['N1'] = np.log(df[var].pct_change(1)+1)
    
    df['G5'] = df[var].pct_change(5)
    df['N5'] = np.log(df[var].pct_change(5)+1)
    
    df['G1_mon'] = df[var].pct_change(25)
    df['N1_mon'] = np.log(df[var].pct_change(25)+1) 
    
    df['G6_mon'] = df[var].pct_change(125)
    df['N6_mon'] = np.log(df[var].pct_change(125)+1)  
    
    df['G1_yr'] = df[var].pct_change(250)
    df['N1_yr'] = np.log(df[var].pct_change(250)+1)   
    
    df['G2_yr'] = df[var].pct_change(500)
    df['N2_yr'] = np.log(df[var].pct_change(500)+1)
    
#     df['MA2'] = df[var].rolling(2, min_periods=2).mean()
    df['MA4'] = df[var].rolling(4, min_periods=4).mean()
    return df

def get_return_std(df, y):
    df['R1_std'] = df[y].rolling(250, min_periods=250).std()
    df['R2_std'] = df[y].rolling(500, min_periods=500).std()
    return df

def get_holidays(df):
    date = datetime.strptime(df.date.min(), "%Y-%m-%d")
    end_date = datetime.strptime(df.date.max(), "%Y-%m-%d")
    holidays = []
    while date <= end_date:
        if date.weekday()<=4:
            date = date.strftime("%Y-%m-%d")
            if date not in list(df.date):
                holidays += [date]
            date = datetime.strptime(date, "%Y-%m-%d")
        date += timedelta(1)
    return holidays

# --- Feature Engineering: return, s.d. of return, weekday, holiday, etc. ---
def get_state(df):
    df = get_return(df, 'close')
    df = get_return_std(df, 'N1')
    # keep states needed
    df = df[['date', 'close', 'vol'] + list(df.filter(regex='^N|^R|^MA'))]
    return df

def get_training_set(dic):
    train = {}
    for key, value in dic.items():
        train[key] = value.loc[(value.date >= start_date) & (value.date < end_date)].reset_index(drop=True)
        train[key] = get_state(train[key])
        train[key] = train[key].iloc[500:].reset_index(drop=True)
        train[key]['weekday'] = pd.to_datetime(train[key].date).dt.dayofweek
    return train  

# save 500 days to calculate 2-year return
def get_test_set(dic):
    test = {}
    for key, value in dic.items():
        test[key] = value.iloc[int(value.index[value.date==end_date].values - 500):].reset_index(drop=True)
        test[key] = get_state(test[key])
        test[key] = test[key].iloc[500:].reset_index(drop=True)
        test[key]['weekday'] = pd.to_datetime(test[key].date).dt.dayofweek
    return test  

def get_holiday_dummy(dic):
    one_day_before_holiday = list((pd.to_datetime(holidays) - timedelta(1)).strftime("%Y-%m-%d"))
    two_day_before_holiday = list((pd.to_datetime(holidays) - timedelta(2)).strftime("%Y-%m-%d"))
    three_day_before_holiday = list((pd.to_datetime(holidays) - timedelta(3)).strftime("%Y-%m-%d"))
    three_day_after_holiday = list((pd.to_datetime(holidays) + timedelta(3)).strftime("%Y-%m-%d"))
    two_day_after_holiday = list((pd.to_datetime(holidays) + timedelta(2)).strftime("%Y-%m-%d"))
    one_day_after_holiday = list((pd.to_datetime(holidays) + timedelta(1)).strftime("%Y-%m-%d"))

    for key, value in dic.items():
        df = dic[key]
        df['before_holiday_1'] = df.date.apply(lambda x: 1 if (x in one_day_before_holiday) else 0)
        df['before_holiday_2'] = df.date.apply(lambda x: 1 if (x in two_day_before_holiday) else 0)
        df['before_holiday_3'] = df.date.apply(lambda x: 1 if (x in three_day_before_holiday) else 0)
        df['after_holiday_1'] = df.date.apply(lambda x: 1 if (x in one_day_after_holiday) else 0)
        df['after_holiday_2'] = df.date.apply(lambda x: 1 if (x in two_day_after_holiday) else 0)
        df['after_holiday_3'] = df.date.apply(lambda x: 1 if (x in three_day_after_holiday) else 0)
    return dic

# load stocks
data_folder = args._daily
os.chdir(data_folder)
stock = [x.replace('.csv', '') for x in os.listdir(data_folder)]
stock = [x for x in stock if x!='GOOGL' and x!='ALLY' and x!='WAL']
# stock = ['SPX', 'BA', 'MS']
tables = [pd.read_csv(x + ".csv") for x in stock ]
stock_d = dict(zip(stock, tables)) # line_plot(stock_d['SPX'], 'close', 'SPX')

# clean a bit
for each in stock:
    stock_d[each] = stock_d[each].rename(columns={'5. adjusted close': 'close', '6. volume': 'vol'})

# env para
start_date = max([value.date.min() for key, value in stock_d.items()])
end_date = '2018-01-05'
worth = 20000
bottom_line = worth * 0.9
cycle = 60

# get train and test
train = get_training_set(stock_d)
test = get_test_set(stock_d)
holidays = get_holidays(train['SPX']) + get_holidays(test['SPX']) 
train = get_holiday_dummy(train)
test = get_holiday_dummy(test)
train['SPX']['close'] = train['SPX']['close']/10


def add_days(date, days):
    new_datetime = datetime.strptime(date, "%Y-%m-%d") + timedelta(days)
    new_date = new_datetime.strftime("%Y-%m-%d")
    return new_date

def add_one_trade_day(date):
    date = add_days(date, 1)
    while (datetime.strptime(date, "%Y-%m-%d").weekday()>=5) or (date in holidays):
        date = add_days(date, 1) 
    return date

def concat(x,y): return pd.concat([x,y], axis=0).reset_index(drop=True)

def get_info(env, date):
    info_matrix = []
    for key, df in env.items():
        df['stock'] = key
        a = df.loc[df.date == date]
        info_matrix += [a]
    return reduce((lambda x, y: concat(x,y)), info_matrix)

def fraction_power(x, n):
    if 0<=x: return x**(1./n)
    return -(-x)**(1./n)

# this env does not have interest rate or mutual fund, and has no transaction cost
class trade_env():
    def __init__(self, env, worth, cycle, rho):
        self.env = env
        self.cycle = cycle
        self.rho = rho
        self.action_space = len(env.keys()) + 1  # add one to add cash holding
        self.state_space = len(env['SPX'].columns) - 1 # minus one to remove date from info 
        self.end_date = env['SPX'].date.max()
        self.reset(worth)
        
    def step(self, action):
        
        # today
        last_worth = self.worth
        last_SPX_worth = self.SPX_worth
        info = get_info(self.env, self.date)
        
        SPX_price = info.loc[info.stock == 'SPX'].close.item()
        SPX_share = self.SPX_worth // SPX_price
        SPX_changes = self.SPX_worth % SPX_price
        
        self.prices = np.append([1], info.close.to_numpy())  # 1 is the opportunity cost of holding cash
        self.share = self.worth * action // self.prices
        changes = (self.worth * action % self.prices)[1: self.action_space]
        self.share[0] = self.share[0] + np.sum(changes) # changes go to cash, i.e., share[0]
        
        # next day
        
        self.date = add_one_trade_day(self.date)
        new_info = get_info(self.env, self.date)
        
        SPX_new_price = new_info.loc[new_info.stock == 'SPX'].close.item()
        self.SPX_worth = SPX_share * SPX_new_price + SPX_changes
        SPX_reward = self.SPX_worth - last_SPX_worth
        
        new_prices = np.append([1], new_info.close.to_numpy())
        self.worth  = np.sum(self.share * new_prices)
    
        # reality
        state = np.array(new_info.drop(['date', 'stock'],axis=1))
        reward = self.worth - last_worth
        utility = fraction_power(reward, self.rho)
        gone = datetime.strptime(self.date, "%Y-%m-%d") - datetime.strptime(self.initial_date, "%Y-%m-%d") 
        
        if (self.worth < bottom_line) or (gone.days > self.cycle) or (self.date == self.end_date):
            done = True
        else:
            done = False
            
        return state, reward, SPX_reward, utility, done
        
    def reset(self, worth):
        self.worth = worth
        self.SPX_worth = worth
        self.date = random.choice(self.env['SPX'].date)
        self.initial_date = self.date
        initial_info = get_info(self.env, self.initial_date)
        initial_state = np.array(initial_info.drop(['date', 'stock'],axis=1))
        return initial_state