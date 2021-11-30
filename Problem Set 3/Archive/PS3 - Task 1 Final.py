# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 08:18:49 2021

@author: Cyril
"""


import pandas as pd
import numpy as np
from scipy import stats
import pandas_datareader as web
import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import norm, kurtosis, skew
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression

#Load data
data_beta = pd.read_excel(r"C:\Users\Cyril\.spyder-py3\PS3 - beta.xlsx")
data_FF = pd.read_excel(r"C:\Users\Cyril\.spyder-py3\PS3 - FF.xlsx")

data_beta.set_index('date', drop = True, inplace = True)
data_FF.set_index('date', drop = True, inplace = True)

# a) Report a table of annualized statistics on mean return, volatility and Sharpe ratios of each of the beta-sorted PF


#Calc
rf_adj = data_FF['RF'].to_numpy()
statistics = pd.DataFrame(columns = data_beta.columns, index = ['Annual_Mean_Return', 'Annual_Std', 'Annual_Excess_Return', 'Annual_Excess_Std', "Annual_Sharpe_Ratio"])
for column in data_beta.columns:
    returns_adj = data_beta[column].to_numpy()
    mean_return = np.mean(returns_adj) * 12
    statistics.loc['Annual_Mean_Return', column] = mean_return
    std = 12**0.5 * np.std(returns_adj)
    statistics.loc['Annual_Std', column] = std
    excess_return = np.sum(returns_adj - rf_adj)/56
    statistics.loc['Annual_Excess_Return', column] = excess_return
    excess_std = np.std(returns_adj)*(12**0.5)
    # excess_std = np.std(returns_adj - rf_adj)*(12**0.5)
    statistics.loc['Annual_Excess_Std', column] = excess_std
    sharpe_ratio = excess_return / excess_std
    statistics.loc["Annual_Sharpe_Ratio", column] = sharpe_ratio
            
print(statistics)


# b) CAPM regression for each PF and show: Estimated alpha and beta coefficients with pvalues, also coefficient-of-determination

data_beta_exc = pd.DataFrame(columns = ['beta1', 'beta2', 'beta3', 'beta4', 'beta5',
                            'beta6', 'beta7', 'beta8', 'beta9', 'beta10'], index = data_beta.index)

data_beta_exc['beta1'] = data_beta['beta1'] - data_FF['RF']
data_beta_exc['beta2'] = data_beta['beta2'] - data_FF['RF']
data_beta_exc['beta3'] = data_beta['beta3'] - data_FF['RF']
data_beta_exc['beta4'] = data_beta['beta4'] - data_FF['RF']
data_beta_exc['beta5'] = data_beta['beta5'] - data_FF['RF']
data_beta_exc['beta6'] = data_beta['beta6'] - data_FF['RF']
data_beta_exc['beta7'] = data_beta['beta7'] - data_FF['RF']
data_beta_exc['beta8'] = data_beta['beta8'] - data_FF['RF']
data_beta_exc['beta9'] = data_beta['beta9'] - data_FF['RF']
data_beta_exc['beta10'] = data_beta['beta10'] - data_FF['RF']

CAPM_Stats = pd.DataFrame(columns = data_beta.columns, index =['Alpha', 'P-Value Alpha', 'Beta', 'P-Value Beta', 'Coefficient Of Determination'])
for column in data_beta.columns:
    LR = sm.OLS(data_beta_exc[column].to_numpy(), sm.add_constant(data_FF['Mkt_RF'].to_numpy()))
    LR_results = LR.fit()
    LR_results.params
    CAPM_Stats.loc['Beta', column] = LR_results.params[1]
    CAPM_Stats.loc['Alpha', column] = LR_results.params[0]
    CAPM_Stats.loc['P-Value Beta', column] = LR_results.pvalues[1]
    CAPM_Stats.loc['P-Value Alpha', column] = LR_results.pvalues[0]
    CAPM_Stats.loc['Coefficient Of Determination', column] = LR_results.rsquared
  
   
statistics = pd.concat([statistics, CAPM_Stats])
print(statistics)


# c) Plot the PF's mean returns (y) against estimated betas and indicate the Mkt PF as well as the security market line
statistics_t = np.transpose(statistics)
X = np.array(statistics_t['Beta'], dtype=float)
Y = np.array(statistics_t['Annual_Mean_Return'], dtype=float)
plt.scatter(X,Y)

b, m = np.polyfit(X, Y, 1)

plt.plot(X, Y, '.', color='Blue')
plt.plot(X, b+m * X, '-', color='Black', label = 'Security Market Line')
plt.plot(1.0, b+m, color = 'Red', marker='*', label = 'Market Portfolio')
plt.xlim(xmin=0.7)
plt.legend()
plt.ylim(ymin=0.1, ymax=0.15)

plt.title('Problem 1 - Part 1 - Task c)')
plt.xlabel('Estimated Betas')
plt.ylabel('Annual Mean Returns')
plt.show()

print("The intercept (Risk Free Rate) is " + str(round(b,5)))
print("The slope is " + str(round(m,5)))

# d) Plot the PF's Alphas (y) versus betas (x) 

plt.plot(statistics_t['Beta'], statistics_t['Alpha'], 'ro', color='Blue')

plt.xlabel('Estimated Betas')
plt.ylabel('Estimated Alphas')

plt.title('Problem 1 - Part 1 - Task d)')
plt.show()

# e) Plot the Regression R2 (y) against the PF's betas (x)

plt.plot(statistics_t['Beta'], statistics_t['Coefficient Of Determination'], 'ro', color='Blue')

plt.xlabel('Estimated Betas')
plt.ylabel('Estimated R^2')

plt.title('Problem 1 - Part 1 - Task e)')
plt.show()

#Problem 1, part II
# f) market neutral portfolio, weight of long position, log-price plot
#X = np.array(statistics_t['Beta'], dtype=float)
#print(X)
#print(statistics_t['Beta'].iloc[0])
#print(statistics_t['Beta'].iloc[9])
weight_beta1 = statistics_t['Beta'].iloc[9]/statistics_t['Beta'].iloc[0]
weight_beta10 = -1
print(weight_beta1)
# w = [weight_beta1, -1]
# print(w)
#print(data_beta.head)
# beta_weighted_returns = (w * data_beta[['beta1','beta10']])
ret_mkt_neutral_portfolio = weight_beta1 * data_beta['beta1'] + weight_beta10 * data_beta['beta10']
# ret_mkt_neutral_portfolio = beta_weighted_returns['beta1'] + beta_weighted_returns['beta10']
ret_mkt_neutral_portfolio = pd.DataFrame(ret_mkt_neutral_portfolio)
ret_mkt_neutral_portfolio

ret_mkt_neutral_portfolio['gross_return'] = (ret_mkt_neutral_portfolio + 1)
ret_mkt_neutral_portfolio['gross_return_shifted'] = ret_mkt_neutral_portfolio['gross_return'].shift(fill_value=100)
ret_mkt_neutral_portfolio['price_dev'] = ret_mkt_neutral_portfolio['gross_return_shifted'].cumprod()
ret_mkt_neutral_portfolio['price_dev_log'] = ret_mkt_neutral_portfolio['price_dev'].apply(np.log)
ret_mkt_neutral_portfolio

data_FF['market_return'] = data_FF.Mkt_RF + data_FF.RF
data_FF['gross_return'] = (data_FF['market_return'] + 1)
data_FF['gross_return_shifted'] = data_FF['gross_return'].shift(fill_value=100)
data_FF['price_dev'] = data_FF['gross_return_shifted'].cumprod()
data_FF['price_dev_log'] = data_FF['price_dev'].apply(np.log)
data_FF

plt.plot(ret_mkt_neutral_portfolio['price_dev_log'], label='market neutral portfolio')
plt.plot(data_FF['price_dev_log'], label="market portfolio")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Log Price')
plt.show()

#g) Mean returns, sharpe rations, and correlation
# mean returns
print(f"market neutral portfolio mean return: {round(ret_mkt_neutral_portfolio.iloc[:,0].mean(),4)}")
print(f"market portfolio mean return: {round(data_FF.loc[:,'market_return'].mean(),4)}")

# sharpe ratios
ret_mkt_neutral_portfolio['excess_return'] = ret_mkt_neutral_portfolio.iloc[:,0] - data_FF['RF']
mkt_neutral_sharpe = ret_mkt_neutral_portfolio['excess_return'].mean() / ret_mkt_neutral_portfolio['excess_return'].std()
mkt_sharpe = data_FF['Mkt_RF'].mean() / data_FF['Mkt_RF'].std()
print(f"market neutral portfolio sharpe ratio: {round(mkt_neutral_sharpe,4)}")
print(f"market portfolio sharpe ratio: {round(mkt_sharpe, 4)}")

# correlation of returns
ret_mkt_neutral_portfolio['returns_mkt_neutral'] = ret_mkt_neutral_portfolio.T.head(1).T
correlation_df = pd.DataFrame([ret_mkt_neutral_portfolio['returns_mkt_neutral'],data_FF['market_return']]).T
correlation_df.corr()

#h) CAPM regression, FF 3-factor, FF 5-factor regression

ols = LinearRegression()
y = ret_mkt_neutral_portfolio['excess_return'].to_numpy().reshape(-1,1)
x = data_FF['Mkt_RF'].to_numpy().reshape(-1,1)
ols.fit(x, y)
print(f"intercept: {ols.intercept_[0]}") # intercept
print(f"coefficient: {ols.coef_[0][0]}") # list of feature coefficients

#also using the sm.OLS model to 1) compare the results as well as for an easier output format. 
# sm.OLS(y, X) y:=to be predicted, X:=matrix of features 
#CAPM regression
LR = sm.OLS(
    ret_mkt_neutral_portfolio['excess_return'].to_numpy(), 
    sm.add_constant(data_FF['Mkt_RF'].to_numpy())
)
LR_results = LR.fit()
print(LR_results.summary())

from sklearn.metrics import r2_score
preds = ols.predict(x)
r2_score(y, preds)
# double checking R^2 with a different method

ols = LinearRegression()
y = ret_mkt_neutral_portfolio['excess_return'].to_numpy().reshape(-1,1)
x = data_FF[['Mkt_RF', 'SMB', 'HML']].to_numpy()
ols.fit(x, y)
print(f"intercept: {ols.intercept_[0]}") # intercept
print(f"coefficient: {ols.coef_[0]}") # list of feature coefficients

preds = ols.predict(x)
r2_score(y, preds)

# sm.OLS(y, X) y:=to be predicted, X:=matrix of features
#Fama-French 3-factor regression
LR = sm.OLS(
    ret_mkt_neutral_portfolio['excess_return'].to_numpy(), 
    sm.add_constant(data_FF[['Mkt_RF', 'SMB', 'HML']].to_numpy())
)
LR_results = LR.fit()
print(LR_results.summary())

ols = LinearRegression()
y = ret_mkt_neutral_portfolio['excess_return'].to_numpy().reshape(-1,1)
x = data_FF[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']].to_numpy()
ols.fit(x, y)
print(f"intercept: {ols.intercept_[0]}") # intercept
print(f"coefficient: {ols.coef_[0]}") # list of feature coefficients

preds = ols.predict(x)
r2_score(y, preds)
# correct?

#for ease of output also doing the regression in the other format:
# sm.OLS(y, X) y:=to be predicted, X:=matrix of features
#Fama-French 5-factor regression
LR = sm.OLS(
    ret_mkt_neutral_portfolio['excess_return'].to_numpy(), 
    sm.add_constant(data_FF[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']].to_numpy())
)
LR_results = LR.fit()
print(LR_results.summary())

from statsmodels.stats.outliers_influence import variance_inflation_factor
data_ff1 = data_FF.iloc[60:]
data_ff1
vif_data = pd.DataFrame()
vif_data["feature"] = data_FF.columns[0:5]
vif_data["VIF"] = [variance_inflation_factor(data_ff1.values, i) for i in range(len(data_ff1.columns[:5]))]

vif_data

#i) Trading strategy discussion
data_FF['mkt_rt_var'] = data_FF['market_return'].rolling(window=1*12).var().shift(1)
data_FF['beta1'] = data_beta['beta1']
data_FF['beta10'] = data_beta['beta10']
data_FF['mkt_b1_cov'] = data_FF[['market_return', 'beta1']].rolling(window=5*12).cov().unstack()['market_return']['beta1'].shift(1)
data_FF['mkt_b10_cov'] = data_FF[['market_return', 'beta10']].rolling(window=5*12).cov().unstack()['market_return']['beta10'].shift(1)
data_FF['beta_b1'] = data_FF['mkt_b1_cov'] / data_FF['mkt_rt_var']
data_FF['beta_b10'] = data_FF['mkt_b10_cov'] / data_FF['mkt_rt_var']
data_FF['weight_b1'] = data_FF['beta_b10'] / data_FF['beta_b1']
data_FF

weight_beta10 = -1
data_FF['ret_portfolio'] = data_FF['weight_b1'] * data_beta['beta1'] + weight_beta10 * data_beta['beta10']
data_FF['excess_ret_portfolio'] = data_FF['ret_portfolio'] - data_FF['RF']

ols = LinearRegression()
y = data_FF['excess_ret_portfolio'].iloc[5*12:].to_numpy().reshape(-1,1)
x = data_FF[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']].iloc[5*12:].to_numpy()
ols.fit(x, y)
print(f"intercept: {ols.intercept_[0]}") # intercept
print(f"coefficient: {ols.coef_[0]}") # list of feature coefficients

preds = ols.predict(x)
r2_score(y, preds)


#for ease of readability also doing the regression with the sm.OLS function 
# sm.OLS(y, X) y:=to be predicted, X:=matrix of features 
LR = sm.OLS(
    data_FF['excess_ret_portfolio'].iloc[5*12:].to_numpy().reshape(-1,1), 
    sm.add_constant(data_FF[['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA']].iloc[5*12:].to_numpy())
)
LR_results = LR.fit()
print(LR_results.summary())

trading_returns = pd.DataFrame()
trading_returns['excess_return_mkt_nt1'] = ret_mkt_neutral_portfolio['excess_return']
trading_returns['excess_return_mkt_rebalanced'] = data_FF['excess_ret_portfolio']
trading_returns = trading_returns.iloc[5*12:]

# calculating price dev series
trading_returns['gross_ret_nt1'] = (trading_returns['excess_return_mkt_nt1'] + 1).shift(fill_value=100)
trading_returns['price_dev_nt1'] = trading_returns['gross_ret_nt1'].cumprod()
trading_returns['price_dev_log_nt1'] = trading_returns['price_dev_nt1'].apply(np.log)

trading_returns['gross_ret_rebalanced'] = (trading_returns['excess_return_mkt_rebalanced'] + 1).shift(fill_value=100)
trading_returns['price_dev_rebalanced'] = trading_returns['gross_ret_rebalanced'].cumprod()
trading_returns['price_dev_log_rebalanced'] = trading_returns['price_dev_rebalanced'].apply(np.log)  

plt.plot(trading_returns['price_dev_log_nt1'], label='market neutral portfolio')
plt.plot(trading_returns['price_dev_log_rebalanced'], label="rebalanced market neutral portfolio")
plt.xlabel('Time')
plt.ylabel('Log Price')
plt.legend()
plt.show()
