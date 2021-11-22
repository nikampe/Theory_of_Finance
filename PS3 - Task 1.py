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


#Load data
data_beta = pd.read_excel(r"C:\Users\Cyril\.spyder-py3\PS3 - beta.xlsx")
data_FF = pd.read_excel(r"C:\Users\Cyril\.spyder-py3\PS3 - FF.xlsx")

data_beta.set_index('date', drop = True, inplace = True)
data_FF.set_index('date', drop = True, inplace = True)

# a) Report a table of annualized statistics on mean return, volatility and Sharpe ratios of each of the beta-sorted PF

#Returns
# mean_returns_beta = data_beta.mean(axis = 0, skipna = True)
# mean_returns_beta_an = mean_returns_beta*12
# print(mean_returns_beta_an)



#Yearly Returns and Std

#Mean
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

CAPM_Stats = pd.DataFrame(columns = data_beta.columns, index =['Alpha', 'Beta', 'P-Value', 'Coefficient Of Determination'])
for column in data_beta.columns:
    LR = sm.OLS(data_beta[column].to_numpy(), sm.add_constant(data_FF['Mkt_RF'].to_numpy()))
    LR_results = LR.fit()
    LR_results.params
    CAPM_Stats.loc['Beta', column] = LR_results.params[1]
    CAPM_Stats.loc['Alpha', column] = LR_results.params[0]
    CAPM_Stats.loc['P-Value', column] = LR_results.pvalues[1]
    CAPM_Stats.loc['Coefficient Of Determination', column] = LR_results.rsquared
    # print(LR_results.summary())
   
statistics = pd.concat([statistics, CAPM_Stats])
print(statistics)

# c) Plot the PF's mean returns (y) against estimated betas and indicate the Mkt PF as well as the security market line
statistics_t = np.transpose(statistics)
X = np.array(statistics_t['Beta'], dtype=float)
Y = np.array(statistics_t['Annual_Mean_Return'], dtype=float)
plt.scatter(X,Y)

b, m = np.polyfit(X, Y, 1)

plt.plot(X, Y, '.')
plt.plot(X, b+m * X, '-', color='Black', label = 'Security Market Line')
plt.xlim(xmin=0.59)
plt.legend()
plt.ylim(ymin=0)

plt.title('Problem 1 - Part 1 - Task c)')
plt.xlabel('Estimated Betas')
plt.ylabel('Annual Mean Returns')
plt.show()


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
