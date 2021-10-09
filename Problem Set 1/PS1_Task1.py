# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:00:01 2021

@author: Cyril
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#Load data
df = pd.read_csv(r"C:\Users\Cyril\.spyder-py3\ps1_data.csv")
df = df.set_index('Unnamed: 0')

#Calculate discrete returns
disc_returns = df.pct_change()
disc_returns = disc_returns.iloc[1:,:]

#Calculate log returns
df['log_return_Deutsche_Bank'] = np.log(df['DEUTSCHE_BANK'])-np.log(df['DEUTSCHE_BANK'].iloc[0])
df['log_return_Allianz'] = np.log(df['ALLIANZ'])-np.log(df['ALLIANZ'].iloc[0])
df['log_return_BASF'] = np.log(df['BASF'])-np.log(df['BASF'].iloc[0])
df['log_return_SIEMENS'] = np.log(df['SIEMENS'])-np.log(df['SIEMENS'].iloc[0])
df['log_return_BMW'] = np.log(df['BMW'])-np.log(df['BMW'].iloc[0])
df['log_return_BAYER'] = np.log(df['BAYER'])-np.log(df['BAYER'].iloc[0])
df['log_return_E_ON'] = np.log(df['E_ON'])-np.log(df['E_ON'].iloc[0])
df['log_return_RWE'] = np.log(df['RWE'])-np.log(df['RWE'].iloc[0])
df['log_return_HENKEL'] = np.log(df['HENKEL'])-np.log(df['HENKEL'].iloc[0])
df['log_return_SAP'] = np.log(df['SAP'])-np.log(df['SAP'].iloc[0])

df_log = df[df.columns[-10:]]
df_log = df_log.iloc[1:,:]

df_log = df_log.rename(columns={'log_return_Deutsche_Bank':'DEUTSCHE_BANK',
                       'log_return_Allianz':'ALLIANZ',
                      'log_return_BASF':'BASF',
                     'log_return_SIEMENS':'SIEMENS',
                    'log_return_BMW':'BMW',
                    'log_return_BAYER':'BAYER',
                   'log_return_E_ON':'E_ON',
                  'log_return_RWE':'RWE',
                'log_return_HENKEL':'HENKEL',
               'log_return_SAP':'SAP'})


mean_returns = disc_returns.mean()
std_dev = disc_returns.std()

#Annualized mean returns disc
months = 300

total_return_Deutsche_Bank = (df.DEUTSCHE_BANK[-1]-df.DEUTSCHE_BANK[0])/df.DEUTSCHE_BANK[0]
ann_return_Deutsche_Bank = ((1+total_return_Deutsche_Bank)**(12/months))-1

#Annualized mean returns log
total_log_return_Deutsche_Bank = (df_log.log_return_Deutsche_Bank[-1]-df_log.log_return_Deutsche_Bank[0])/df_log.log_return_Deutsche_Bank[0]
ann_log_return_Deutsche_Bank = ((1+total_log_return_Deutsche_Bank)**(12/months))-1

#Max and min values, Difference

diff_disc_log = abs(df_log.subtract(disc_returns))

max_values = []
min_values = []
for i in diff_disc_log:
    j = max(diff_disc_log[i])
    k = min(diff_disc_log[i])
    max_values.append(j)
    min_values.append(k)
#Max is SAP, min is HENKEL

#Plot disc returns of SAP and HENKEL

#SAP log on x-axis and discrete on y-axis
SAP_log = df_log['SAP']
SAP_disc = disc_returns['SAP']

plt.plot(SAP_log, SAP_disc)
plt.show()

    





















