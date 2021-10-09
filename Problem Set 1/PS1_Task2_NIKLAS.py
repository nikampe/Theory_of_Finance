import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#Load data
df = pd.read_csv(r"PS1 - Data.csv")
del df['Unnamed: 0']

#Calculate %returns and delete first nan-row
returns = df.pct_change()
returns = returns.iloc[1:,:]
return_std = (returns.std())
stocks_std = (df.std())

#Sort from High to Low
return_std = return_std.sort_values(ascending=False)

v_return_std = np.array(return_std)

#Portfolio Stocks
df1 = df[['DEUTSCHE_BANK']]
df2 = df[['DEUTSCHE_BANK', "SAP"]]
df3 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ"]]
df4 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS"]]
df5 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW"]]
df6 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE"]]
df7 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER"]]
df8 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER", "BASF"]]
df9 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER", "BASF", "E_ON"]]
df10 = df[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER", "BASF", "E_ON", "HENKEL"]]

#Portfolio Stock Returns
rdf1 = returns[['DEUTSCHE_BANK']]
rdf2 = returns[['DEUTSCHE_BANK', "SAP"]]
rdf3 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ"]]
rdf4 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS"]]
rdf5 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW"]]
rdf6 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE"]]
rdf7 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER"]]
rdf8 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER", "BASF"]]
rdf9 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER", "BASF", "E_ON"]]
rdf10 = returns[['DEUTSCHE_BANK', "SAP", "ALLIANZ", "SIEMENS", "BMW", "RWE", "BAYER", "BASF", "E_ON", "HENKEL"]]

# #Mean Std Deviation Stocks
# std1 = (df1.std())
# std1_mean = std1
# std2 = (df2.std())
# std2_mean = std2.mean()
# std3 = (df3.std())
# std3_mean = std3.mean()
# std4 = (df4.std())
# std4_mean = std4.mean()
# std5 = (df5.std())
# std5_mean = std5.mean()
# std6 = (df6.std())
# std6_mean = std6.mean()
# std7 = (df7.std())
# std7_mean = std7.mean()
# std8 = (df8.std())
# std8_mean = std8.mean()
# std9 = (df9.std())
# std9_mean = std9.mean()
# std10 = (df10.std())
# std10_mean = std10.mean()

#Mean Std Deviation Returns
rstd1 = (rdf1.std())
rstd1_mean = rstd1
rstd2 = (rdf2.std())
rstd2_mean = rstd2.mean()
rstd3 = (rdf3.std())
rstd3_mean = rstd3.mean()
rstd4 = (rdf4.std())
rstd4_mean = rstd4.mean()
rstd5 = (rdf5.std())
rstd5_mean = rstd5.mean()
rstd6 = (rdf6.std())
rstd6_mean = rstd6.mean()
rstd7 = (rdf7.std())
rstd7_mean = rstd7.mean()
rstd8 = (rdf8.std())
rstd8_mean = rstd8.mean()
rstd9 = (rdf9.std())
rstd9_mean = rstd9.mean()
rstd10 = (rdf10.std())
rstd10_mean = rstd10.mean()

# calc covariance matrix 
cov_matrix2 = df2.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix3 = df3.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix4 = df4.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix5 = df5.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix6 = df6.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix7 = df7.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix8 = df8.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix9 = df9.pct_change().apply(lambda x: np.log(1+x)).cov()
cov_matrix10 = df10.pct_change().apply(lambda x: np.log(1+x)).cov()

cov_matrix2_NEW = df2.pct_change().cov()
cov_matrix3_NEW = df3.pct_change().cov()
cov_matrix4_NEW = df4.pct_change().cov()
cov_matrix5_NEW = df5.pct_change().cov()
cov_matrix6_NEW = df6.pct_change().cov()
cov_matrix7_NEW = df7.pct_change().cov()
cov_matrix8_NEW = df8.pct_change().cov()
cov_matrix9_NEW = df9.pct_change().cov()
cov_matrix10_NEW = df10.pct_change().cov()

#PF variance and Std Dev
w2 = {'DEUTSCHE BANK': .50, 'SAP': .50}
w3 = {'DEUTSCHE BANK': 1/3, 'SAP': 1/3, 'ALLIANZ': 1/3}
w4 = {'DEUTSCHE BANK': 1/4, 'SAP': 1/4, 'ALLIANZ': 1/4, 'SIEMENS': 1/4}
w5 = {'DEUTSCHE BANK': 1/5, 'SAP': 1/5, 'ALLIANZ': 1/5, 'SIEMENS': 1/5, 'BMW': 1/5}
w6 = {'DEUTSCHE BANK': 1/6, 'SAP': 1/6, 'ALLIANZ': 1/6, 'SIEMENS': 1/6, 'BMW': 1/6, 'RWE': 1/6}
w7 = {'DEUTSCHE BANK': 1/7, 'SAP': 1/7, 'ALLIANZ': 1/7, 'SIEMENS': 1/7, 'BMW': 1/7, 'RWE': 1/7, 'BAYER': 1/7}
w8 = {'DEUTSCHE BANK': 1/8, 'SAP': 1/8, 'ALLIANZ': 1/8, 'SIEMENS': 1/8, 'BMW': 1/8, 'RWE': 1/8, 'BAYER': 1/8, 'BASF': 1/8}
w9 = {'DEUTSCHE BANK': 1/9, 'SAP': 1/9, 'ALLIANZ': 1/9, 'SIEMENS': 1/9, 'BMW': 1/9, 'RWE': 1/9, 'BAYER': 1/9, 'BASF': 1/9, 'E_ON': 1/9}
w10 = {'DEUTSCHE BANK': 1/10, 'SAP': 1/10, 'ALLIANZ': 1/10, 'SIEMENS': 1/10, 'BMW': 1/10, 'RWE': 1/10, 'BAYER': 1/10, 'BASF': 1/10, 'E_ON': 1/10, 'HENKEL': 1/10}

w2_NEW = np.array([1/2, 1/2])
w3_NEW = np.array([1/3, 1/3, 1/3])
w4_NEW = np.array([1/4, 1/4, 1/4, 1/4])
w5_NEW = np.array([1/5, 1/5, 1/5, 1/5, 1/5])
w6_NEW = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
w7_NEW = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
w8_NEW = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
w9_NEW = np.array([1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9])
w10_NEW = np.array([1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10])

PF2_var = cov_matrix2.mul(w2, axis=0).mul(w2, axis=1).sum().sum()
PF2_Std = np.sqrt(PF2_var)
PF3_var = cov_matrix3.mul(w3, axis=0).mul(w3, axis=1).sum().sum()
PF3_Std = np.sqrt(PF3_var)
PF4_var = cov_matrix4.mul(w4, axis=0).mul(w4, axis=1).sum().sum()
PF4_Std = np.sqrt(PF4_var)
PF5_var = cov_matrix5.mul(w5, axis=0).mul(w5, axis=1).sum().sum()
PF5_Std = np.sqrt(PF5_var)
PF6_var = cov_matrix6.mul(w6, axis=0).mul(w6, axis=1).sum().sum()
PF6_Std = np.sqrt(PF6_var)
PF7_var = cov_matrix7.mul(w7, axis=0).mul(w7, axis=1).sum().sum()
PF7_Std = np.sqrt(PF7_var)
PF8_var = cov_matrix8.mul(w8, axis=0).mul(w8, axis=1).sum().sum()
PF8_Std = np.sqrt(PF8_var)
PF9_var = cov_matrix9.mul(w9, axis=0).mul(w9, axis=1).sum().sum()
PF9_Std = np.sqrt(PF9_var)
PF10_var = cov_matrix10.mul(w10, axis=0).mul(w10, axis=1).sum().sum()
PF10_Std = np.sqrt(PF10_var)

PF2_var_NEW = np.dot(w2_NEW.T, np.dot(cov_matrix2_NEW, w2_NEW))
PF2_Std_NEW = np.sqrt(PF2_var_NEW)
PF3_var_NEW = np.dot(w3_NEW.T, np.dot(cov_matrix3_NEW, w3_NEW))
PF3_Std_NEW = np.sqrt(PF3_var_NEW)
PF4_var_NEW = np.dot(w4_NEW.T, np.dot(cov_matrix4_NEW, w4_NEW))
PF4_Std_NEW = np.sqrt(PF4_var_NEW)
PF5_var_NEW = np.dot(w5_NEW.T, np.dot(cov_matrix5_NEW, w5_NEW))
PF5_Std_NEW = np.sqrt(PF5_var_NEW)
PF6_var_NEW = np.dot(w6_NEW.T, np.dot(cov_matrix6_NEW, w6_NEW))
PF6_Std_NEW = np.sqrt(PF6_var_NEW)
PF7_var_NEW = np.dot(w7_NEW.T, np.dot(cov_matrix7_NEW, w7_NEW))
PF7_Std_NEW = np.sqrt(PF7_var_NEW)
PF8_var_NEW = np.dot(w8_NEW.T, np.dot(cov_matrix8_NEW, w8_NEW))
PF8_Std_NEW = np.sqrt(PF8_var_NEW)
PF9_var_NEW = np.dot(w9_NEW.T, np.dot(cov_matrix9_NEW, w9_NEW))
PF9_Std_NEW = np.sqrt(PF9_var_NEW)
PF10_var_NEW = np.dot(w10_NEW.T, np.dot(cov_matrix10_NEW, w10_NEW))
PF10_Std_NEW = np.sqrt(PF10_var_NEW)

# x_axis = [1,2,3,4,5,6,7,8,9,10]
# y_axis_mean = [std1, std2_mean, std3_mean, std4_mean, std5_mean, std6_mean, std7_mean, std8_mean, std9_mean, std10_mean]
# y_axis_rmean = [rstd1_mean, rstd2_mean, rstd3_mean, rstd4_mean, rstd5_mean, rstd6_mean, rstd7_mean, rstd8_mean, rstd9_mean, rstd10_mean]
# y_axis_PF_std = [std1, PF2_Std, PF3_var, PF4_Std, PF5_Std, PF6_Std, PF7_Std, PF8_Std, PF9_Std, PF10_Std]

x_axis = [1,2,3,4,5,6,7,8,9,10]
# y_axis_mean = [rstd1, rstd2_mean, rstd3_mean, rstd4_mean, rstd5_mean, rstd6_mean, rstd7_mean, rstd8_mean, rstd9_mean, rstd10_mean]
y_axis_rmean = [rstd1_mean, rstd2_mean, rstd3_mean, rstd4_mean, rstd5_mean, rstd6_mean, rstd7_mean, rstd8_mean, rstd9_mean, rstd10_mean]
y_axis_PF_std = [rstd1, PF2_Std_NEW, PF3_Std_NEW, PF4_Std_NEW, PF5_Std_NEW, PF6_Std_NEW, PF7_Std_NEW, PF8_Std_NEW, PF9_Std_NEW, PF10_Std_NEW]


fig2 = plt.plot(x_axis, y_axis_rmean, y_axis_PF_std)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of Stocks')
ax1.set_ylabel('Portfolio Standard Deviation', color=color)
ax1.plot(x_axis, y_axis_PF_std, color = color)
ax1.tick_params(axis = 'y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Mean Standard Deviation', color=color)
ax2.plot(x_axis, y_axis_rmean, color = color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
