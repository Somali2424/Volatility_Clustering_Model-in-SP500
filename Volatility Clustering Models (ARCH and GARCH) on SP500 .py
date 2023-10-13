#!/usr/bin/env python
# coding: utf-8

# Importing the Necessary Libraries

# In[47]:


pip install fredapi


# In[4]:


import warnings
warnings.filterwarnings('ignore')

#Data manipulation and analysis libraries
import numpy as np
import pandas as pd

#Data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Time series analysis libraries
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

#Evaluation metrics libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error

#FredAPI to fetch data
from fredapi import Fred
fredkey="d1cb2348d40ceaae24b7f6d6b07274f4"
fred=Fred(api_key=fredkey)


# Fetching the Data

# In[5]:


sp_search=fred.search("S&P",order_by='popularity')
sp_search
sp500=fred.get_series(series_id='SP500')
sp500


# In[6]:


sp500.info()


# In[8]:


#Checking for null values
null_values=sp500.isnull().sum()
print(null_values)
sp500.plot(figsize=(10,5),title="S&P 500",lw=2)
# The plot looks broken because of the null values near 2021 and 2022


# In[9]:


sp500=sp500.dropna()
sp500
# Removing the null values have reduced the length of the dataset from 2610 t0 2518 
#92 dates corresponded to null values


# In[10]:


sp500.plot(figsize=(10,5),title="S&P 500",lw=2, color='Green')
#The broken parts of the plots have been removes


# In[11]:


sp500.index


# In[12]:


sp500=sp500.to_frame(name='closing value')


# In[13]:


sp500.index.name='Date'


# In[14]:


sp500


# In[15]:


sp500.plot()
#Another Plot with name and without the null value breaks


# In[16]:


sp500


# a Kernel Density Estimation (KDE) graph that estimates the 
# probability density function of the data, providing insight into the distribution and 
# shape of the data. This KDE chart allows you to visualize the distribution of closing 
# prices. The resulting curve provides an estimate of the probability density function, 
# with higher peaks indicating areas of higher density and lower troughs indicating 
# areas of lower density. The shading below the curve provides a visual representation 
# of the estimated probability density function 

# In[17]:


sns.kdeplot(sp500,shade=True)
plt.show()


# KDE chart corresponds to the mode of the distribution that represents the most 
# frequent closing price. This can be useful for traders in identifying potential support 
# or resistance levels - from this point of view we can identify four support levels, the 
# main one being around 2000. It is also important to note that the KDE chart can help 
# identify potential outliers in the closing prices. Outliers are data points that deviate 
# significantly from the overall distribution pattern. These outliers may represent 
# important events or anomalies that have affected the closing prices - of which we do 
# not observe any 

# In[18]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[19]:


#Identify which process it follows AR,MA or ARMA
def data_generating_process(data):
    # Plot ACF (AutoCorrelation Function)
    plt.figure(figsize=(12, 6))
    plot_acf(data)  # Adjust the number of lags as needed
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()

# Plot PACF (Partial AutoCorrelation Function)
    plt.figure(figsize=(12, 6))
    plot_pacf(data)  # Adjust the number of lags as needed
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.xlabel('Lag')
    plt.ylabel('PACF')
    plt.show()



# In[20]:


data_generating_process(sp500)


# FROM THE ACF and PACF above it is understandable that the model follows AR process with 1 lag in PACF.
# Also  we need to check if the model is random walk
# yₜ = ρyₜ₋₁ + uₜ

# The null hypothesis says that the model has unit roots and p value is 1

# In[22]:


pip install statsmodels


# In[23]:


#Test for stationarity with the Dickey Fuller test
print(f'Results of Dickey-Fuller Test:')
adft=adfuller(sp500)
output=pd.Series(adft[0:4],index=['Test Statistics','p value','No of lags used','No of observation used'])
for key,value in adft[4].items():
    output['Critical value(%s)'%key]=value
print(output)


# In[24]:


#If null hypothesis is rejected, then Test 
#statistic < Critical Value and p-value < 0.05, the time series is stationary


# Test statistic is -0.69 which is greater than the critical value and p-value is not less than 0.05. So we cannot reject the null hypothesis

# we can conclude that the time 
# series is not stationary, since we have not rejected the null hypothesis of nonstationarity. This may indicate the presence of a trend, seasonal fluctuations or other 
# systematic changes in the data.

# In[25]:


#Apply Seasonal decomposition
seasonally_decomposed=seasonal_decompose(sp500,model='multiplicative',period=365)

#Visualization of the decomposition results
with plt.rc_context({'figure.figsize':(18,8)}):
    fig=seasonally_decomposed.plot()

#Differentiation to achieve stationarity
sp500_diff=sp500.diff(10).dropna()


# A rolling average is calculated by taking data from the previous 12 months 
# and calculating an average consumption value at each subsequent point in the series. 
# The following code is used to smooth the time series and analyze its variability. The 
# logarithmic transformation helps to reduce non-stationarity and smooth fluctuations 
# in the data, and calculating the moving average and standard deviation allows you 
# to assess the overall trend and variability of the series

# In[26]:


plt.figure(figsize=(10,6))
#Apply a logarithmic transformation of the series
ln_sp500=np.log(sp500)
#Calculating again the rolling moving average and the standard deviation
moving_average_new=ln_sp500.rolling(window=12).mean()
standard_deviation=ln_sp500.rolling(window=12).std()

#Plot moving average and standard deviation charts
plt.plot(standard_deviation,label='Standard Deviation of the Logarithmic Series')
plt.plot(moving_average_new,label='Moving Average of the Logarithmic Series')

#Chart Legend and Title Settings
plt.legend(loc='best')
plt.title("Moving Average and Standard Deviation of the Logarithmic Series")

plt.show()


# Trend removal is performed by 
# subtracting the moving average from the original time series. This highlights shorterterm fluctuations, such as business cycles and seasonal patterns, and makes it easier 
# to analyze these components

# In[27]:


#Subtracting the moving average to detrend the time series
sp500_dtrend=ln_sp500-moving_average_new

#Removing any null values from the resulti
sp500_dtrend.dropna()


# In[28]:


sp500_dtrend2=sp500-moving_average_new
sp500_dtrend2.dropna(inplace=True)


# In[29]:


print(f'Results of Dickey-Fuller Test for ln_sp500:')
adft=adfuller(sp500_dtrend2,autolag='AIC')
output=pd.Series(adft[0:4],index=['Test Statistics','p value','No of lags used','No of observation used'])
for key,value in adft[4].items():
    output['Critical value(%s)'%key]=value
print(output)


# In[43]:


adf_test = adfuller(sp500)
print(f'p-value = {adf_test[1]}')


# #As AR
# model cannot capture this type of behaviour of a time series, we need a better model
# for analysing a time series with non-constant volatility. The importance of forecast and analysis of the errors in econometric model has been increasing with the
# increasing importance of risk and uncertainty in economic theory. Because even after taking the differences(detrending the data removing the moving average from it) the p value is not becoming significant...Basically the behaviour of the stock market follows random walk process...It cannot be modelled to forecast anything at all, since it follows a stochastic trend that does not revolve around the mean

# In[33]:


sp500_returns= sp500.pct_change() * 100


# To model the S&P 500, we will need to detrend the series. The easiest way to do this is to compute the percentage change in the daily closing prices, and obtain the daily returns for the S&P 500

# In[44]:


sp500_returns=sp500_returns.dropna()


# In[45]:


plt.plot(sp500_returns)


# The daily returns data exhibits stationarity, implying a mean-reverting behavior. Such a series is characterized by having an average of zero and varying volatility, indicating alternating periods of low and high volatility. As evident from the plot, we can observe that high spikes in returns tend to be followed by more high spikes, while low spikes are succeeded by more low spikes. These spikes denote volatility, and the proximity of low and high spikes suggests the presence of volatility clustering in the data. This property is valuable for risk forecasting.
# 
# Modelling Volatility: ARCH and GARCH Models

# In[35]:


pip install arch


# In[36]:


import arch


# In[37]:


import statsmodels.api as sm


# In[38]:


# Testing for the ARCH effect
model = arch.arch_model(sp500_returns, vol = 'GARCH', p = 1, q = 1, rescale = False)
results = model.fit(disp='off',show_warning = False)
residuals = results.resid
squared_residuals = residuals**2
arch_test = sm.stats.diagnostic.het_arch(squared_residuals)
print(f'ARCH test results:\n')
print(f'LM Statistic: {arch_test[0]}')
print(f'p-value: {arch_test[1]}')
print(f'F Statistic: {arch_test[2]}')
print(f'p-value: {arch_test[3]}')


# Both the LM and F tests have p-values of  1.7681340869059653e-142 and 2.124147319274508e-167, respectively. Both values are in a scientific notation and represent very small numbers that are way below the 0.05 threshold, which suggests that there is an ARCH effect to the data, and we can use these models to forecast volatility.

# In[39]:


# Fitting ARCH(1) model
model = arch.arch_model(sp500_returns,
                       vol = 'ARCH',
                       p = 1, # One lag of the squared residuals
                       rescale = False)
results = model.fit(disp = 'off', show_warning = False)
print(results.summary()) # Plotting summary


# The summary above shows us the results of fitting the ARCH(1) model to the data. The estimated coefficient for the mean is 0.0967 and it’s considered statistically significant with a p-value below 0.05.
# 
# The Volatility Model estimates the parameters of the ARCH(1) model, where omega is constant at 0.6124 , and alpha[1] is the coefficient for the lagged squared residuals at 0.5420. Both are statistically significant with their p-values below the threshold of 0.05.

# These results give us statistical evidence that the ARCH(1) model is suitable for modelling the volatility of returns for the S&P 500

# In[40]:


# Fitting ARCH(2) model
model = arch.arch_model(sp500_returns,
                       vol = 'ARCH',
                       p = 2, # Two lags of the squared residuals
                       rescale = False)
results = model.fit(disp = 'off', show_warning = False)
print(results.summary())


# These results give us statistical evidence that the ARCH(2) model is also suitable for modelling the volatility of returns for the S&P 500

# In[41]:


# Fitting GARCH(1,1) model
model = arch.arch_model(sp500_returns,
                       vol = 'GARCH', # GARCH model 
                       p = 1,
                       q = 1)
results = model.fit(disp = 'off', show_warning = False)
print(results.summary())


# The GARCH(1,1) model is an extension of the ARCH model, which takes into account the persistence of volatility over time. It assumes that the variance of the error term at a given time t is a function of the past squared error terms and the past conditional variances. It is given by the following equation:
# 
# σ²ₜ=ω+α₁ϵ²ₜ ₋ ₁+β₁ϵ²ₜ ₋ ₂
# 
# Where σ²ₜ represents the conditional variance of the error term at a given time t, ω is a constant representing the unconditional variance, α₁ and β₁ are coefficients of the squared error terms at time t — 1, respectively, and ϵ²ₜ ₋ ₁ is the squared error term at time t — 1.
# 
# We can also fit the GARCH(1,1) using the arch library, by passing GARCH in the vol parameter and defining p=1 and q=1.

# The Volatility Model estimates the parameters of the GARCH(1,1) model, where omega is constant at 0.0379 , and alpha[1] and beta[1] are the coefficients for the lagged squared residuals at 0.1960 and 0.0970, respectively. All parameters are statistically significant with their p-values below the threshold of 0.05.

# To decide which of the three models above are the best fit to the data, we must look at the AIC and the BIC values. These are both statistical metrics that can be used to compare performance of different models. The AIC (Akaike information criterion) estimates the amount of information lost by a model, and it’s given by the following equation:
# 
# AIC=2k−2ln(L)
# 
# Where k represents the number of parameters in the model, and L is the maximum likelihood estimate of the likelihood function for the model. Considering that the AIC estimates the amount of information lost, we can conclude the lower its value, the better the model is.
# 
# The BIC (Bayesian information criterion) is similar to the AIC, but has a different penalty term. The equation for the BIC is as follows:
# 
# BIC=kln(n)−2ln(L)
# 
# Where L is the maximum likelihood function of the model, k is the number of parameters, and n is the sample size. Once again, the model with the lowest BIC value is preferred.

# By observing the summaries above, it’s clear that the Garch(1,1) model has the best fit to the data,

# In[ ]:


#The standard errors are very small so we can proceed with this GARCH (1,1) model.. So the Model to be fitted looks like  


# The Estimated Equation we found the model to capture the volatility is σ²ₜ=0.0379 +0.3630ϵ²ₜ ₋ ₁+0.3630ϵ²ₜ ₋ ₂
# 

# In[ ]:




