# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- UVXY historical data ----------
# 5 trading days in a week 
# check yearlyreturn leverageToUsed 
# yearlyreturn UVXY 1.58(F30%BA1%) WM 1.135
# leverageToUsed: UVXY 0.936(F30%BA1%) WM 1.577
# yearlyreturn without floor: UVXY1.24(0.005bidask)1.11(1%) HSY1.065 MSFT1.196 
# leverageToUsed without floor: UVXY0.51(0.005 bidask)0.37(1%) HSY1.507 MSFT1.857 

ticker = 'hsy'
Floor1 = 0.8
bidaskspread = 0.00015           # in portion relative to 1
AnnualDividend = 0.0

start = 2 # > 2
# MSFT 1.65% avg, AAPL 1.38%, AMZN 0%, VIG ~2%, SPY ~1.8%, QQQ ~0.6%, KO
# 3%, AMD 0, NVDA 0.1%, BSV 1.6%, HSY 1.9%, TLT 1.6%
Ticker = pd.read_csv(ticker + '.csv')
Open = Ticker.iloc[:,1].values
Close = Ticker.iloc[:,4].values
# bidaskspreadPercent = 0.5
# ----------
leverage = np.linspace(0.01, 3.5, 600)
portion = (Close[-1]-Close[1])/abs(Close[-1]-Close[1])
N = len(Ticker) - start 

leverageUsed = 1.5          # HSY: 3.817
PriceChange = np.zeros(N)
PriceChangeFloored = np.zeros(N)
PriceChangeLeverage = np.zeros(N)
PriceChangeLeverageFloored = np.zeros(N)
PriceChangeAccumulate = np.zeros(N)
PriceChangeAccumulateFloored = np.zeros(N)

for k in range(start, len(Ticker)):
    PriceChange[k-start] = portion*(Close[k]-Close[k-1])/Close[k-1]
    PriceChangeFloored[k-start] = max(portion*(Close[k]-Close[k-1])/Close[k-1], -abs(portion*Floor1))
    PriceChangeLeverage[k-start] = leverageUsed*PriceChange[k-start]
    PriceChangeLeverageFloored[k-start] = leverageUsed*PriceChangeFloored[k-start]
    PriceChangeAccumulate[k-start] = np.prod(1+PriceChangeLeverage[0:k-start+1])
    PriceChangeAccumulateFloored[k-start] = np.prod(1+PriceChangeLeverageFloored[0:k-start+1])

BiggestLoss = np.min(PriceChange)
BiggestGain = np.max(PriceChange)
PriceChangeSum = np.sum(PriceChange)
PriceChangeNet = np.prod(1+PriceChangeLeverage)
PriceChangeNetFloored = np.prod(1+PriceChangeLeverageFloored)
PriceChangeNetDividend = PriceChangeNet*(1+leverageUsed*(AnnualDividend))**((N+1)/52)
PriceChangeNetDividendFloored = PriceChangeNetFloored*(1+leverageUsed*(AnnualDividend))**((N+1)/52)

totalcapital = 1
dailyexpectedreturn = np.zeros(len(leverage))
for l in range(len(leverage)):
    totalcapital = 1
    for k in range(start, start+N-1):
        totalcapital = totalcapital*(1+leverage[l]*PriceChangeFloored[k-1]-leverage[l]*bidaskspread) # Remove Floored if deep in the money options
    if totalcapital > 0:
        dailyexpectedreturn[l] = totalcapital**(1/N/7)*(1+leverage[l]*AnnualDividend)**(1/365)  # include weekend as 2 days
    else: 
        dailyexpectedreturn[l] = 1.0
        
highestreturn, temp = np.max(dailyexpectedreturn), np.argmax(dailyexpectedreturn)     # including weekend
leverageToUsed = leverage[temp]
yearlyreturn = highestreturn**(365) # has bidaskspread already




year = np.arange(start, len(Close))
year = year/51
year = 2023.4-year[-1]+year
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(leverage, np.real(dailyexpectedreturn))
plt.subplot(2, 1, 2)
plt.semilogy(year, Close[start:])
plt.figure(3)
plt.subplot(2, 1, 1)
plt.plot(year, PriceChange)
plt.subplot(2, 1, 2)
plt.semilogy(year, PriceChangeAccumulate)
plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(year, PriceChangeFloored)
plt.title('Floored')
plt.subplot(2, 1, 2)
plt.semilogy(year, PriceChangeAccumulateFloored)

fiveyear = yearlyreturn**5