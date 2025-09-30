import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- UVXY historical data ----------
# 5 trading days in a week 
# 2.58:2.27:2.68   aapl, fb, msft Avoid 2000 crash of AAPL
# 1.8 SQQQ, short: put negative to portion
# HSY:MSFT = 0.3773:1-x    np.cos(theta_best)**2, leverage 4.5

#check bestyearlyreturn leverageToUse_thetabest np.cos(theta_best)**2
# yrlyReturn 1.6105674023476897 (0.413MSFT UVXY30%F)
showplots = 1

ticker = 'wm'
ticker2 = 'HSY'
theta = np.linspace(0, np.pi/2, 10)
yearlyreturnOptimize = np.zeros(len(theta))
Floor1 = -0.1          # sign doesn't matter
Floor2 = -0.1
bidaskspread1 = 0.0015          # must be positive
bidaskspread2 = 0.0015
AnnualDividend = 0.0
AnnualDividend2 = 0.000
for m in range(0, len(theta), 1):
    start = 2
    # MSFT 1.65% avg, AAPL 1.38%, AMZN 0%, VIG ~2%, SPY ~1.8%, QQQ ~0.6%, KO
    # 3%, AMD 0, NVDA 0.1%, BSV 1.6%, HSY 1.9%
    Ticker = pd.read_csv(ticker+'.csv')
    Ticker2 = pd.read_csv(ticker2+'.csv')
    commonlength = min([len(Ticker), len(Ticker2)])
    Close1 = Ticker.iloc[-commonlength:, 4].to_numpy()
    Close2 = Ticker2.iloc[-commonlength:, 4].to_numpy()
    BestLeverage1 = (Close1[-1]-Close1[1])/abs(Close1[-1]-Close1[1])*np.cos(theta[m])**2
    BestLeverage2 = (Close2[-1]-Close2[1])/abs(Close2[-1]-Close2[1])*np.sin(theta[m])**2
    NetSum = abs(BestLeverage1) + abs(BestLeverage2)
    # ----------
    leverage = np.linspace(0.0, 5, 800)
    portion = [BestLeverage1/NetSum, BestLeverage2/NetSum]
    bidaskspread = abs(portion[0]*bidaskspread1) + abs(portion[1]*bidaskspread2)
    N = len(Close1) - start 
    leverageUsed = 3.07
    PriceChange = np.zeros(N)
    PriceChangeFloored = np.zeros(N)
    PriceChangeLeverage = np.zeros(N)
    PriceChangeLeverageFloored = np.zeros(N)
    PriceChange1 = np.zeros(N)
    PriceChange2 = np.zeros(N)
    PriceChangeAccumulate = np.zeros(N)
    PriceChangeAccumulateFloored = np.zeros(N)
    for k in range(start, len(Close1)):
        PriceChange[k-start] = (portion[0]*(Close1[k]-Close1[k-1])/Close1[k-1] + portion[1]*(Close2[k]-Close2[k-1])/Close2[k-1])
        PriceChangeFloored[k-start] = (max(portion[0]*(Close1[k]-Close1[k-1])/Close1[k-1], -abs(portion[0]*Floor1)) + max(portion[1]*(Close2[k]-Close2[k-1])/Close2[k-1], -abs(portion[1]*Floor2)))
        PriceChangeLeverage[k-start] = leverageUsed*PriceChange[k-start]
        PriceChangeLeverageFloored[k-start] = leverageUsed*PriceChangeFloored[k-start]
        PriceChange1[k-start] = (Close1[k]-Close1[k-1])/Close1[k-1]
        PriceChange2[k-start] = (Close2[k]-Close2[k-1])/Close2[k-1]
        PriceChangeAccumulate[k-start] = np.prod(1+PriceChangeLeverage[0:k-start+1])
        PriceChangeAccumulateFloored[k-start] = np.prod(1+PriceChangeLeverageFloored[0:k-start+1])
    
    BiggestLoss = np.min(PriceChange)
    BiggestGain = np.max(PriceChange)
    PriceChangeSum = np.sum(PriceChange)
    PriceChangeNet = np.prod(1+PriceChangeLeverage)
    PriceChangeNetFloored = np.prod(1+PriceChangeLeverageFloored)
    PriceChangeNetDividend = PriceChangeNet*(1+leverageUsed*(portion[0]*AnnualDividend+ portion[1]*AnnualDividend2))**((N+1)/52)
    PriceChangeNetDividendFloored = PriceChangeNetFloored*(1+leverageUsed*(portion[0]*AnnualDividend+ portion[1]*AnnualDividend2))**((N+1)/52)
    
    totalcapital = 1
    dailyexpectedreturn = np.zeros((len(leverage),))
    for l in range(len(leverage)):
        totalcapital = 1
        for k in range(start, start+N-1):
            totalcapital = totalcapital*(1+leverage[l]*PriceChangeFloored[k-1]-leverage[l]*bidaskspread)
        if totalcapital > 0:
            dailyexpectedreturn[l] = totalcapital**(1/(N+1)/7)*(1+leverage[l]*(portion[0]*AnnualDividend+portion[1]*AnnualDividend2))**(1/365)
        else: 
            dailyexpectedreturn[l] = 1.0
    
    for ii in range(2,len(leverage)):
        if np.real(dailyexpectedreturn[ii]) < np.real(dailyexpectedreturn[ii-1]) and np.real(dailyexpectedreturn[ii-1]) > np.real(dailyexpectedreturn[ii-2]):
            temp = ii
            break
        
    if showplots == 1:        
        fig, axs = plt.subplots(2, 1, figsize=(8,8))
        axs[0].plot(leverage, np.real(dailyexpectedreturn))
        axs[1].plot(np.linspace(1,len(Close1), len(Close1)-start), Close1[start:], np.linspace(1,len(Close1), len(Close1)-start), Close2[start:])
        fig.tight_layout()
    
    highestreturn = dailyexpectedreturn[temp]
    leverageToUse = leverage[temp]
    yearlyreturn = highestreturn**(365)
    
    
    
    year = np.arange(start, len(Close1))
    year = year/52
    year = 2021.4-year[-1]+year
    
    fiveyear = yearlyreturn**5
    NetReturn = yearlyreturn**((N+1)/52)
    Error = NetReturn/PriceChangeNetDividendFloored
    
    yearlyreturnOptimize[m] = yearlyreturn
    
    if m>1.9 and yearlyreturnOptimize[m] < yearlyreturnOptimize[m-1] and yearlyreturnOptimize[m-1] > yearlyreturnOptimize[m-2]:
        theta_idx = m
        fig, axs = plt.subplots(2, 1, figsize=(8,8))
        axs[0].plot(leverage, np.real(dailyexpectedreturn))
        axs[1].plot(year, Close1[start:], year, Close2[start:])
        fig.tight_layout()
        
        #fig, axs = plt.subplots(2, 1, figsize=(8,8))
        #axs[0].plot(PriceChange)
        #axs[1].semilogy(PriceChangeAccumulate)
        #fig.tight_layout()
        
        fig, axs = plt.subplots(2, 1, figsize=(8,8))
        axs[0].plot(year, PriceChange)
        axs[1].semilogy(year, PriceChangeAccumulate)
        fig.tight_layout()
        
        fig, axs = plt.subplots(2, 1, figsize=(8,8))
        axs[0].plot(year, PriceChangeFloored)
        axs[0].set_title('Floored')
        axs[1].semilogy(year, PriceChangeAccumulateFloored, year, PriceChangeAccumulate)
        axs[1].set_ylim([1, None])
        fig.tight_layout()
        
        theta_best = theta[m]
        theta_best_angle = theta[m]*180/np.pi
        bestyearlyreturn = np.nanmax(dailyexpectedreturn)**365
        temp = np.argmax(dailyexpectedreturn)
        leverageToUse_thetabest = leverage[temp]

plt.figure(40)
plt.plot(theta*180/np.pi, yearlyreturnOptimize)
plt.show()

# parameters to check
# leverageToUse_thetabest   bestyearlyreturn
# theta_best_angle      np.sin(theta_best)**2  = portion in stock 1

mycapital = 10000
price1 = 3.55
price2 = 270
cash1 = leverageToUse_thetabest*np.cos(theta_best)**2*mycapital
cash2 = leverageToUse_thetabest*np.sin(theta_best)**2*mycapital
share1 = leverageToUse_thetabest*np.cos(theta_best)**2*mycapital/price1
share2 = leverageToUse_thetabest*np.sin(theta_best)**2*mycapital/price2