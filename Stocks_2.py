#!/usr/bin/env python
# coding: utf-8

# In[318]:


############################################################################################################################
#### **Disclaimer:** This analysis is solely for educational purposes and should not be taken as financial advice ##########
#### or a recommendation to invest in any of the mentioned stocks. The information provided is based on historical data ####
#### and should not be used as a basis for making investment decisions. Always consult with a qualified financial advisor ##
#### before making any investment choices. #################################################################################
############################################################################################################################


# In[233]:


#####################################################################################################
######################### BANKS STOCK DATA SET ######################################################
#####################################################################################################


# In[234]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[235]:


import yfinance as yf

# Define the ticker symbols
stocks = ['BAC', 'C','GS','JPM','MS','WFC']

# Download stock data

for i in stocks:
    globals()[i] = yf.download(i, start='2006-01-01', end='2016-01-01')
    


# In[236]:


###################################################################
####################### Part II - EDA
###################################################################


BAC.head()


# In[237]:


closing_df = yf.download(stocks,start='2006-01-01', end='2016-01-01')['Adj Close']


# In[238]:


closing_df.head()                 #### we are more interested in closing so thats what we will pay attention to most, hence the new df


# In[239]:


return_df = closing_df.pct_change().fillna(0)


# In[240]:


return_df                    #### basically this is returns we get after investing, the first row will be 0 for obvious reasons


# In[241]:


return_df.C.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')         #### interestingly we see massive peak and plummet, lets investigate that date


# In[242]:


return_df.BAC.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')      #### similar crash we see here for Bank of America


# In[243]:


return_df.GS.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='green')       #### similar pattern


# In[244]:


return_df.JPM.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='white')               #### same pattern here as well


# In[245]:


return_df.MS.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='purple')          #### we see a massive massive peak before the crash on Morgan Stanley, interesting


# In[246]:


return_df.WFC.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='brown')        #### similar pattern here


# In[247]:


return_df.BAC.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black') 
return_df.C.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black') 
return_df.GS.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black') 
return_df.JPM.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black') 
return_df.MS.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black') 
return_df.WFC.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black')

#### from this it seems they all are in sync but Citi bank having the worst downfall here then any other banks


# In[248]:


return_df.head()


# In[249]:


return_df[return_df.C == return_df.C.min()]                     #### seems like this the infamous crash of 2008 which affected banks massively

#### The crash you are referring to is the 2008 financial crisis, which had significant impacts in 2009. 
#### It was triggered by the collapse of the housing bubble in the United States, leading to a severe banking crisis. 
#### we will looking how it affected these banks involved in our df


# In[250]:


return_df.loc['2009-01-01':'2009-02-27']


# In[251]:


return_df[return_df.C == return_df.C.max()]


# In[252]:


return_df.loc['2008-09-01':'2009-04-27'].C.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)

plt.axvline(x='2009-02-27',linewidth=3,color='red',linestyle='dashed')

plt.axvline(x='2008-11-24',linewidth=3,color='red',linestyle='dashed')

#### this is very interesting because we see a massive peak so if anybody who sold their shares at this point were the smartest or 
#### if anybody shorted here were the biggest profit gainer because it will never recover from this high
#### and anybody who bought stocks on 2008-11-24 hoping it will go higher made a massive mistake


# In[253]:


return_df.loc['2008-09-01':'2009-04-27'].BAC.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')
return_df.loc['2008-09-01':'2009-04-27'].C.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')
return_df.loc['2008-09-01':'2009-04-27'].GS.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')
return_df.loc['2008-09-01':'2009-04-27'].JPM.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')
return_df.loc['2008-09-01':'2009-04-27'].MS.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')
return_df.loc['2008-09-01':'2009-04-27'].WFC.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')

#### most of them are in sync but City bank having a very hard time here


# In[254]:


closing_df.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red')

#### we see a massive dip from Citi bank and then it never recovered while Goldman Sach did recover very quickly


# In[255]:


closing_df.loc['2008-09-01':'2009-04-27'].plot(legend=True,figsize=(20,7),marker='o',markersize=10)

#### more closer look at the dates which is most crucial to understand the financial crisis, Citi bank having the worst time


# In[256]:


closing_df['C'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10,linestyle='dashed',linewidth=2)

plt.title('Citi Stock Close Graph')

plt.xlabel('Date')

plt.ylabel('Adj Close')

#### obviously we see the massive plummet and it never recovered


# In[257]:


closing_df['GS'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10,linestyle='dashed',linewidth=2)

plt.title('Goldman Sachs Stock Close Graph')

plt.xlabel('Date')

plt.ylabel('Adj Close')

#### it did plummet but recovered almost to the same point as before which is just amazing


# In[258]:


g = sns.jointplot(x='C',y='GS',data=closing_df,kind='kde',fill=True,color='brown')

g.fig.set_size_inches(17,9)

#### interesting but not suprising


# In[259]:


g = sns.jointplot(x='C',y='GS',data=return_df,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

#### definately correlated which is not suprising at all because it was the financial crisis so it was following a pattern


# In[260]:


from scipy.stats import pearsonr                  #### lets see this with pearsonr


# In[261]:


co_eff, p_value = pearsonr(return_df.GS,return_df.C)


# In[262]:


co_eff                               #### correlated


# In[263]:


p_value                              #### just like we had thought, doesn't suprise us at all


# In[264]:


avg = return_df[return_df.columns].mean()

avg = pd.DataFrame(avg)

avg.columns = ['avg']

avg

#### again Citi bank having the worst average so in short anybody who invested in their stocks had the worst downfall
#### but if you were shorting Citi then you made huge profits no doubt


# In[265]:


avg.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',linestyle='dashed',linewidth=4,markersize=20)

plt.scatter(avg.index[0], avg.iloc[0], color='white', s=150, zorder=5)

plt.scatter(avg.index[1], avg.iloc[1], color='red', s=150, zorder=5)

plt.scatter(avg.index[2], avg.iloc[2], color='green', s=150, zorder=5)

plt.scatter(avg.index[3], avg.iloc[3], color='blue', s=150, zorder=5)

plt.scatter(avg.index[4], avg.iloc[4], color='brown', s=150, zorder=5)

plt.scatter(avg.index[5], avg.iloc[5], color='pink', s=150, zorder=5)

plt.title('Stock Mean Profit Graph')

plt.xlabel('Banks')

plt.ylabel('Density Mean')

#### the best mean being for Wells Fargo, wasn't expecting this honestly


# In[266]:


std = return_df[return_df.columns].std()

std = pd.DataFrame(std)

std.columns = ['std']

std


# In[267]:


std.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',color='red',linestyle='dashed',linewidth=4,markersize=20)

plt.scatter(std.index[0], std.iloc[0], color='white', s=150, zorder=5)

plt.scatter(std.index[1], std.iloc[1], color='red', s=150, zorder=5)

plt.scatter(std.index[2], std.iloc[2], color='green', s=150, zorder=5)

plt.scatter(std.index[3], std.iloc[3], color='blue', s=150, zorder=5)

plt.scatter(std.index[4], std.iloc[4], color='brown', s=150, zorder=5)

plt.scatter(std.index[5], std.iloc[5], color='pink', s=150, zorder=5)

plt.title('Stock STD Graph')

plt.xlabel('Banks')

plt.ylabel('Density STD')

#### the safest being Goldman Sachs investment and most risky being Citi


# In[268]:


g = sns.lmplot(x='WFC',y='C',data=return_df,scatter_kws={'color':'black'},line_kws={'color':'red'})

g.fig.set_size_inches(17,9)

#### definately again highly correlated which again is not suprising


# In[269]:


corr = return_df.corr()

corr


# In[270]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### eveything is so highly correlated to each other which does makes sense because financial crisis of 2008 affected every banks from our df


# In[271]:


g = sns.lmplot(x='WFC',y='JPM',data=return_df,scatter_kws={'color':'black'},line_kws={'color':'red'})

g.fig.set_size_inches(17,9)

#### this is just beautiful to look at, love stats


# In[272]:


return_df['BAC'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='blue',color='black',markersize=10)

return_df['C'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=10)

return_df['GS'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='white',color='black',markersize=10)

return_df['JPM'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='brown',color='black',markersize=10)

return_df['MS'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='green',color='black',markersize=10)

return_df['WFC'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='pink',color='black',markersize=10)

plt.title('Return Profit Graph')

plt.xlabel('Date')

plt.ylabel('Profit Density')

#### well most of them are in sync hence the high correlation


# In[273]:


corr = closing_df.corr()

corr


# In[274]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### we see again here that JPM and WFC is highly highly highly correlated, such data is very extremely rare to find.
#### its almost like 1 correlation between them


# In[275]:


g = sns.lmplot(x='WFC',y='JPM',data=closing_df,scatter_kws={'color':'black'},line_kws={'color':'red'})

g.fig.set_size_inches(17,9)

#### i can't emphasize how rare such data is to come by, this is almost perfection correlation


# In[276]:


df = return_df.reset_index()

df


# In[277]:


df.info()


# In[278]:


closing_df


# In[279]:


df['month'] = df.Date.apply(lambda x:x.month)

df['month_name'] = df.month.map({1:'Jan',
                         2:'Feb',
                         3:'Mar',
                         4:'Apr',
                         5:'May',
                         6:'Jun',
                         7:'Jul',
                         8:'Aug',
                         9:'Sep',
                         10:'Oct',
                         11:'Nov',
                         12:'Dec'})

df['day_of_week'] = df.Date.apply(lambda x:x.dayofweek)

df['Day'] = df.day_of_week.map({0:'Mon',
                                     1:'Tue',
                                     2:'Wed',
                                     3:'Thr',
                                     4:'Fri',
                                     5:'Sat',
                                     6:'Sun'})

df['day'] = df.Date.apply(lambda x:x.day)

df


# In[280]:


df['year'] = df.Date.apply(lambda x:x.year)

df.head()


# In[281]:


heat = df.groupby(['year','month_name','day'])['C'].sum().unstack().unstack().fillna(0)

heat             #### now we have all the information to see what really happened with Citi bank


# In[282]:


fig, ax = plt.subplots(figsize=(90,45))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### this gives us a very detailed information about Citi bank stocks


# In[283]:


heat['sum'] = heat.sum(axis=1)             #### we wanna see which year was most profitable

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[284]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### you will see that years 2008,2007,2011,2015 are all in negatives meaning people lost money in stocks those years, major loss being in years 2008 and 2007 which are both related to financial crisis
#### unless they were shorting the stocks for Citi bank


# In[285]:


heat = df.groupby(['year','month_name'])['C'].sum().unstack().fillna(0)

heat


# In[286]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### Feb of 2009 of Citi bank being the worst month and year for it
#### but in reality year 2008 had the negative gains in almsot all months compared to any other years months


# In[287]:


heat['sum'] = heat.sum(axis=1)             #### this will be interesting to see

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[288]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### again we see the worst being 2008 with massive negative gains or in short loss


# In[289]:


heat = df.groupby(['year','Day','month_name'])['C'].sum().unstack().fillna(0)

heat


# In[290]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### Friday of Feb 2009 being the worst day for Citi bank


# In[291]:


heat['sum'] = heat.sum(axis=1)             #### this will be interesting to see

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[292]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### 2008 of every month of year for Citi bank being the worst day in general to invest in that stock
#### 2009 of Friday similar thing happened


# In[293]:


heat = df.groupby(['year','month_name','day'])['BAC'].sum().unstack().unstack().fillna(0)

heat             #### now we have all the information to see what really happened with Bank of America


# In[294]:


fig, ax = plt.subplots(figsize=(90,45))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### this gives us a very detailed information about Bank of America stocks


# In[295]:


heat['sum'] = heat.sum(axis=1)             #### we wanna see which year was most profitable

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[296]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### interestingly worst year for Bank of America was 2011 then followed by financial crisis year 2008

######################       Details for 2011 worst year for BAC    ########################################################

#### Bank of America (BAC) faced significant challenges that led to a particularly tough year, even compared to
#### the 2008 financial crisis. The issues stemmed from the ongoing fallout of the financial crisis, including legal 
#### troubles and settlements related to the acquisition of Countrywide Financial and its subprime mortgage portfolio. 
#### Additionally, the stock was heavily impacted by broader market fears regarding the European debt crisis and the 
#### overall economic uncertainty, leading to a steep decline in share prices.


# In[297]:


heat = df.groupby(['year','month_name'])['BAC'].sum().unstack().fillna(0)

heat


# In[298]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### Jan of 2009 of Bank of America being the worst month


# In[299]:


heat['sum'] = heat.sum(axis=1)             #### this will be interesting to see

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[300]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### again we see the worst being 2011 instead of 2007 for BAC with massive negative gains


# In[301]:


heat = df.groupby(['year','Day','month_name'])['BAC'].sum().unstack().fillna(0)

heat


# In[302]:


fig, ax = plt.subplots(figsize=(26,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### June Thursday of 2009 being the worst day for BAC


# In[303]:


heat['sum'] = heat.sum(axis=1)             #### this will be interesting to see

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[304]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis',annot=True)

#### 2008 of every month of year for BAC being the worst day in general to invest in that stock
#### followed by 2009 and 2011


# In[305]:


df.head()


# In[306]:


heat = df.groupby(['year','day','month_name'])['GS'].sum().unstack().fillna(0)

heat             #### lets see now Goldman Sachs


# In[307]:


fig, ax = plt.subplots(figsize=(70,30))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')

#### September of 2008 being the worst for Goldman Sachs


# In[308]:


heat['sum'] = heat.sum(axis=1)          

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2


# In[309]:


fig, ax = plt.subplots(figsize=(70,30))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### 20th of April of 2009 being the worst for Goldman Sachs
#### followed by 2008 7th,10th,9th and 15th of April


# In[310]:


heat = df.groupby(['year','day','month_name'])['JPM'].sum().unstack().fillna(0)

heat['sum'] = heat.sum(axis=1)          

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2             #### similar treatment to JPM

#### seems like 2009 was bad for JP Morgan Chase


# In[311]:


fig, ax = plt.subplots(figsize=(70,30))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### same here 20th april of 2009 being the worst trading day for JP Morgan Chase


# In[312]:


heat = df.groupby(['year','day','month_name'])['MS'].sum().unstack().fillna(0)

heat['sum'] = heat.sum(axis=1)          

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2             #### similar treatment to Morgan Stanley

#### seems like 2008 was bad for JP Morgan Stanley for obvious reasons


# In[313]:


fig, ax = plt.subplots(figsize=(70,30))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### similar pattern with Morgan Stanley


# In[314]:


heat = df.groupby(['year','day','month_name'])['WFC'].sum().unstack().fillna(0)

heat['sum'] = heat.sum(axis=1)          

heat_2 = heat.sort_values(by='sum',ascending=False)

heat_2             #### time for wells fargo


# In[315]:


fig, ax = plt.subplots(figsize=(70,30))

sns.heatmap(heat_2,ax=ax,linewidths=0.5,cmap='viridis')

#### 2009 being the worst for wells fargo


# In[316]:


#### lets see the risk analysis

return_df.head()


# In[317]:


risk = return_df

fig, ax = plt.subplots(figsize=(15, 4))

colors = ['black', 'red', 'blue', 'green','brown','pink']

for i, ticker in enumerate(risk.mean().index):
    sns.scatterplot(x=[risk.mean()[ticker]], y=[risk.std()[ticker]], ax=ax, color=colors[i], label=ticker)

plt.xlabel('Expected Returns')
plt.ylabel('Risk')

ax.set_xlim(risk.mean().min() - 0.0005, risk.mean().max() + 0.0005)
ax.set_ylim(risk.std().min() - 0.0005, risk.std().max() + 0.0005)
ax.set_xticks([round(x, 5) for x in plt.xticks()[0]])
ax.set_yticks([round(y, 5) for y in plt.yticks()[0]])

for i, ticker in enumerate(risk.mean().index):
    plt.text(risk.mean()[i], risk.std()[i], ticker, fontsize=9, ha='right')

plt.legend()

#### the safest bet being Goldman Sachs and the worst investment being Citi Bank


# In[ ]:


###################################################################
####################### Part III - Monte Carlo
###################################################################


# In[225]:


#### monte carlo method to predict the future stock price

closing_df.head()


# In[226]:


mean_return = return_df.BAC.mean()
std_return = return_df.BAC.std()

closing_prices = closing_df.BAC


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Bank of America Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[227]:


mean_return = return_df.C.mean()
std_return = return_df.C.std()

closing_prices = closing_df.C


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Citi Bank Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[229]:


mean_return = return_df.GS.mean()
std_return = return_df.GS.std()

closing_prices = closing_df.GS


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Goldman Sachs Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[230]:


mean_return = return_df.JPM.mean()
std_return = return_df.JPM.std()

closing_prices = closing_df.JPM


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of JPMorgan Chase Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[231]:


mean_return = return_df.MS.mean()
std_return = return_df.MS.std()

closing_prices = closing_df.MS


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Morgan Stanley Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')


# In[232]:


mean_return = return_df.WFC.mean()
std_return = return_df.WFC.std()

closing_prices = closing_df.WFC


# Step 4: Simulate Future Prices
# Parameters
num_simulations = 1000000
num_days = 252  # Number of trading days in a year

# Simulation
simulated_prices = np.zeros((num_simulations, num_days))
simulated_prices[:, 0] = closing_prices[-1]

for i in range(1, num_days):
    random_returns = np.random.normal(mean_return, std_return, num_simulations)
    simulated_prices[:, i] = simulated_prices[:, i-1] * (1 + random_returns)
    
plt.figure(figsize=(15, 6))
plt.plot(simulated_prices.T)

plt.axhline(y=closing_prices[-1], color='black', linestyle='--')


plt.title(f'Monte Carlo Simulation of Wells Fargo Stock Price')
plt.xlabel('Days')
plt.ylabel('Price')

plt.savefig('Stock_WFC_Monte_Carlo_Simulation_plot.jpeg', dpi=300, bbox_inches='tight')


# In[ ]:


############################################################################################################################
#### Between 2006 and 2016, we analyzed the stock performance of major financial institutions, particularly during the #####
#### 2008 financial crisis. The analysis revealed that Citigroup (C) suffered the most significant losses during the #######
#### crash. We also examined the stocks of Bank of America (BAC), Goldman Sachs (GS), JPMorgan Chase (JPM), ################
#### Morgan Stanley (MS), and Wells Fargo (WFC). To project future stock prices, we employed Monte Carlo simulations,#######
#### providing insights into potential price movements based on historical data. ###########################################
############################################################################################################################

