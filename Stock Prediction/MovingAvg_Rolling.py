#import panda packages
import pandas as pd

#Read Data from CSV file And Store It IN DATAFRAME
df = pd.read_csv('NSE-TATAGLOBAL.csv')

#print TOP 10 RECORD
df.head(10)






#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#to plot the graph
import matplotlib.pyplot as plt
%matplotlib inline

#plot
plt.figure(figsize=(16,8))
plt.plot(df['Close'], label='Share Closing Price')
plt.title('ORIGINAL CLOSING PRICE PLOT', fontsize=20, color='g')
plt.xlabel('DATES - YEAR/MONTH/DAY',fontsize=14, color='r')
plt.ylabel('SHARE CLOSING PRICE',fontsize=14, color='r')
plt.legend()




# Get the STOCK CLOSE PRICE timeseries. This now returns a Pandas Series object indexed by date.
stockcloseingprice = df.loc[:, 'Close']

# Calculate the 10 and 150 days moving averages of the closing prices
short_rolling = stockcloseingprice.rolling(window=10).mean()
long_rolling = stockcloseingprice.rolling(window=150).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(stockcloseingprice.index, stockcloseingprice, label='Original Closeing Price')
ax.plot(short_rolling.index, short_rolling, label='10 Days Rolling Prediction')
ax.plot(long_rolling.index, long_rolling, label='150 Days Rolling Prediction')

ax.set_xlabel('DATES - YEAR/MONTH/DAY',fontsize=14, color='r')
ax.set_ylabel('SHARE CLOSING PRICE',fontsize=14, color='r')
ax.set_title('Original Closing Share Price VS Predict Closing Price', fontsize=20, color='g')
ax.legend()





#calculate rmse
#MSE (Mean Squared Error) = np.mean((y_test - y_predtest)**2)
#RMSE (Root Mean Squared Error) = np.sqrt(MSE)

import numpy as np

x = np.power((np.array(df['Close'])-short_rolling),2)
mse = np.mean(x)
rmse = np.sqrt(mse)
print ('SHORT TIME SERIES ROLLING ', rmse)

x = np.power((np.array(df['Close'])-long_rolling),2)
mse = np.mean(x)
rmse = np.sqrt(mse)
print ('LONG TIME SERIES ROLLING ', rmse)





