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

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(df['Close'], label='Share Closing Price')

ax.set_xlabel('DATES - YEAR/MONTH/DAY',fontsize=14, color='r')
ax.set_ylabel('SHARE CLOSING PRICE',fontsize=14, color='r')
ax.set_title('ORIGINAL CLOSING PRICE PLOT', fontsize=20, color='g')
ax.legend()



#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

new_data.to_csv('01_1.MovingAvgData.csv')
new_data.head(10)




#Last 3 months Prediction
#splitting into train and test
train = new_data[:2037]
test = new_data[2037:]

train.to_csv('01_2.MovingAvgTrainData.csv')
test.to_csv('01_3.MovingAvgValidData.csv')





# Shape Return No of ROWS AND COLUMN OF A DATAFRAME
new_data.shape, train.shape, test.shape




# TO Check the First and Last Record Value
train['Date'].min(), train['Date'].max(), valid['Date'].min(), valid['Date'].max()




#make predictions
stcokclose_preds = []

# range value taken from test.shape
test_shape_noofrecords = 63
#Taking No of Record to Calculate Avg
no_of_record = 10

for i in range(0,test_shape_noofrecords):
    if i < no_of_record:
        a = train['Close'][len(train)-no_of_record+i:].sum() + sum(stcokclose_preds)
    else:
        c = stcokclose_preds[-no_of_record:]
        a = sum(c)
    
    b = a/no_of_record
    stcokclose_preds.append(b)    





date_list = []

date_list = list(test.Date) # TAKE DATE FROM TEST DATA FRAME AND CONVERT IT TO LIST
#print (stcokclose_preds)
#print (date_list)
 
dict = {'Date':date_list,'Predict_Close': stcokclose_preds} # CREATE DICTIONARY USING LIST
#print(dict)

predict_df = pd.DataFrame(dict)  #CONVERT DICTIONARY TO DATAFRAME
predict_df.to_csv('01_4.MovingAvgPredictData.csv')





#calculate rmse
#MSE (Mean Squared Error) = np.mean((y_test - y_predtest)**2)
#RMSE (Root Mean Squared Error) = np.sqrt(MSE)

import numpy as np

x = np.power((np.array(valid['Close'])-stcokclose_preds),2)
mse = np.mean(x)
rmse = np.sqrt(mse)
rmse




# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(train['Date'],train['Close'],label = 'Training Closing Price')
ax.plot(test['Date'],test['Close'],label = 'Original Closing Price')
ax.plot(predict_df['Date'],predict_df['Predict_Close'],label = 'Predict Closing Price')

ax.set_xlabel('DATES - YEAR/MONTH/DAY',fontsize=14, color='r')
ax.set_ylabel('SHARE CLOSING PRICE',fontsize=14, color='r')
ax.set_title('Training Closing Price VS Original Closing Price VS Predict Closing Price', fontsize=20, color='g')
ax.legend()