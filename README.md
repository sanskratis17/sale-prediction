# sale-prediction
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#dataframe
df=pd.read_csv('/content/Advertising.csv')
df
#finding number of rows and columns
df.shape
df.info()
df.describe()
df.coloumn()
#checking duplicates
df.duplicated().sum()
plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['TV'],y=df['Sales'])
plt.show()
plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Radio'],y=df['Sales'])
plt.show()
plt.figure(figsize=(4,4))
sns.scatterplot(data=df,x=df['Newspaper'],y=df['Sales'])
plt.show()
#splitting the datset into X,the attributes and y,the target variable
X=df.drop('Sales',axis=1)
X
y=df['Sales']
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)
from sklearn.linear_model import LinearRegression
model= LinearRegression()
#fitting the model to the dataset
model.fit(X_train,y_train)
#predictions
y_predictions=model.predict(X_test)
y_predictions
# Lets evaluate the model for its accuracy using various metrics such as RMSE and R-Squared
from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_predictions,y_test))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_predictions,y_test)))
print('R-Squared',metrics.r2_score(y_predictions,y_test))
