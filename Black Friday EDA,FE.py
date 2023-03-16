#!/usr/bin/env python
# coding: utf-8

# In[1]:


#dataset link -https://www.kaggle.com/datasets/sdolezel/black-friday?resource=download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Problem Statement
# A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month. The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.
# 
# Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products.

# In[2]:


#importing dataset
df_train=pd.read_csv('blackfriday_train.csv')
df_train.head()


# In[3]:


df_test=pd.read_csv('blackfriday_test.csv')
df_test.head()


# In[4]:


#df=pd.merge(df_train,df_test,on='User_ID',how='left')
df=df_train.append(df_test)


# In[5]:


df.head()


# In[6]:


#basic 
df.info()


# In[7]:


df.describe()


# In[8]:


#drop userid-waste
df.drop(['User_ID'],axis=1,inplace=True)


# In[9]:


df.head()


# In[10]:


#cat to numerical
#gender--encode
pd.get_dummies(df['Gender'])
#have to add this df to the main df and drop gender so directly do encoding


# In[11]:


#handling gender (cat feature)
df['Gender']=df['Gender'].map({'F':0,'M':1})
df.head()


# In[12]:


#handle age
df['Age'].unique()


# In[13]:


#encoding age
#pd.get_dummies(df['Age'],drop_first=True)
df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[14]:


df.head()


# In[15]:


#second technique
from sklearn import preprocessing

#label_encoder object knows how to understand word labels.
label_encoder=preprocessing.LabelEncoder()

#encode lables in column Age
df['Age']=label_encoder.fit_transform(df['Age'])

df['Age'].unique()


# In[16]:


#fixing categorical City_Category
df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[17]:


df_city


# In[18]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[19]:


#drop City_Category feature
df.drop('City_Category',axis=1,inplace=True)


# In[20]:


df.head()


# In[21]:


#missing values
df.isnull().sum()


# In[22]:


#purchase is test data is it should be null
#fix the other two
#focus on replacing missing values
df['Product_Category_2'].unique()


# In[23]:


df['Product_Category_2'].value_counts()


# In[24]:


df['Product_Category_2'].mode()


# In[25]:


#replace the missing value with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[26]:


df['Product_Category_2'].isnull().sum()


# In[27]:


#replace missing values fro product_category_3
df['Product_Category_3'].unique()


# In[28]:


df['Product_Category_3'].value_counts()


# In[29]:


df['Product_Category_3'].mode()


# In[30]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[31]:


df['Product_Category_3'].isnull().sum()


# In[32]:


df.head()


# In[33]:


#stay in current yrs
df['Stay_In_Current_City_Years'].unique()


# In[34]:


df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+',' ')


# In[35]:


df.head()


# In[36]:


df.info()


# In[37]:


#Stay_In_Current_City_Years is in object
#convert it to integer
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)
df.info()


# In[38]:


df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)
df.info()


# In[39]:


#visualization
sns.pairplot(df)


# In[40]:


#Age vs Purchase
sns.barplot('Age','Purchase',hue='Gender',data=df)


# men purchased more than women

# In[41]:


#visualization of purchase with occupation
sns.barplot('Occupation','Purchase',hue='Gender',data=df)


# In[42]:


#product_cat_1 vs purchase
sns.barplot('Product_Category_1','Purchase',hue='Gender',data=df)


# In[43]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=df)


# In[44]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data=df)


# product_category_1 is bought more around 20,000

# In[45]:


#feature scaling
#whenever purchase col is null it belongs to test data
df_test=df[df['Purchase'].isnull()]


# In[46]:


#the other belongs to train
df_train=df[~df['Purchase'].isnull()]


# In[47]:


X=df_train.drop('Purchase',axis=1)


# In[48]:


X.head()


# In[49]:


X.shape


# In[50]:


y=df_train['Purchase']


# In[51]:


y


# In[52]:


y.shape


# In[53]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[54]:


X_train.drop('Product_ID',axis=1,inplace=True)
X_test.drop('Product_ID',axis=1,inplace=True)


# In[55]:


#standard scaler as feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[56]:


#train the model


# In[57]:


from sklearn.ensemble import RandomForestRegressor
reg_rf=RandomForestRegressor()
reg_rf.fit(X_train,y_train)


# In[58]:


y_pred=reg_rf.predict(X_test)


# In[59]:


reg_rf.score(X_train,y_train)


# In[62]:


reg_rf.score(X_test,y_test)


# In[63]:


sns.displot(y_test-y_pred)
plt.show()


# In[64]:


from sklearn import metrics


# In[65]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[66]:


metrics.r2_score(y_test,y_pred)


# In[67]:


from sklearn.model_selection import RandomizedSearchCV

#number of trees in random forest
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
#number of features to consider at every split
max_features=['auto','sqrt']
#number of levels in the tree
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
#min no of samples req to split the node
min_samples_split=[2,5,10,15,100]
#min no of samples req at each leaf node
min_samples_leaf=[1,2,5,10]


# In[68]:


#create the random grid

randomgrid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf
}


# In[69]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = randomgrid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42,n_jobs=1)


# In[70]:


rf_random.fit(X_train,y_train)


# In[71]:


rf_random.best_params_


# In[72]:


prediction=rf_random.predict(X_test)


# In[73]:


plt.figure(figsize=(8,8))
sns.displot(y_test-prediction)
plt.show()


# In[74]:


plt.scatter(y_test,prediction,alpha=.5)
plt.xlabel("y_test")
plt.ylabel("prediction")
plt.show()


# In[75]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[77]:





# In[ ]:




