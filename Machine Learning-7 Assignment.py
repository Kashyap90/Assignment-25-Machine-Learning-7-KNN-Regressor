
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
import math
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


with open("nba_2013.csv", 'r') as csvfile:
    nba = pd.read_csv(csvfile)


# In[3]:


nba.info()


# In[4]:


nba.describe()


# In[5]:


nba.columns


# In[6]:


# Check for null values in the dataframe:


# In[7]:


nba.isnull().sum()


# In[8]:


nba[nba['x3p.'].isnull()].head()


# In[9]:


nba.season.value_counts()


# In[10]:


nba.season_end.value_counts()


# In[11]:


# Retrieve features having datatype as Object:


# In[12]:


feat_list_obj = []
for i in nba.columns:
    if nba[i].dtype =='object':
        feat_list_obj.append(i)


# In[13]:


feat_list_obj


# In[14]:


# # prepare the list of features to be dropped from the Features to be used to train the model:


# In[15]:


feat_drop_list = feat_list_obj + ['season_end','pts']
feat_drop_list


# In[16]:


Features = nba.drop(feat_drop_list, axis=1)
# To make predictions for pts hence used as label:
Labels = nba['pts']
print(Features.shape)
print(Labels.shape)


# In[17]:


Features.isnull().sum()


# In[18]:


# DATA IMPUTATION

# Imputation is a process of replacing missing values with substituted values. In our dataset, some columns have missing values.
# We have replaced missing values with corresponding feature's median value.


# In[19]:


imp = Imputer(missing_values="NaN", strategy='median', axis=0)
# Independent Variable:
X = imp.fit_transform(Features)
# Dependent Values:
Y = Labels.values


# In[20]:


Features.shape,X.shape,Y.shape


# In[21]:


Sample = Features.dropna()


# In[22]:


def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
            #plt.show()
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)
            #plt.show()
plot_distribution(Sample, cols=2, width=20, height=35, hspace=0.8, wspace=0.8)


# In[23]:


# TEST TRAIN SPLIT the sample data:


# In[24]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[25]:


sns.set_style('whitegrid')
for K in range(20):
    K_value = K+1
    neigh = KNeighborsRegressor(n_neighbors = K_value)
    neigh.fit(X_train, Y_train)
    Y_pred = neigh.predict(X_test)
    print("RMSE is", np.sqrt(mean_squared_error(Y_pred,Y_test))," for K-Value:",K_value)


# In[26]:


# Note:

# It shows that we are get less error for values of K = 5,6 .


# In[27]:


K_value = 6
neigh = KNeighborsRegressor(n_neighbors = K_value)
neigh.fit(X_train, Y_train)
Y_pred = neigh.predict(X_test)
print("RMSE for KNN Regressor is ", np.sqrt(mean_squared_error(Y_pred,Y_test))," for K-Value:", K_value)
print("R Squared for KNN Regressor is ",r2_score(Y_test,Y_pred))


# In[28]:


# R Squared is a statistical measure of how close the data points are to thr fitted regression line.


# In[29]:


plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred)
plt.plot([0, 2500], [0, 2500], '--k')
plt.axis('tight')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.tight_layout()
plt.title("Actual vs Predicted points")


# In[30]:


# Applying Random Forest Regressor to predict NBA players score:


# In[31]:


from sklearn.ensemble import RandomForestRegressor
RFreg = RandomForestRegressor(random_state=1)
RFreg.fit(X_train, Y_train)
print("RMSE of Random Forest Regressor is ",np.sqrt(mean_squared_error(Y_pred,Y_test)))
print("R Squared for Random Forest Regressor is ", r2_score(Y_test,Y_pred))


# In[32]:


plt.figure(figsize=(10, 8))
plt.scatter(Y_test, Y_pred)
plt.plot([0, 2500], [0, 2500], '--k')
plt.axis('tight')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.tight_layout()
plt.title("Actual vs Preicted points- Random Forest Regressor")


# In[33]:


for depth in range(30):
    depth = depth + 1
    RFreg = RandomForestRegressor(max_depth=depth,random_state=1)
    RFreg.fit(X_train,Y_train)
    Y_pred = RFreg.predict(X_test)
    print("RMSE is ",np.sqrt(mean_squared_error(Y_pred,Y_test))," for max_depth ",depth)


# In[34]:


# CONCLUSION:

# The R Squared for KNN Regressor is 0.974834237452
# The R Squared for Random Forest Regressor is 0.991871085435

#R Squared is a statistical measure of how close the sample data points are to the fitted regression line.

# As also evident from the plot Random Forest Regressor gives a better prediction for the NBA players score as the data point are
# more fitted to the regression line compared to that of KNN Regressor.

