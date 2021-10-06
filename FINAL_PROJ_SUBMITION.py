#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


# Import data and perform processing 
import os as os 


# In[4]:


# import the relevent package
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier   # we are applying the boosting techinque
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

## Importing train_test_split,cross_val_score,GridSearchCV,KFold, RandomizedSearchCV - Validation and OptimizationC
from sklearn.model_selection import ShuffleSplit, train_test_split,cross_val_score,GridSearchCV,KFold, RandomizedSearchCV

# Importing Regressors - Modelling
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

# Importing Regression Metrics - Performance Evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pickle


# # Setting the working directory

# In[6]:


os.chdir('E:\\Data Science Programme- DSP-33\\Capstone_Project_Group_2\\Prasad')


# In[7]:


os.getcwd()


# In[8]:


os.listdir()


# In[9]:


import xlrd
# read the dataset
df = pd.read_excel('staff utlz latest 16-17_masked.xlsx')

df1 = pd.read_excel('staff utlz latest 17-18_masked.xlsx')

df2 = pd.read_excel('Terminations 15-18.xlsx')


# In[10]:


# check the dimension of the datset
print("Dimension of the df data",df.shape)
print("Dimension of the df1 data",df1.shape)
print("Dimension of the df1 data",df2.shape)


# In[11]:


import warnings
warnings.filterwarnings('ignore')


# # Concatingnating dataset

# In[12]:


alldata = df.append(df1) 
 
# here we are concatinating df and df1 dataset using .append method


# In[13]:


alldata = alldata.append(df2)

# here we are concatinating alldat and df2 dataset using .append method


# In[14]:


alldata.head()


# In[15]:


alldata.shape


# In[17]:


# dropping the unimportant variables and also some variable having missing values 80%
new_data = alldata.drop(['Employee No','Employee Number','Join Date','Supervisor name','Previous Employer','Last Update Date','Emp Ref.','Employee Name','Latest  Available Rating','YEAR of Birth','Termination Date'], axis=1)


# In[18]:


new_data.shape


# # Checking datatypes

# In[19]:


new_data.dtypes


# # Summary Statistic

# Here we take a look at the summary of each attributes

# In[20]:


new_data.describe()
   


# In[22]:


import seaborn as sns

sns.distplot(new_data['Avg Total Available Hours'])

plt.show()  # distribution of Avg Total Available Hours

# showing left skewed in this between 1500 and 2000 Avg Total Available Hours very high


# # Missing values analysis

# In[23]:


# check for missing values
missing = new_data.isnull().sum()/len(new_data)
missing


# In[24]:


# check for missing values
missing = alldata.isnull().sum()
missing = missing[missing > 0]    # wherever you have missing values greater than 0 it is doing the bar plot.

# We can use bar plot to check missing values
missing.sort_values(inplace=True)
missing.plot.bar()

plt.show()

# A lot of missing values are present in the datasets.


# In[25]:


df1=new_data.replace(np.NaN, 0)


# In[26]:


df2=df1.replace(['-'],[0])


# # Missing values treatment for numerical feature

# In[27]:


# fill na with median
new_data.fillna(df2.median(),inplace=True)

# here we are taking the entire dataset and replacing the missing values with median so impact of this line is that, for all
# the categorical variables the imputation could not be carried out. And for all the numerical variables the missing values 
# will be replace with median value.


#  There is lots of NaN entries. 

# In[28]:


# using heatmap to figure out missing data if any
sns.heatmap(df2.isnull(),yticklabels=False,cbar=False) 


# In[29]:


# Again check for missing values
df2.isnull().sum()


# # Examine the numerical and categorical features in the dataset after missing value treatment 

# In[30]:


df_numeric_features = df2.select_dtypes(include=[np.number])

df_numeric_features.columns

# here we are creating an object to have all the numeric features in it.


# In[31]:


df_categorical_features = df2.select_dtypes(include=[np.object])

df_categorical_features.columns

# In this object have all the categorical features.


# In[32]:


df_numeric_features.shape


# In[33]:


df_numeric_features.info()


# In[34]:


df_numeric_features.head()


# In[35]:



df_categorical_features.shape


# In[36]:


df_categorical_features.head()


# In[42]:


sns.countplot(x='People Group',hue='Profit Center',data=df2);
plt.show()

# Approximetly 650 employees have clent group and they are falls in profit center 3.
# 300 employees dont have any group.


# In[43]:


sns.countplot(x='Profit Center',hue='Current Status',data=df2);
plt.show()

# In case of profit center 1 and 2 active employees are very high.
# profit center 6,7,8,9 there are no employees active.


# In[44]:


sns.countplot(x='Current Status',hue='Employee Location',data=df2);
plt.show()

# Location 3,7 and 1 almost 400 employees are active and 100 employees resigned.


# In[45]:


sns.countplot(x='Profit Center',hue='Employee Position',data=df2);
plt.show()

# large no. of employees which do not have any position.
# profit center 1 there are 200 employees which falls in level 8 and 90 employees which falls in level 5


# In[46]:


sns.countplot(x='Gender',hue='Leaving Reason',data=df2);
plt.show()

# There are 1750 employees which dont give leaving reason.
# 100 males leaving company because of career growth. And approximetly 50 females giving job related reasons.


# In[47]:


sns.countplot(x='People Group',hue='Current Status',data=df2);
plt.show()

# In Client group 1400 employees are active and 240 employees resigned.
# In Service Staff 30 employees are New Joiner
# There are approximetly 250 employees which do not follow any group.


# In[48]:


df_categorical_features.columns


# In[ ]:





# In[49]:


from sklearn.preprocessing import LabelEncoder
cols = ('Profitn Center', 'Employee Position', 'Employee Location',
       'People Group', 'Employee Category', 'Current Status',
       'Gender', 'Leaving Reason')

# here we are applying the labelEcoder for on these categorical features. So we are just going to start the labelEncoding for 
# all the categorical features so the different levels of this categorical features are going to be assign a label so this
# we will go and carry out the creation of dummy variables for this categorical features.


# In[50]:


# process columns, apply labelEncoder to categorical features
for c in cols:
    label_ec = LabelEncoder()
    label_ec.fit(list(df2[c].values))
    df2[c] = label_ec.transform(list(df2[c].values))
    
# here we are doing a for loop and we are iterating over cols object and c will take a FireplaceQu in 1st iteration so on, 
# and it will transform the label of each of this categorical variables after you execute this code.


# In[51]:


df2.head()


# In[52]:


df2 = pd.get_dummies(df2)
print(df2.shape)


# In[53]:


df2.head()


# In[54]:


#df2=df2.drop(['Total_Hours-Apr-16','Utilization%-Mar-17','NC_Hours-Mar-17','BD_Hours-Mar-17','Training_Hours-Mar-17',
#'Training_Hours-Mar-17','Leave_Hours-Mar-17','Work_Hours-Mar-17','Total_Available Hours-Mar-17',
#'Total_Hours-Mar-17','Utilization%-Feb-17','NC_Hours-Feb-17','BD_Hours-Feb-17','Training_Hours-Feb-17',
#'Training_Hours-Feb-17','Leave_Hours-Feb-17','Work_Hours-Feb-17','Total_Available Hours-Feb-17',
#'Total_Hours-Feb-17','NC_Hours-Jan-17','BD_Hours-Jan-17','Training_Hours-Jan-17','Training_Hours-Jan-17',
#'Leave_Hours-Jan-17','Work_Hours-Jan-17','Total_Available_Hours-Jan-17','Total_Hours-Jan-17','Utilization%-Jan-17','Utilization%-Dec-16','NC_Hours-Dec-16','BD_Hours-Dec-16','Training_Hours-Dec-16','Training_Hours-Dec-16','Leave_Hours-Dec-16','Work_Hours-Dec-16','Total_Available_Hours-Dec-16',
#'Total_Hours-Dec-16','Utilization%-Nov-16','NC_Hours-Nov-16','BD_Hours-Nov-16','Training_Hours-Nov-16','Training_Hours-Nov-16','Leave_Hours-Nov-16','Work_Hours-Nov-16','Total_Available_Hours-Nov-16',
#'Total_Hours-Nov-16','Utilization%-Oct-16','NC_Hours-Oct-16','BD_Hours-Oct-16','Training_Hours-Oct-16','Training_Hours-Oct-16','Leave_Hours-Oct-16','Work_Hours-Oct-16','Total_Available_Hours-Oct-16',
#'Total_Hours-Oct-16','Total_Available Hours-Apr-17','Total_Hours-Apr-17','Utilization%-Apr-17','NC_Hours-Apr-17','BD_Hours-Apr-17',
#'Training_Hours-Apr-17','Leave_Hours-Apr-17','Work_Hours-Apr-17','Total_Available Hours-Apr-17','Total_Hours-May-17','NC_Hours-May-17','BD_Hours-May-17','Training_Hours-May-17','Training_Hours-May-17','Leave_Hours-May-17','Work_Hours-May-17','Total_Available_Hours-May-17',
#'Total_Hours-Jun-17','Utilization%-Jun-17','NC_Hours-Jun-17','BD_Hours-Jun-17','Training_Hours-Jun-17','Training_Hours-Jun-17','Leave_Hours-Jun-17','Work_Hours-Jun-17','Total_Available_Hours-Jun-17',
#'Total_Hours-Jul-17','Utilization%-Jul-17','NC_Hours-Jul-17','BD_Hours-Jul-17','Training_Hours-Jul-17','Training_Hours-Jul-17','Leave_Hours-Jul-17','Work_Hours-Jul-17','Total_Available_Hours-Jul-17',
#'Total_Hours-Aug-17','Utilization%-Aug-17','NC_Hours-Aug-17','BD_Hours-Aug-17','Training_Hours-Aug-17','Training_Hours-Aug-17','Leave_Hours-Aug-17','Work_Hours-Aug-17','Total_Available_Hours-Aug-17',
#'Total_Hours-Oct-17'], axis =1)


# In[55]:


#df2=df2.drop(['Total_Hours-Sep-16','Utilization%-Sep-16','NC_Hours-Sep-16','BD_Hours-Sep-16',
#'Training_Hours-Sep-16','Training_Hours-Sep-16','Leave_Hours-Sep-16','Work_Hours-Sep-16','Total_Available_Hours-Sep-16',
#'Utilization%-Aug-16','NC_Hours-Aug-16','BD_Hours-Aug-16',
#'Training_Hours-Aug-16','Training_Hours-Aug-16','Leave_Hours-Aug-16','WorkHours-Aug-16','Total_Available_Hours-Aug-16',
#'Total_Hours-Aug-16','NC_Hours-Jul-16','BD_Hours-Jul-16',
#'Training_Hours-Jul-16','Training_Hours-Jul-16','Leave_Hours-Jul-16','Work_Hours-Jul-16','Total_Available Hours-Jul-16',
#'Total_Hours-Jul-16','Utilization%-Jul-16','Utilization%-Jun-16','NC_Hours-Jun-16','BD_Hours-Jun-16',
#'Training_Hours-Jun-16','Training_Hours-Jun-16','Leave_Hours-Jun-16','Work_Hours-Jun-16','Total_Available_Hours-Jun-16',
#'Total_Hours-Jun-16','Utilization%-May-16','NC_Hours-May-16','BD_Hours-May-16',
#'Training_Hours-May-16','Training_Hours-May-16','Leave_Hours-May-16','Work_Hours-May-16','Total_Available_Hours-May-16',
#'Total_Hours-May-16','Utilization%-Apr-16','NC_Hours-Apr-16','BD_Hours-Apr-16',
#'Training_Hours-Apr-16','Training_Hours-Apr-16','Leave_Hours-Apr-16','Work_Hours-Apr-16','Total_Available_Hours-Apr-16'], axis =1)


# In[56]:


df2.head


# In[57]:


#df2=df2.drop(['Total_Hours-Apr-17',
 #      'Total_Available_Hours-Apr-17', 'Work_Hours-Apr-17',
  #     'Leave_Hours-Apr-17', 'Training_Hours-Apr-17', 'BD_Hours-Apr-17',
   #    'NC_Hours-Apr-17', 'Utilization%-Apr-17', 'Total_Hours-May-17',
    #   'Total_Available_Hours-May-17', 'Work_Hours-May-17',
     #  'Leave_Hours-May-17', 'Training_Hours-May-17', 'BD_Hours-May-17',
 #      'NC _Hours-May-17', 'Utilization%-May-17', 'Total_Hours-Jun-17',
  #     'Total_Available_Hours-Jun-17', 'Work_Hours-Jun-17',
   #    'Leave_Hours-Jun-17', 'Training_Hours-Jun-17', 'BD_Hours-Jun-17',
   #    'NC_Hours-Jun-17', 'Utilization%-Jun-17', 'Total_Hours-Jul-17',
   #    'Total_Available_Hours-Jul-17', 'Work_Hours-Jul-17',
   #    'Leave_Hours-Jul-17', 'Training_Hours-Jul-17', 'BD_Hours-Jul-17',
   #    'NC_Hours-Jul-17', 'Utilization%-Jul-17', 'Total_Hours-Aug-17',
   #    'Total_Available Hours-Aug-17', 'Work_Hours-Aug-17',
   #    'Leave_Hours-Aug-17', 'Training_Hours-Aug-17', 'BD_Hours-Aug-17',
   #    'NC_Hours-Aug-17', 'Utilization%-Aug-17', 'Total_Hours-Sep-17',
   #    'Total_Available_Hours-Sep-17', 'Work_Hours-Sep-17',
#   'Leave_Hours-Sep-17', 'Training_Hours-Sep-17', 'BD_Hours-Sep-17',
#       'NC_Hours-Sep-17', 'Utilization%-Sep-17', 'Total_Hours-Oct-17',
#       'Total_Available_Hours-Oct-17', 'Work_Hours-Oct-17',
#       'Leave_Hours-Oct-17', 'Training_Hours-Oct-17', 'BD_Hours-Oct-17',
#       'NC_Hours-Oct-17', 'Utilization%-Oct-17', 'Total_Hours-Nov-17',
#       'Total_Available_Hours-Nov-17', 'Work_Hours-Nov-17',
#      'Leave_Hours-Nov-17', 'Training_Hours-Nov-17', 'BD_Hours-Nov-17',
#      'NC_Hours-Nov-17', 'Utilization%-Nov-17', 'Total_Hours-Dec-17',
#       'Total_Available_Hours-Dec-17', 'Work_Hours-Dec-17',
#       'Leave_Hours-Dec-17', 'Training_Hours-Dec-17', 'BD_Hours-Dec-17',
#       'NC_Hours-Dec-17', 'Utilization%-Dec-17', 'Total_Hours-Jan-18',
#       'Total_Available_Hours-Jan-18', 'Work_Hours-Jan-18',
#       'Leave_Hours-Jan-18', 'Training_Hours-Jan-18', 'BD_Hours-Jan-18',
#       'NC_Hours-Jan-18', 'Utilization%-Jan-18', 'Total_Hours-Feb-18',
#       'Total_Available_Hours-Feb-18', 'Work Hours-Feb-18',
#       'Leave_Hours-Feb-18', 'Training_Hours-Feb-18', 'BD_Hours-Feb-18',
#       'NC_Hours-Feb-18', 'Utilization%-Feb-18', 'Total_Hours-Mar-18',
#       'Total_Available_Hours-Mar-18', 'Work_Hours-Mar-18',
#       'Leave_Hours-Mar-18', 'Training_Hours-Mar-18', 'BD_Hours-Mar-18',
#       'NC_Hours-Mar-18', 'Utilization%-Mar-18'],axis =1)


# # Spliting Target Variable

# In[58]:


x = df2.iloc[:, df2.columns !='Current Status']  # all input
y = df2.iloc[:, df2.columns=='Current Status']   # target

# here we are segregating the input variables and target variables and we are putting all the input variables under the 
# predictor. And we are putting the dependent variable under the label target.


# # Spliting Dataset into Train and Test

# In[59]:


# Let us now split the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
print("x_train",x_train.shape)
print("x_test",x_test.shape)
print("y_train",y_train.shape)
print("y_test",y_test.shape)

# here we are going to use 80 and 20 ratio. So we have 80 observation in train data and 20 obs. in the test data.
# In the output we are seeing the dimension of the input and output variable for both training and testing.


# # Feature scaling

# In[60]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# # Performing LDA

# In[61]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=1)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)


#          In this case we set the n_components to 1, since we first want to check the performance of our classifier with a 
#     single linear discriminant. Finally we execute the fit and transform methods to actually retrieve the linear
#     discriminants.
# 
#         Notice, in case of LDA, the transform method takes two parameters: the x_train and the y_train.
#     However in the case of PCA, the transform method only requires one parameter i.e. x_train. This reflects the fact
#     that LDA takes the output class labels into account while selecting the linear discriminants, while 
#     PCA doesn't depend upon the output labels.

# # Training and Making Predictions

# In[62]:


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


#         Since we want to compare the performance of LDA with one linear discriminant to the performance of PCA with 
#     one principal component, we will use the same Random Forest classifier that we used to evaluate performance 
#     of PCA-reduced algorithms.

# # Evaluating the Performance

# In[63]:


# As always, the last step is to evaluate performance of the algorithm with the help of a confusion matrix and
# find the accuracy of the prediction.

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))


# In[64]:


import xgboost
classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.4, learning_rate=0.1,
       max_delta_step=0, max_depth=6, min_child_weight=7, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)


# In[65]:


from sklearn.metrics import (accuracy_score, log_loss, classification_report)
classifier.fit(x_train,y_train)
prediction=classifier.predict(x_test)

print("Accuracy score: {}".format(accuracy_score(y_test, prediction)))
print("="*80)
print(classification_report(y_test, prediction))


# In[66]:


from sklearn.linear_model import LinearRegression
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lm  = LinearRegression()
model  = lm.fit(x_train,y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
    
print('Linear Regression -', 'RMSE Train:', math.sqrt(mean_squared_error(y_train_pred, y_train)))
print('Linear Regression -', 'RMSE Test:' ,math.sqrt(mean_squared_error(y_test_pred, y_test)))  
print('Linear Regression -', 'R2_score Train:', r2_score(y_train_pred, y_train))
print('Linear Regression -', 'R2_score Test:' ,r2_score(y_test_pred, y_test)) 


# In[67]:


#Other Regressors 


# In[68]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor

regressors = [
            ("Linear - ", LinearRegression(normalize=True)),
            ("Ridge - ",  Ridge(alpha=0.5, normalize=True)),
            ("Lasso - ",  Lasso(alpha=0.5, normalize=True)),
            ("ElasticNet - ",  ElasticNet(alpha=0.5, l1_ratio=0.5, normalize=True)),
            ("Decision Tree - ",  DecisionTreeRegressor(max_depth=5)),
            ("Random Forest - ",  RandomForestRegressor(n_estimators=100)),
            ("AdaBoost - ",  AdaBoostRegressor(n_estimators=100)),
            ("GBM - ", GradientBoostingRegressor(n_estimators=100))]


# In[69]:


for reg in regressors:
    reg[1].fit(x_train, y_train)
    y_test_pred= reg[1].predict(x_test)
    print(reg[0],"\n\t R2-Score:", reg[1].score(x_test, y_test),
                "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")


# In[70]:


#Feature selection


# In[71]:


rndf = RandomForestRegressor(n_estimators=150)
rndf.fit(x_train, y_train)
importance = pd.DataFrame.from_dict({'cols':x_train.columns, 'importance': rndf.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
plt.figure(figsize=(20,15))
sns.barplot(importance.cols, importance.importance)
plt.xticks(rotation=90)


# In[72]:


imp_cols = importance[importance.importance >= 0.005].cols.values
imp_cols


# In[73]:


# Fitting models with columns where feature importance>=0.005
x_train, x_test, y_train, y_test = train_test_split(x[imp_cols],y,test_size=0.25, random_state = 100)
for reg in regressors:
    reg[1].fit(x_train, y_train)
    y_test_pred= reg[1].predict(x_test)
    print(reg[0],"\n\t R2-Score:", reg[1].score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")


# In[74]:


imp_cols = importance[importance.importance >= 0.001].cols.values
imp_cols


# In[75]:


# Fitting models with columns where feature importance>=0.001
x_train, x_test, y_train, y_test = train_test_split(x[imp_cols],y,test_size=0.25, random_state = 100)
for reg in regressors:
    reg[1].fit(x_train, y_train)
    y_test_pred= reg[1].predict(x_test)
    print(reg[0],"\n\t R2-Score:", reg[1].score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(y_test_pred, y_test)),"\n")


# In[76]:


#RandomForest,decision forest and GBM provide us with the best RMSE and R2-Score when selecting columns with feature importance >= 0.001


# In[77]:


#Validation of the model..Validating our models using K-Fold Cross Validation for Robustness


# In[78]:


from sklearn.model_selection import ShuffleSplit, train_test_split,cross_val_score,GridSearchCV,KFold, RandomizedSearchCV
scoring = 'neg_mean_squared_error'
results=[]
names=[]
for modelname, model in regressors:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, x_train,y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(modelname)
    print(modelname,"\n\t CV-Mean:", cv_results.mean(),
                    "\n\t CV-Std. Dev:",  cv_results.std(),"\n")


# In[79]:


#RandomForest and GBM provide us with the best validation score, both w.r.t. CV-Mean and CV-Std. Dev


# In[80]:


#Therefore we choose these two models to optimize. We do this by finding best hyper-parameter values which give us even better R2-Score and RMSE values


# In[81]:


#Tuning Model for better Performance -- Hyper-Parameter Optimization


# In[82]:


#Tuning the RandomForestRegressor, GradientBoostingRegressor Hyper-Parameters using GridSearchCV


# In[83]:


regressors


# In[84]:


#Random Forest Regressor


# In[85]:


RF_Regressor =  RandomForestRegressor(n_estimators=100, n_jobs = -1, random_state = 100)

CV = ShuffleSplit(test_size=0.25, random_state=100)

param_grid = {"max_depth": [5, None],
              "n_estimators": [50, 100, 150, 200],
              "min_samples_split": [2, 4, 5],
              "min_samples_leaf": [2, 4, 6]
             }


# In[86]:


rscv_grid = GridSearchCV(RF_Regressor, param_grid=param_grid, verbose=1)


# In[87]:


rscv_grid.fit(x_train, y_train)


# In[ ]:


rscv_grid.best_params_


# In[ ]:


model = rscv_grid.best_estimator_
model.fit(x_train, y_train)


# In[ ]:


model.score(x_test, y_test)


# In[ ]:


import pickle
RF_reg = pickle.dumps(rscv_grid)


# In[88]:


#Gradient Boosting Regressor


# In[89]:


GB_Regressor =  GradientBoostingRegressor(n_estimators=100)

CV = ShuffleSplit(test_size=0.25, random_state=100)

param_grid = {'max_depth': [5, 7, 9],
              'learning_rate': [0.1, 0.3, 0.5]
             }


# In[90]:


scv_grid = GridSearchCV(GB_Regressor, param_grid=param_grid, verbose=1)


# In[91]:


rscv_grid = GridSearchCV(GB_Regressor, param_grid=param_grid, verbose=1)


# In[92]:


rscv_grid.fit(x_train, y_train)


# In[93]:


rscv_grid.best_params_


# In[94]:


model = rscv_grid.best_estimator_
model.fit(x_train, y_train)


# In[95]:


model.score(x_test, y_test)


# In[96]:


import pickle
GB_reg = pickle.dumps(rscv_grid)


# In[97]:


#Comparing performance metric of the different models


# In[98]:


RF_regressor = pickle.loads(RF_reg)
GB_regressor = pickle.loads(GB_reg)


# In[99]:


print("RandomForestRegressor - \n\t R2-Score:", RF_regressor.score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(RF_regressor.predict(x_test), y_test)),"\n")
      
print("GradientBoostingRegressor - \n\t R2-Score:", GB_regressor.score(x_test, y_test),
                 "\n\t RMSE:", math.sqrt(mean_squared_error(GB_regressor.predict(x_test), y_test)),"\n")


# In[100]:


#Choosing the model


# In[101]:


#We can see that XGBOOST gives better result with an R2-Score of more than 96% and while keeping RMSE value low(=0.0.1715880). 
#So, XGBOOST should be used as the regression model for this dataset. However Random Forest Regressor works well too


# In[ ]:





# In[ ]:




