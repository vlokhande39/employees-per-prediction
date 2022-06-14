# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:24:30 2022
@author: lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:17:19 2022
@author:
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel('C:\\Users\\SAI\\Desktop\\final_project67\\final_data.xlsx')
pd.set_option('display.max_columns',None)
df=data
#data1[['Age','Marital_status']] = data1[['Age','Marital_status']].mask(np.random.random(data1[['Age','Marital_status']].shape) < .1)
#data1['WorkLifeBalance'] =data1['WorkLifeBalance'].mask(np.random.random(data1['WorkLifeBalance'].shape)<0.3)

#To view first five rows in the dataset
df.head()
#To view last five data
df.tail()
#To view the categorical and numerical columns and its datatypes in the dataset
df.info()
#To check the size of the data
df.shape
#To see the column name 
df.columns

#To find the statistical property of data
df.describe() 
# describe statistics and shape of the data distribution 


# =======================================================================

#Data Preprocessing

# Data Cleaning

#Dropping the columns not necessary for analysis
df=data.drop('Employee_ID',axis=1)

# checking any null values
df.isnull().sum()

# Visualisation 'Graphical Representation'

#To check imbalance 
df['Performancerating'].value_counts() #3 good, average, poor - 3 classes in taget feature

plt.pie(df.Performancerating.value_counts(), labels = ['good', 'average', 'poor'], autopct='%1.f%%', pctdistance=0.5)
plt.title('Performance rating of employees')

sns.countplot(df['Performancerating'])
# given data is balanced dataset.

import seaborn as sns
#Visualizing the Missing values
sns.heatmap(df.isnull(),cbar=False,cmap='viridis')
df.isna().sum()

#boxplot - checking presence of outliers
f, ax = plt.subplots(2,4, figsize = (15,7))
plt.suptitle('Boxplot of numerical features')
sns.boxplot(x = df.Age, ax = ax[0][0])
sns.boxplot(x = df.Job_level, ax = ax[0][1])
sns.boxplot(x = df.Annual_income, ax = ax[0][2])
sns.boxplot(x = df.Experience, ax = ax[0][3])
sns.boxplot(x = df.Trainingtime, ax = ax[1][0])
sns.boxplot(x = df.ProjectCompletion, ax = ax[1][1])
sns.boxplot(x = df.PercentSalaryHike, ax = ax[1][2])
f.delaxes(ax[1,3]) 

# age and experience feature have outliers 

#creating winsorization techniques to handle outliers
from feature_engine.outliers import Winsorizer
gaussian_winsor = Winsorizer(capping_method='gaussian', tail='both', fold=3)
iqr_winsor = Winsorizer(capping_method='iqr', tail='both',fold=1)
quantiles_winsor = Winsorizer(capping_method='quantiles', tail='both', fold=0.10)
#df['Age'].value_counts()
#handling outliers
df[['Age']] = iqr_winsor.fit_transform(df[['Age']])
df[['Experience']] = iqr_winsor.fit_transform(df[['Experience']])
df['Experience'].value_counts()
# age & experience features have been winsorized using 'iqr' method.

# boxplot for checking outliers after winsorization
sns.boxplot(df.Age)
sns.boxplot(df.Experience)

#checking duplicates 
duplicates=df.duplicated()
duplicates
sum(duplicates)
#data1 = data1.drop_duplicates("Employee_ID", keep='first')
#data1= data1.reset_index(drop=True)

#  objects(categoricals) into numeric - Label Encoding

from sklearn.preprocessing import LabelEncoder
categories = ['Gender', 'Marital_status', 'Education',
       'EnvironmentSatisfaction', 'JobInvolvement',
       'Job_satisfaction', 'RelationshipSatisfaction',
       'Working_hrs_per_Day', 'WorkLifeBalance',
       'BehaviouralCompetence', 'On_timeDelivery', 'TicketSolvingManagements',
       'WorkingFromHome', 'Psycho_social_indicators',
       'Over_time', 'Attendance', 'NetConnectivity',
       'Department', 'Position', 'Performancerating']

# Encode Categorical Columns
le = LabelEncoder()
df[categories] = df[categories].apply(le.fit_transform)
df 

# ===========================================================================

#  EDA (Exploratory Data Analysis)
# --------------------------------
 
# Descrptive Analytics
# # Measure of central tendancy - 1st moment business decision 

df.mean()

# Observation: 1) average age of employee is 43.81
        #      2) average salary of employee is 15.18 in lakhs
        #      3) average experience of employee is 14.08 yrs
        #      4) average training time is 3.5 months
        #      5) average no.of projects completed is 10
        #      6) average salary hike of employee in percentages is 8.5
        
df.median()
# Observation:  1. average age of employee is 43
        #       2. average annual income of employee is 15 lakhs
        #       3. average work experience of employee is 16
        #       4. average training time is 4 months
        #       5. average no.of. projects completed is 12
        #       6. avereage percent salary hike is 8
        #       7. average job level is 3
      
        
    # Observation: Mean and median values are not same due to outliers in the dataset.
    
#Finding mode for categorical data
df[["Gender","Marital_status","EnvironmentSatisfaction","Over_time"]].mode()

# most occuring value..
# gender - female, marital status - single, environment satisfaction- medium, overtime-no

# # measure of dispersion
df.var() # 2nd moment business decision -var(), std()
# variance - The variance measures the average degree to which each point differs from the mean.


# Variance Observation : age - 105.17
             
             #  annual income - 36.95
             # work experience - 41.9
             # training time  - 2.86
             # project evaluation - 11.25
             # percent salary hike - 20.79
             
df.std() # standard deviation - Standard deviation is the spread of a group of numbers from the mean.

# Standard Deviation Observation: 
     #  age  10.25
     # anual income - 6
     
     # work experience - 6.48
     # training time - 1.69
     # project evaluation -  3.35
     # percent salary hike 4.5
     
  # Note: While standard deviation is the square root of the variance, variance is the average of all data points within a group.
  
range = max(df.Age)-min(df.Age)
range #48

# # measure of skewess and kurtosis - 3rd & 4th business moment decisions

df.skew() #3rd moment business decision - skewness - a long tail
# Skewness refers to a distortion or asymmetry that deviates from the symmetrical bell curve, or normal distribution, in a set of data. If the curve is shifted to the left or to the right, it is said to be skewed

# Observations:
    # age 0.17 psitively or right skewed
    # joblevel   0.025 - positively or right skewed
    # annual income -0.05 - positively or right skewed
    # work experience -0.58 negatively or left skewed
    # training time  -0.0040  - negatively or left skewed
    # project evaluation   -0.37 - negatively or left skewed
    # percent salary hike  0.01  - positively or right skewed

df.kurt() # 4th moment business decision- measure of tailedness of probability distribution
# Kurtosis is defined as the standardized fourth central moment of a distribution minus 3 (to make the kurtosis of the normal distribution equal to zero).
# standard normal distribution has kurtosis of 3 (Mesokurtic), 
# kurtosis >3 is  - leptokurtic, <3 is platykurtic

   # Observations:
       # age  is    -0.66- platykurtic
       # joblevel    -1.28  - leptokurtic
       # anual income  -1.01- platykurtic
       # work experience  -1.55 platykurtic
       # training time  -1.24 - leptokurtic
       # project evaluation   -1.25 - platykurtic
       # percent salary hike  -1.18 - platykurtic

# --------------------------------

#Fifth moment business decision - Graphical representation

# Univariate analysis
# --------------------------------------------------

# histogram
df.hist() # overall distribution of data

sns.histplot(data=df, x='Age', kde=True)
# observation: 
    # age is slightly skewed. (right skewed) 

sns.histplot(data=df, x='Annual_income', kde=True)
# observation:
    # salary is uniformly distributed.
        
sns.histplot(data=df, x='Experience', kde=True)#####
# observations:
    # experience feature is  left skewed.

sns.histplot(data=df, x='Trainingtime', kde=True)
# observations:
    # training time - data is uniformly distributed.
    
df.columns


# count plot  - categorical features 

ax = sns.countplot(x="Performancerating", data=df) # countplot for performance rating feature
    # Observation:
        # data is balanced dataset..
      
sns.countplot(x='Performancerating', hue="Gender", data=df)
# observations : 
# Female employees performance is good than male employees performance.

sns.countplot(x='Performancerating', hue="WorkingFromHome", data=df)
# observation:
    # both work form home & working from office employees performance is similar.
    
sns.countplot(y='Department', hue="Performancerating", data=df)

# observation:
    # employees in Finance & HR departments performing similiar in performance.
    
sns.countplot(y='Position', hue="Performancerating", data=df)
# observation:
    # budget analyst & hr positions performing good, managers are performing poor.
    
sns.countplot(y='Department', hue="EnvironmentSatisfaction", data=df)
# observation:
    # R&D department employees are very highly satisfied with environment & IT deaprtment employees are low satisfied.
sns.countplot(x='Job_satisfaction', hue="WorkingFromHome", data=df)
# observation:
    # most employees working from home are not satisfied with their job.
    
sns.countplot(x='EnvironmentSatisfaction', hue="WorkingFromHome", data=df)
# observation:
    # most emplpoyees working from home are satisfied with the environment.
    
sns.countplot(x='JobInvolvement', hue="WorkingFromHome", data=df)
# observation:
    # employees working from home are highly involved in job.
        
sns.countplot(x='RelationshipSatisfaction', hue="Gender", data=df)
# observation:
    # Male employees are satisfied with relationship
sns.countplot(y='Position', hue="BehaviouralCompetence", data=df)
# observation:
    # scientist having excellent behaviourial competence & recruiters have HR in behaviourial competence.

sns.countplot(x='WorkLifeBalance', hue="WorkingFromHome", data=df)
# observation:
    # 
sns.countplot(x='WorkLifeBalance', hue="Gender", data=df)
# observation:
    #most of Male employees have best work life balance.
        
sns.countplot(x='Marital_status', hue="On_timeDelivery", data=df)
# observation:
    # single employees are excellent in delivering projects.
    
sns.countplot(x='TicketSolvingManagements', hue="Gender", data=df)
# observation:
    # male employees are excellent in ticket solving management.
    
sns.countplot(y='Department', hue="TicketSolvingManagements", data=df)
# observation:
    # HRM department is excellent in ticket solving management & finance department is poor in ticket solving management.

sns.countplot(x='Performancerating', hue="NetConnectivity", data=df)
# observation:
    # employees with good internet connection have good performance & We can see that the issue of bad net connection leads to low performance"""
sns.countplot(x='Over_time', hue="NetConnectivity", data=df)
# We can see that the issue of bad net connection leads to over time.
sns.countplot(x='WorkingFromHome', data=df)
# employees working from home are slight more than work from office employees

# ------------------------------------------------------------------

#pie chart - target variable
plt.pie(df.performance_rating.value_counts(), labels = ['excellent', 'good', 'low', 'outstanding'], autopct='%1.f%%', pctdistance=0.5)
plt.title('Performance rating of employees')
"""
Performance rating is the target variable of our dataset
35% of the employees performance rating are excellent and 20% of the employees has low performance rating
 17% of the employees only shown to have outstanding performance rating
"""
df.columns
#---> Bivarirate analysis

#bar plot - categorical vs numerical features

sns.barplot(x = df.Performancerating, y = df.Experience)
# observation:
    #  employees with more experience performance is good.

sns.barplot(x = df.Performancerating, y = df.Age)
 #aged employees performing well.
 
sns.barplot(x = df.ProjectCompletion, y = df.Age)
# employees who complted more projects in the ages 40-45.
 
sns.barplot(x = df.Gender, y = df.PercentSalaryHike)
    # observation:
    # male employees have got bit more hike than female employees.
    
sns.barplot(x = df.Education, y = df.Annual_income)
# employeess with vocational qualification getting high salaries than others.

# -----------------------------


# histogram for age column after outlier treatment
sns.histplot(data=df, x='Age', kde=True) 
# its looking like normally distributed.


#correlation matrix
corr = df.corr()
corr
# Correlation between different variables
corr = df.corr()
# Set up the matplotlib plot configuration
f, ax = plt.subplots(figsize=(30, 10))
# Generate a mask for upper traingle
mask = np.triu(np.ones_like(corr, dtype=bool))
# Configure a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, annot=True, mask = mask, cmap=cmap)
#DEsicion tree

#Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score,confusion_matrix

#separating dependent and independent columns 
X = df.iloc[:,0:26]  #independent columns
y = df.iloc[:,-1]    #target column

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, 0:26]))
df[df.columns[0:26]] = df_scaled[df_scaled.columns[0:26]]


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30) # 70% training and 30% test

     
# Model building for Decision Tree

clf = DecisionTreeClassifier(criterion="gini", max_depth=3)

clf.fit(X_train,y_train)
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

print(f'Train score {accuracy_score(y_train_pred,y_train)}') # 98.62
print(f'Test score {accuracy_score(y_test_pred,y_test)}')  #  98.36

# confusion matrix for performance metrics
cm = confusion_matrix(y_test, y_test_pred)
cm
# or
pd.crosstab(y_test, y_test_pred, rownames=['Actual'], colnames=['Predictions'])

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_test_pred))

# =============================================================


# ROC Curve

from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


dtc = OneVsRestClassifier(DecisionTreeClassifier())
dtc.fit(X_train, y_train)
pred = dtc.predict(X_test)
pred_prob = dtc.predict_proba(X_test)

# roc curve for classes

fpr = {}
tpr = {}
thresh = {}

n_class = 3

for i in range(n_class):
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:,i], pos_label=i)
    
# plotting

plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 0 vs Rest')
plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 1 vs Rest')
plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 2 vs Rest')
plt.title('Multiclass ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.savefig('Multiclass ROC', dpi=300)

# -----------------------

import pickle
#open the pickle file in witebyte mode
file = open("model.pkl",'wb')
#dump info to that file
pickle.dump(clf,file)
file.close()

X['Age'].unique()
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[27, 1,1,0,2,1,2,6,6,3,0,2,6,1,2,0,1,2,1,0,1,0,8,1,3,2]]))

y_pred_pickle = model.predict(X_test)
print(f'Test score {accuracy_score(y_pred_pickle,y_test)}')


