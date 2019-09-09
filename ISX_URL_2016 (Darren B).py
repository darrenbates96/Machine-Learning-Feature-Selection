#!/usr/bin/env python
# coding: utf-8

# In[3]:



## ISCXURL2016 ##

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# In[4]:


### Read in the dataset ###

df_raw = pd.read_csv('All.csv', low_memory = False)


# In[5]:



## SOME GENERAL CLEANING AND CREATION OF BINARY COLUMN ##

### Cleaning column names ### 
df_raw.columns = df_raw.columns.str.strip()
df_raw.columns = df_raw.columns.str.replace(' ', '_')
df_raw.columns = df_raw.columns.str.replace('.', '_')


# In[6]:


### Clean Attack Labels ###
df_raw.rename(columns={'URL_Type_obf_Type':'Label'}, inplace=True)
df_raw['Label'] = df_raw['Label'].str.title()


# In[7]:


### Creation of binary column for normal and attack flows ###
Traffic_Class = []

for x in df_raw['Label']:
    if x == 'Benign':
        Traffic_Class.append(0)
    else:
        Traffic_Class.append(1)

df_raw['Traffic_Class'] = Traffic_Class

#Value count below shows that there are 28926 attack flows within the full dataset
df_raw['Traffic_Class'].value_counts()


# In[8]:



## EDA ##

df_raw.info()


# In[9]:


### Value counts for different attack types ###

df_raw.Label.value_counts()


# In[10]:


### Check dataset for missing values ###

df_raw.isnull().sum().sum()
#Thus there are 19183 NaN's in the dataset


# In[11]:


#Only other 'object' variable is argPathRatio

df_raw.argPathRatio.value_counts().head()


# In[12]:


df_raw.describe()


# In[13]:


### argPathRatio contains 'Infinity' ###
count = 0 
for x in df_raw.argPathRatio:
    if x == 'Infinity':
        count += 1

count


# In[14]:


### Replace 'Infinity' with NaN (there is only 10) ###

df_raw = df_raw.replace('Infinity', np.nan)
df_raw.isnull().sum().sum()


# In[15]:



## IMPUTATION (Remove NaN's)##

#Create new dataframe to impute on that excludes the object variable 'Label'
df_imp = df_raw.drop(df_raw.columns[[79, 80]], axis=1)
#Change object variable 'argPthRatio' to numeric 
df_imp['argPathRatio'] = pd.to_numeric(df_raw['argPathRatio'])


# In[16]:


#Impute missing data (encoded as NaN's above)
imp = SimpleImputer(strategy = 'mean')
imp_out = imp.fit_transform(df_imp)

#Restructuring Imputation output back into a dataframe
colnames = df_imp.columns
df_imp_out = pd.DataFrame(imp_out, columns = colnames)


# In[17]:



## STANDARDIZATION (SCALING) ##

#Reset the index to avoid previously encountered error
df_imp_out = df_imp_out.reset_index()

#Convert all dtypes to float64 before scaling to avoid "DataConversionWarning"
for convert in df_imp_out.columns:
    df_imp_out[convert] = df_imp_out[convert].astype(float)

scaler = MinMaxScaler()

for column in df_imp_out.columns:
    df_imp_out[column] = scaler.fit_transform(df_imp_out[column].values.reshape(-1,1))

#Re-check new dataset for missing values
df_imp_out.isnull().sum().sum()    


# In[18]:


### Reincorporate previously removed 'Label' ###

df_imp_out['Label'] = df_raw['Label']
df_imp_out['Traffic_Class'] = df_raw['Traffic_Class']
col_order = df_raw.columns
df = df_imp_out[col_order]


# In[19]:



## MACHINE LEARNING ALGORITHMS ##

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from time import time

### Split the dataset ###

X = df.iloc[:,0:79]
y = df[['Traffic_Class']]


# In[20]:


### Split dataset into training and test data ###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42)


# In[21]:


print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)


# In[82]:


## Create Accuracy Matrix ##

acc_mat = [['KNN', 'LogReg', 'Random Forest', 'Decision Tree', 'KMeans', 
            'KMeans (Unsupervised)', 'SVM', 'ExtraTreeClassifier', 
            'GradientBoostingClassifier', 'XGBoost'], [], [], [], [], []]


# In[83]:


### KNearest Neighbor ###

t0 = time()
knn = KNeighborsClassifier()
knn.fit(X_train, y_train.values.ravel())
y_pred = knn.predict(X_test)
print ("Standard KNN accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[84]:


### Logistic Regression ###

t0 = time()
logreg = LogisticRegression(solver = 'liblinear')
logreg.fit(X_train, y_train.values.ravel())
y_pred = logreg.predict(X_test)
print ("Standard Logistic Regression accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[85]:


### Random Forest ###

t0 = time()
rf = RandomForestClassifier(n_estimators = 100) #Was getting deprecation warning.
rf.fit(X_train, y_train.values.ravel())
y_pred = rf.predict(X_test)
print ("Standard Random Forest accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[86]:


### Decision Trees ###

t0 = time()
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train.values.ravel())
y_pred = tree.predict(X_test)
print ("Standard Decision Tree accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[87]:


### KMeans ###

t0 = time()
kmc = KMeans()
kmc.fit(X_train, y_train.values.ravel())
y_pred = kmc.predict(X_test)
print ("Standard KMeans accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[88]:


### KMeans (Unsupervised) ###

t0 = time()
kmc = KMeans()
kmc.fit(X_train)
y_pred = kmc.predict(X_test)
print ("Standard (unsupervised) KMeans accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[89]:


### Support Vector Machine ###

t0 = time()
svm = SVC(gamma = 'scale') #Was getting deprecation warning.
svm.fit(X_train, y_train.values.ravel())
y_pred = svm.predict(X_test)
print ("Support Vector Machine accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[90]:



## FEATURE SELECTION ##

from sklearn.feature_selection import RFECV


# In[91]:


### RECURSIVE FEATURE ELIMINATION ###

#Append to acc_mat to fill RFE gaps
acc_mat[2].append(np.nan)

### With Logistic Regression ###

t0 = time()
rfecv_lr = RFECV(LogisticRegression(solver = 'liblinear'), cv = 5, scoring = 'accuracy') #Specified cv because kept getting future warning
rfecv_lr.fit(X_train, y_train.values.ravel())
pred = rfecv_lr.predict(X_test)
print("Number of features after elimination:", rfecv_lr.n_features_)
print("Training Accuracy:", rfecv_lr.score(X_train, y_train))
print("Model prediction Accuracy:", accuracy_score(y_test, pred))
print ("Duration:", time() - t0)

acc_mat[2].append(accuracy_score(y_test, y_pred))


# In[92]:


### With Random Forest ###

t0 = time()
rfecv_rf = RFECV(RandomForestClassifier(n_estimators = 100), cv = 5, scoring = 'accuracy') #Specified cv because kept getting future warning
rfecv_rf.fit(X_train, y_train.values.ravel())
pred = rfecv_rf.predict(X_test)
print("Number of features after elimination:", rfecv_rf.n_features_)
print("Training Accuracy:", rfecv_rf.score(X_train, y_train))
print("Model prediction Accuracy:", accuracy_score(y_test, pred))
print("Duration:", time() - t0) 

acc_mat[2].append(accuracy_score(y_test, y_pred))


# In[93]:


### With Decision Tree ###

t0 = time()
rfecv_dt = RFECV(DecisionTreeClassifier(), cv = 5, scoring = 'accuracy') #Specified cv because kept getting future warning
rfecv_dt.fit(X_train, y_train.values.ravel())
pred = rfecv_dt.predict(X_test)
print("Number of features after elimination:", rfecv_dt.n_features_)
print("Training Accuracy:", rfecv_dt.score(X_train, y_train))
print("Model prediction Accuracy:", accuracy_score(y_test, pred))
print("Duration:", time() - t0)  

acc_mat[2].append(accuracy_score(y_test, y_pred))


# In[94]:


#Append to acc_mat to fill RFE gaps
acc_mat[2].append(np.nan)
acc_mat[2].append(np.nan)
acc_mat[2].append(np.nan)
acc_mat[2].append(np.nan)
acc_mat[2].append(np.nan)
acc_mat[2].append(np.nan)


# In[95]:


get_ipython().run_line_magic('matplotlib', 'inline')

### PRINCIPAL COMPONENT ANALYSIS ###

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Fit PCA algorithm to data 
pca = PCA().fit(X)

#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Pulsar Dataset Explained Variance')
plt.show()

#Plot below shows that 30 components used will preserve somewhere between
#90 and 99% of the total varience of the data


# In[96]:


pca = PCA(n_components = 30) #Worked out/justified above
pc_X_train = pca.fit_transform(X_train)
pc_X_test = pca.fit_transform(X_test)

print("PCA X_train dimensions:", pc_X_train.shape)
print("PCA X_test dimensions:", pc_X_test.shape)


# In[97]:


### With KNearest Neighbor ###

t0 = time()
pca_knn = knn = KNeighborsClassifier()
pca_knn.fit(pc_X_train, y_train.values.ravel())
y_pred = pca_knn.predict(pc_X_test)
print ("KNN accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[98]:


### With Logistic Regression ###

t0 = time()
pca_logreg = LogisticRegression(solver = 'liblinear')
pca_logreg.fit(pc_X_train, y_train.values.ravel())
y_pred = pca_logreg.predict(pc_X_test)
print ("Logistic Regression accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[99]:


### With Random Forest ###

t0 = time()
pca_rf = RandomForestClassifier(n_estimators = 100) #Was getting deprecation warning.
pca_rf.fit(pc_X_train, y_train.values.ravel())
y_pred = pca_rf.predict(pc_X_test)
print ("Random Forest accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[100]:


### With Decision Trees ###

t0 = time()
pca_tree = DecisionTreeClassifier()
pca_tree.fit(pc_X_train, y_train.values.ravel())
y_pred = pca_tree.predict(pc_X_test)
print ("Decision Tree accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[101]:


### With KMeans ###

t0 = time()
pca_kmc = KMeans()
pca_kmc.fit(pc_X_train, y_train.values.ravel())
y_pred = pca_kmc.predict(pc_X_test)
print ("KMeans accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[102]:


### With KMeans (Unsupervised) ###

t0 = time()
pca_kmc = KMeans()
pca_kmc.fit(pc_X_train)
y_pred = pca_kmc.predict(pc_X_test)
print ("KMeans (unsupervised) accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[103]:


### With Support Vector Machine ###

t0 = time()
pca_svm = SVC(gamma = 'scale') #Was getting deprecation warning.
pca_svm.fit(pc_X_train, y_train.values.ravel())
y_pred = pca_svm.predict(pc_X_test)
print ("Support Vector Machine accuracy with PCA:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[3].append(accuracy_score(y_test, y_pred))


# In[104]:


#Fill gaps in accuracy matrix 
acc_mat[3].append(np.nan)
acc_mat[3].append(np.nan)
acc_mat[3].append(np.nan)


# In[105]:


### T-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING ###

from sklearn.manifold import TSNE

#Important to note, I have used the pc_X_train and pc_X_test as inputs,
#which were created above using PCA...

#This is because in the t-SNE documentation it is advised that 
#dimentionality of a dataset is reduced to 50 or lower in order to 
#suppress noise. It also explicitly recommends using PCA to do so...

tsne = TSNE() #Might need to specify n_components 
tsne_X_train = tsne.fit_transform(pc_X_train)
tsne_X_test = tsne.fit_transform(pc_X_test)

print("t-SNE X_train dimensions:", tsne_X_train.shape)
print("t-SNE X_test dimensions:", tsne_X_test.shape)


# In[106]:


### With KNearest Neighbor ###

t0 = time()
tsne_knn = KNeighborsClassifier()
tsne_knn.fit(tsne_X_train, y_train.values.ravel())
y_pred = tsne_knn.predict(tsne_X_test)
print ("KNN accuracy with t_SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[107]:


### With Logistic Regression ###

t0 = time()
tsne_logreg = LogisticRegression(solver = 'liblinear')
tsne_logreg.fit(tsne_X_train, y_train.values.ravel())
y_pred = tsne_logreg.predict(tsne_X_test)
print ("Logistic Regression accuracy with t-SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[108]:


### With Random Forest ###

t0 = time()
tsne_rf = RandomForestClassifier(n_estimators = 100) #Was getting deprecation warning.
tsne_rf.fit(tsne_X_train, y_train.values.ravel())
y_pred = tsne_rf.predict(tsne_X_test)
print ("Random Forest accuracy with t-SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[109]:


### With Decision Trees ###

t0 = time()
tsne_tree = DecisionTreeClassifier()
tsne_tree.fit(tsne_X_train, y_train.values.ravel())
y_pred = tsne_tree.predict(tsne_X_test)
print ("Decision Tree accuracy with t-SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[110]:


### With KMeans ###

t0 = time()
tsne_kmc = KMeans()
tsne_kmc.fit(tsne_X_train, y_train.values.ravel())
y_pred = tsne_kmc.predict(tsne_X_test)
print ("KMeans accuracy with t-SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[111]:


### With KMeans (Unsupervised) ###

t0 = time()
tsne_kmc = KMeans()
tsne_kmc.fit(tsne_X_train)
y_pred = tsne_kmc.predict(tsne_X_test)
print ("KMeans (unsupervised) accuracy with t-SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[112]:


### With Support Vector Machine ###

t0 = time()
tsne_svm = SVC(gamma = 'scale') #Was getting deprecation warning.
tsne_svm.fit(tsne_X_train, y_train.values.ravel())
y_pred = tsne_svm.predict(tsne_X_test)
print ("Support Vector Machine accuracy with t-SNE:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[4].append(accuracy_score(y_test, y_pred))


# In[113]:


#Fill gaps in accuracy matrix 
acc_mat[4].append(np.nan)
acc_mat[4].append(np.nan)
acc_mat[4].append(np.nan)


# In[114]:


### ENSAMBLE (FEATURE IMPORTANCE) ###

### ExtraTreeClassifier ###

from sklearn.ensemble import ExtraTreesClassifier

t0 = time()
et = ExtraTreesClassifier(n_estimators = 100) #To "Future Warning" I was getting.
et.fit(X_train, y_train.values.ravel())
y_pred = et.predict(X_test)
print ("ExtraTreeClassifier accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[115]:


### ADABoost Classifier ###

from sklearn.ensemble import AdaBoostClassifier 
from sklearn.metrics import roc_auc_score

#ADABoost requires a base estimator to be provided...

#Fill gap for missing KNN in accuracy matrix
acc_mat[5].append(np.nan)


# In[116]:


### ADABoost With Logistic Regression ###
logreg = LogisticRegression(solver = 'liblinear')

t0 = time()
ada_logreg = AdaBoostClassifier(base_estimator = logreg)
ada_logreg.fit(X_train, y_train.values.ravel())
y_pred = ada_logreg.predict(X_test)
print ("ADABoost with Logistic Regression accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[5].append(accuracy_score(y_test, y_pred))


# In[117]:


### ADABoost With Random Forest ###
rf = RandomForestClassifier(n_estimators = 100)

t0 = time()
ada_rf = AdaBoostClassifier(base_estimator = rf)
ada_rf.fit(X_train, y_train.values.ravel())
y_pred = ada_rf.predict(X_test)
print ("ADABoost with Random Forest accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[5].append(accuracy_score(y_test, y_pred))


# In[118]:


### ADABoost With Decision Trees ###
dt = DecisionTreeClassifier()

t0 = time()
ada_dt = AdaBoostClassifier(base_estimator = dt)
ada_dt.fit(X_train, y_train.values.ravel())
y_pred = ada_dt.predict(X_test)
print ("ADABoost with Decision Tree accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[5].append(accuracy_score(y_test, y_pred))


# In[119]:


### ADABoost With KMeans ###
kmc = KMeans()

t0 = time()
ada_kmc = AdaBoostClassifier(base_estimator = dt)
ada_kmc.fit(X_train, y_train.values.ravel())
y_pred = ada_kmc.predict(X_test)
print ("ADABoost with KMeans accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[5].append(accuracy_score(y_test, y_pred))
acc_mat[5].append(np.nan)


# In[120]:


### ADABoost With Support Vector Machines ###
svm = SVC(gamma = 'scale')

t0 = time()
ada_svm = AdaBoostClassifier(base_estimator = dt)
ada_svm.fit(X_train, y_train.values.ravel())
y_pred = ada_svm.predict(X_test)
print ("ADABoost with Support Vector Machine accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[5].append(accuracy_score(y_test, y_pred))


# In[ ]:


#Fill gaps in accuracy matrix
acc_mat[5].append(np.nan)
acc_mat[5].append(np.nan)
acc_mat[5].append(np.nan)


# In[121]:


### GradientBoostingClassifier ###

from sklearn.ensemble import GradientBoostingClassifier

t0 = time()
gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train.values.ravel())
y_pred = gbc.predict(X_test)
print ("Gradient Boosting Classifier accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[27]:


### XGBoost ###

import xgboost as xgb

t0 = time()
xgboost = xgb.XGBClassifier()
xgboost.fit(X_train, y_train.values.ravel())
y_pred = xgboost.predict(X_test)
print ("XGBoost accuracy:", accuracy_score(y_test, y_pred))
print ("Duration:", time() - t0)

# acc_mat[1].append(accuracy_score(y_test, y_pred))


# In[122]:


#Convert accuracy matrix list into DataFrame
accuracy_matrix = pd.DataFrame(acc_mat)
accuracy_matrix = accuracy_matrix.transpose()
accuracy_matrix.columns = ['Algorithm', 'Baseline', 'RFE', 'PCA', 't-SNE', 'ADABoost']

accuracy_matrix

