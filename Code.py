import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from seaborn           import heatmap
import time
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

df= pd.read_csv('winequality-red.csv')

df.head()

df['quality'].unique()
df['quality'].value_counts()

df.isnull().sum()

df.describe()

df.info()

fig = plt.figure(figsize=(20,10))
plt.subplot(3,4,1)
sns.barplot(x='quality',y='fixed acidity',data=df)   
plt.subplot(3,4,2)
sns.barplot(x='quality',y='volatile acidity',data=df)

plt.subplot(3,4,3)
sns.barplot(x='quality',y='citric acid',data=df)

plt.subplot(3,4,4)
sns.barplot(x='quality',y='residual sugar',data=df)

plt.subplot(3,4,5)
sns.barplot(x='quality',y='chlorides',data=df)

plt.subplot(3,4,6)
sns.barplot(x='quality',y='free sulfur dioxide',data=df)

plt.subplot(3,4,7)
sns.barplot(x='quality',y='total sulfur dioxide',data=df)

plt.subplot(3,4,8)
sns.barplot(x='quality',y='density',data=df)

plt.subplot(3,4,9)
sns.barplot(x='quality',y='pH',data=df)

plt.subplot(3,4,10)
sns.barplot(x='quality',y='sulphates',data=df)

plt.subplot(3,4,11)
sns.barplot(x='quality',y='alcohol',data=df)

plt.tight_layout()
plt.savefig('output.jpg',dpi=1000)

# Data binning
ranges= (2,6,8)
groups = ['bad','good']
df['quality'] = pd.cut(df['quality'],bins=ranges,labels=groups)
df.head()

#label encoding
le = LabelEncoder()
df['quality'] = le.fit_transform(df['quality'])
df.head()

df['quality'].value_counts()

#WITHOUT BALANCING

#checking correlation
df.corr()['quality'].sort_values(ascending=False)
X = df.drop('quality',axis=1) 
y = df['quality']

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt= DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred_dt = dt.predict(X_test)
dt_acc=(accuracy_score(y_test,y_pred_dt))

#Using Random Forest
from sklearn.ensemble import RandomForestClassifier 
rf= RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
rf_acc= (accuracy_score(y_test,y_pred_rf))

# Comparing the accurcacy from two models
print(f'The accuracy score for Decision tree and Random Forest are {dt_acc}, {rf_acc}')

# hyperparameter tuning for random forest
np.random.seed(42)
start = time.time()

param_dist = {'max_depth': [3, 4,5,6,7,8,9,10],
              'bootstrap': [True],
              'max_features': ['auto', 'sqrt', 'log2'],
              'criterion': ['gini', 'entropy']}

cv_rf = GridSearchCV(rf, cv = 10,
                     param_grid=param_dist, 
                     n_jobs = 3)

cv_rf.fit(X_train, y_train)
print('Best Parameters using grid search: \n', cv_rf.best_params_)
end = time.time()
print('Time taken in grid search: {0: .2f}'.format(end - start))

# Set best parameters given by grid search 
rf.set_params(criterion = 'entropy',
                  max_features = 'log2', 
                  max_depth = 10)

rf.set_params(warm_start=True, 
                  oob_score=True)

min_estimators = 102
max_estimators = 1000

error_rate = {}

for i in range(min_estimators, max_estimators + 1):
    rf.set_params(n_estimators=i)
    rf.fit(X_train,y_train)

    oob_error = 1 - rf.oob_score_
    error_rate[i] = oob_error

# Convert dictionary to a pandas series for easy plotting 
oob_series = pd.Series(error_rate)

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_facecolor('#fafafa')

oob_series.plot(kind='line',color = 'red')
plt.axhline(0.097, color='#875FDB',linestyle='--')
plt.axhline(0.0925, color='#875FDB',linestyle='--')
plt.xlabel('n_estimators')
plt.ylabel('OOB Error Rate')
plt.title('OOB Error Rate Across various Forest sizes \n(From 102 to 1000 trees)')

print('OOB Error rate for 800 trees is: {0:.5f}'.format(oob_series[800]))
# Refine the tree via OOB Output
rf.set_params(n_estimators=800,
                  bootstrap = True,
                  warm_start=False, 
                  oob_score=False)
rf.fit(X_train, y_train)

#accuracy
rf.fit(X_train,y_train)
y_pred_rf = rf.predict(X_test)
rf_acc= (accuracy_score(y_test,y_pred_rf))
print(f'the accuracy is {rf_acc}')

#constructing confusion matrix
def create_conf_mat(y_test, y_pred_rf):
    """Function returns confusion matrix comparing two arrays"""
    if (len(y_test.shape) != len( y_pred_rf.shape) == 1):
        return print('Arrays entered are not 1-D.\nPlease enter the correctly sized sets.')
    elif (y_test.shape != y_pred_rf.shape):
        return print('Number of values inside the Arrays are not equal to each other.\nPlease make sure the array has the same number of instances.')
    else:
        # Set Metrics
        test_crosstb_comp = pd.crosstab(index = y_test,
                                        columns = y_pred_rf)
        # Changed for Future deprecation of as_matrix
        test_crosstb = test_crosstb_comp.values
        return test_crosstb
conf_mat = create_conf_mat(y_test, y_pred_rf)
sns.heatmap(conf_mat, annot=True, fmt='d', cbar=False)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Actual vs. Predicted Confusion Matrix')
plt.show()

#plotting AUC curve
from sklearn.metrics import roc_curve, auc
predictions_prob = rf.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions_prob,pos_label=1)
auc_rf = auc(fpr_rf, tpr_rf)
def plot_roc_curve(fpr, tpr, auc, estimator, xlim=None, ylim=None):
    
    my_estimators = {'rf': ['Random Forest', 'red']}

    try:
        plot_title = my_estimators[estimator][0]
        color_value = my_estimators[estimator][1]
    except KeyError as e:
        print("'{0}' does not correspond with the appropriate key inside the estimators dictionary. \
\nPlease refer to function to check `my_estimators` dictionary.".format(estimator))
        raise

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#fafafa')

    plt.plot(fpr, tpr,
             color=color_value,
             linewidth=1)
    plt.title('ROC Curve For {0} (AUC = {1: 0.3f})'\
              .format(plot_title, auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Add Diagonal line
    plt.plot([0, 0], [1, 0], 'k--', lw=2, color = 'black')
    plt.plot([1, 0], [1, 1], 'k--', lw=2, color = 'black')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    plt.close()
    plot_roc_curve(fpr_rf, tpr_rf, auc_rf, 'rf',
               xlim=(-0.01, 1.05), 
               ylim=(0.001, 1.05))

#Classification report
print(classification_report(y_test,y_pred_rf))

# BALANCING DATASET
good_quality =df[df['quality']==1]
bad_quality = df[df['quality']==0]

bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:len(good_quality)]

new_df = pd.concat([good_quality,bad_quality])
new_df = new_df.sample(frac=1)
new_df
new_df['quality'].value_counts()

X = new_df.drop('quality',axis=1) 
y = new_df['quality']

# splitting the datset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

param = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000]}

grid_rf = GridSearchCV(RandomForestClassifier(),param,scoring='accuracy',cv=10,)
grid_rf.fit(X_train, y_train)

print('Best parameters --> ', grid_rf.best_params_)

pred = grid_rf.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))