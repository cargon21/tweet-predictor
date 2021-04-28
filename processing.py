#------------------------------------------------------------------------------
# - Carson Sytner
# - Honors Data Science Project
# - This program applies machine learning algorithms to tweet data from students
#   and entrepreneurs to predict who wrote a given tweet
#------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from numpy.random import binomial
from sklearn import preprocessing

np.random.seed(0)

def confusion(X_train, X_test, y_train, y_test, algorithm_type, labels):
    if algorithm_type == "rf":  
        model = RandomForestClassifier()
    else:
        model = LogisticRegression(max_iter=np.inf) # ask if bad idea
        
    model.fit(X_train, y_train)
    train_predicted = model.predict(X_test)
    
    return confusion_matrix(y_test, train_predicted, labels=labels)
    

def learn(X_train, X_test, y_train, y_test, algorithm_type, cohort):
    cols = ["Cohort", "Algorithm", "Test Score", "Train Score"]
    
    if algorithm_type == "rf":  
        model = RandomForestClassifier()
    else:
        model = LogisticRegression(max_iter=np.inf) # ask if bad idea
    
    model.fit(X_train, y_train)
    test_score =  model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)

    return pd.Series([cohort, algorithm_type, test_score, train_score], index = cols)    


data = pd.read_csv("tweet_data.csv", index_col=("id"))

data["Phase"] = "P" + data["Phase"].astype(int).astype(str)

# Kmeans Clustering
#-----------------------------------------------------------------------------
data_to_scale = data.loc[:, "Avg len":"Emoji Freq"]

scaler = MinMaxScaler().fit(data_to_scale.values) # Create a MinMaxScaler object

km = KMeans(n_clusters = 6) # Create a clustering object

# Scale the data
scaled_data = pd.DataFrame(scaler.transform(data.loc[:, "Avg len":"Emoji Freq"]),
                           columns=(data_to_scale.columns), index = data.index)

y_predicted = km.fit_predict(scaled_data)

scaled_data["g"] = y_predicted

scaled_data.plot.scatter(x = "Emoji Freq", y = "URL Freq", c = "g", cmap = "Accent")
plt.savefig("kmeans.png")
plt.clf()

# RandomForests and Logistic Regression
#-----------------------------------------------------------------------------
outcomes = pd.DataFrame(columns = ["Cohort", "Algorithm", "Test Score", "Train Score"])

# Group and calculate binomial distribution to avoid biased data
groups = data.groupby(["Phase", "Type"])
sizes = groups.size()
fractions = (sizes.min() * .7) / sizes

test  = pd.DataFrame()
train = pd.DataFrame()

for g, d in groups:
    mask = binomial(1, fractions.loc[g], d.shape[0]).astype(bool)
    train = train.append(d[mask])
    test = test.append(d[~mask])                

# X values contain the cohorts, Y values contain the statistics
y_train_t = train["Type"].reset_index(drop=True)
y_test_t = test["Type"].reset_index(drop=True)

y_train_p = train["Phase"].reset_index(drop=True)
y_test_p = test["Phase"].reset_index(drop=True)

y_train_b = y_train_t + "_" + y_train_p
y_test_b = y_test_t + "_" + y_test_p

X_train = train.loc[:, "Avg len":"Emoji Freq"].reset_index(drop=True)
X_test = test.loc[:, "Avg len":"Emoji Freq"].reset_index(drop=True)

train.groupby(["Phase", "Type"]).size()
test.groupby(["Phase", "Type"]).size()

# Apply machine learning for different groups
outcomes = outcomes.append(learn(X_train, X_test, y_train_t, y_test_t,  "rf", "by_type"), ignore_index=True)
outcomes = outcomes.append(learn(X_train, X_test, y_train_t, y_test_t, "lr", "by_type"), ignore_index=True)
outcomes = outcomes.append(learn(X_train, X_test, y_train_p, y_test_p, "rf", "by_phase"), ignore_index=True)
outcomes = outcomes.append(learn(X_train, X_test, y_train_p, y_test_p, "lr", "by_phase"), ignore_index=True)
outcomes = outcomes.append(learn(X_train, X_test, y_train_b, y_test_b, "rf", "both"), ignore_index=True)
outcomes = outcomes.append(learn(X_train, X_test, y_train_b, y_test_b, "lr", "both"), ignore_index=True)
 
# Find the confusion matrix for all the data
type_rf_cm = confusion(X_train, X_test, y_train_t, y_test_t,  "rf", ["S", "E"])
type_lr_cm = confusion(X_train, X_test, y_train_t, y_test_t,  "lr", ["S", "E"])

phase_rf_cm = confusion(X_train, X_test, y_train_p, y_test_p,  "rf", ["P0", "P1", "P2"])
phase_lr_cm = confusion(X_train, X_test, y_train_p, y_test_p,  "lr", ["P0", "P1", "P2"])

both_rf_cm = confusion(X_train, X_test, y_train_b, y_test_b,  "rf", ["S_P0", "S_P1", "S_P2", "E_P0", "E_P1", "E_P2"])
both_lr_cm = confusion(X_train, X_test, y_train_b, y_test_b,  "lr", ["S_P0", "S_P1", "S_P2", "E_P0", "E_P1", "E_P2"])
#%%-----------------------------------------------------------------------------
plt.imshow(both_lr_cm, cmap="spring")
plt.xticks(range(6), ["S_P0", "S_P1", "S_P2", "E_P0", "E_P1", "E_P2"])
plt.yticks(range(6), ["S_P0", "S_P1", "S_P2", "E_P0", "E_P1", "E_P2"])
plt.colorbar()
plt.savefig("lr.png")
plt.clf()

#%%-----------------------------------------

plt.savefig("rf.png")
plt.clf()



