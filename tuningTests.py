import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from IPython import display
from matplotlib.legend_handler import HandlerLine2D


# === Load Source Data ===
df = pd.read_csv("./data/prepped_data.csv")

# === Process the data before using ===
# Use Pandas to create dummy variables for categorical data to prevent numerical hierarchy
df = pd.get_dummies(df, columns = ["data_registry","publication_year","author_birth","gender","verb_conjugation","pre_particle","post_particle","genre"], prefix_sep="_",drop_first=True)

# LabelEncode the classes
le = LabelEncoder()
df["entry_key"] = le.fit_transform(df["entry_key"])

# === Train & Test ===
# Split data into train and test subsets 
X = df.iloc[:,3:77] # Select the features
y = df.iloc[:,1] # Select the classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # Generate the sets by random selection

# =========================================================================================================
# === Tree Number Test ===
'''
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

train_results = []
test_results = []

for estimator in n_estimators:
    rf = RandomForestClassifier(estimator, random_state=0)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    acc = accuracy_score(y_train,train_pred)
    train_results.append(acc)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    test_results.append(acc)

line1, = plt.plot(n_estimators, train_results, 'b', label="Train Acc")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test Acc")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy Score')
plt.xlabel('# of Trees')
plt.show()
'''
# =========================================================================================================
# === Max Tree Depth Test ===
'''
max_depths = np.linspace(1, 32, 32, endpoint=True)

train_results = []
test_results = []

for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    acc = accuracy_score(y_train,train_pred)
    train_results.append(acc)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    test_results.append(acc)

line1, = plt.plot(max_depths, train_results, 'b', label="Train Acc")
line2, = plt.plot(max_depths, test_results, 'r', label="Test Acc")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy Score')
plt.xlabel('Tree Depth')
plt.show()
'''
# =========================================================================================================
# === Minimum Sample Split Test ===
'''
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)

train_results = []
test_results = []

for min_samples_split in min_samples_splits:
    rf = RandomForestClassifier(min_samples_split=min_samples_split)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    acc = accuracy_score(y_train,train_pred)
    train_results.append(acc)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    test_results.append(acc)

line1, = plt.plot(min_samples_splits, train_results, 'b', label="Train Acc")
line2, = plt.plot(min_samples_splits, test_results, 'r', label="Test Acc")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy Score')
plt.xlabel('Min Samples Split')
plt.show()
'''
# =========================================================================================================
# === Minimum Samples Leaf Test ===
'''
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)

train_results = []
test_results = []

for min_samples_leaf in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    acc = accuracy_score(y_train,train_pred)
    train_results.append(acc)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    test_results.append(acc)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label="Train Acc")
line2, = plt.plot(min_samples_leafs, test_results, 'r', label="Test Acc")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy Score')
plt.xlabel('Min Samples Leafs')
plt.show()
'''
# =========================================================================================================
# === Max Features Test ===

max_features = list(range(1,74))

train_results = []
test_results = []

for max_feature in max_features:
    rf = RandomForestClassifier(max_features=max_feature)
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)

    acc = accuracy_score(y_train,train_pred)
    train_results.append(acc)

    y_pred = rf.predict(X_test)

    acc = accuracy_score(y_test,y_pred)
    test_results.append(acc)

line1, = plt.plot(max_features, train_results, 'b', label="Train Acc")
line2, = plt.plot(max_features, test_results, 'r', label="Test Acc")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy Score')
plt.xlabel('Max Features')
plt.show()
