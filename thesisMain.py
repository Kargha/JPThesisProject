import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import thesisLib

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz
from IPython import display
from matplotlib.legend_handler import HandlerLine2D

# === Variables ===
is_training = False # Report scoring on training (True) or test (False) data
create_tree_png = False # Create an image of a sample tree based on the data set (WINDOWS ONLY)
print_importance_scores = False # Display a table of all importance scores (without feature names)
print_importance_features = False # Display a table of all importance scores (with feature names)
print_importance_top5 = False # Create a graph of the Top 5 importance features

# === Load Source Data ===
df = pd.read_csv("./data/prepped_data.csv")

# === Process the data before using ===
# Use Pandas to create dummy variables for categorical data to prevent numerical hierarchy
# issues which would occur with a nominal variation
df = pd.get_dummies(df, columns = ["data_registry","publication_year","author_birth","gender","verb_conjugation","pre_particle","post_particle","genre"], prefix_sep="_", drop_first=True)

# LabelEncode the classes, turning them into nominal values
le = LabelEncoder()
df["entry_key"] = le.fit_transform(df["entry_key"])

# === Train & Test ===
# Split data into train and test subsets 
X = df.iloc[:,3:77] # Select the features
y = df.iloc[:,1] # Select the classes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # Generate the sets by random selection

regressor = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=18, min_samples_split=0.2, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Print classification report, confusion matrix, and 5-fold cross-validation score (if training set)
thesisLib.print_score(regressor, X_train, y_train, X_test, y_test, train=is_training)

# Print a table over all feature importances in column order
print(regressor.feature_importances_)


# === Utility Code ===
# Take out a sample tree (Windows Only, requires graphviz)
if create_tree_png:
    tree = regressor.estimators_[74]
    export_graphviz(tree, out_file="tree.dot", feature_names=X.volumns, rounded=True, proportion=False, precision=2, filled=True)
    from subprocess import call
    call(['C:/Program Files/Graphviz/bin/dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


# Create a list of all features and their related importance score
if print_importance_features:
    feat_importances = pd.Series(regressor.feature_importances_, index=X.columns)
    print(feat_importances.to_string())

# Create a plot over the top 5 features, ranked by score
if print_importance_top5:
    mpl.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'ms gothic',
        }
    )
    feat_importances = pd.Series(regressor.feature_importances_, index=X.columns)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.title("Top 5 Features")
    plt.show()
