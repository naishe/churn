from __future__ import division
import pandas as pd
import numpy as np

from churn_measurements import calibration
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier as RF


print "Importing data"
churn_df = pd.read_csv('data/churn.csv')

print "Formatting feature space"
# Isolate target data
churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)

# We don't need these columns
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = churn_df.drop(to_drop,axis=1)

# 'yes'/'no' has to be converted to boolean values
# NumPy converts these from boolean to 1. and 0. later
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

# Pull out features for future use
features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

print "Scaling features"
# This is important
scaler = StandardScaler()
X = scaler.fit_transform(X)

def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob

error = []
n_trees = []

for n in range(5,100):
    probs = run_prob_cv(X,y,RF,n_estimators=n)
    errors[n] = calibration(probs[:,1],y==1)

calibration_errors = pd.DataFrame({'calibration_error': error,
                                   'n_trees': n_trees})

try:
    from ggplot import *
    ggplot(calibration_errors,aes(x='n_trees',y='calibration_error')) + \
            geom_point()
except:
    print calibration_errors
