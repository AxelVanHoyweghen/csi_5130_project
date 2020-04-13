#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
# %matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

#Loading the dataset

Y = pd.read_csv('./dataset/trimmed/movies_metadata_y_profit.csv', low_memory=False)
X = pd.read_csv('./dataset/trimmed/movies_metadata_x.csv', low_memory=False)
print(X.values[:, 0:X.shape[1]])
# dataset.head()
test = SelectKBest(score_func=chi2, k=10)
fit = test.fit(X.values[:, 0:X.shape[1]], Y.values[:, 0])

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

features = fit.transform(X)
# Summarize selected features
print(features[0:5,:])

# filter method

# # Using Pearson Correlation
# plt.figure(figsize=(12,10))
# cor = dataset.corr()
# # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# print(cor)

# # plt.show()

# # # Correlation with output variable
# cor_target = abs(cor["rev_cat"])
# # # Selecting highly correlated features
# relevant_features = cor_target[cor_target>0]

# relevant_features.to_csv('./stats/corr.csv')


# print(relevant_features)