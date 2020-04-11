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

#Loading the dataset
dataset = pd.read_csv('./dataset/trimmed/movies_metadata.csv', low_memory=False)


Y = dataset["rev_cat"]
X = dataset.drop("rev_cat", 1)

# dataset.head()

# filter method

# Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
print("showing results")
plt.savefig('./stats/correlation.png')

plt.show()

# # Correlation with output variable
# cor_target = abs(cor["rev_cat"])
# # Selecting highly correlated features
# relevant_features = cor_target[cor_target>0.5]
# relevant_features