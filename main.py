import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import unittest
import math
from sklearn import metrics
from sklearn.tree import export_graphviz
import IPython, re, os, sys

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 12, 6

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

train_csv = 'train/house_prices_train.csv'
test_csv = 'train/house_prices_test.csv'

df_train = pd.read_csv(train_csv)
df_test = pd.read_csv(test_csv)

X = df_train[['OverallQual', 'GrLivArea', 'GarageCars']]
y = df_train['SalePrice']

from sklearn.metrics import mean_squared_error
from math import sqrt


def rmse(h, y):
  return sqrt(mean_squared_error(h, y))


# from sklearn.ensemble import RandomForestRegressor
#
# reg = RandomForestRegressor(n_estimators=1, max_depth=2, bootstrap=False, random_state=RANDOM_SEED)
# reg.fit(X, y)
#
# preds = reg.predict(X)
# metrics.r2_score(y, preds)
#
# rmse(preds, y)

class Node:
  def __init__(self, x, y, idxs, min_leaf=5):
    self.x = x
    self.y = y
    self.idxs = idxs
    self.idxs_native = np.array(np.arange(len(self.idxs)))
    self.min_leaf = min_leaf
    self.row_count = len(idxs)
    self.col_count = x.shape[1]
    self.val = np.mean(y[idxs])
    self.score = float('inf')
    self.find_varsplit()

  def find_varsplit(self):
    for c in range(self.col_count): self.find_better_split(c)
    if self.is_leaf: return
    x = self.split_col
    lhs = np.nonzero(x <= self.split)[0]  # lhs indexes
    rhs = np.nonzero(x > self.split)[0]  # rhs indexes
    self.lhs = Node(self.x, self.y, self.idxs[lhs])
    self.rhs = Node(self.x, self.y, self.idxs[rhs])

  def find_better_split(self, var_idx):
    x, y = self.x.values[self.idxs_native, var_idx], self.y[self.idxs]

    for r in range(self.row_count):
      lhs = x <= x[r]  # any value in x that is less or equal than this value
      rhs = x > x[r]  # any value in x that is greater than this value

      # lhs/rhs = [1,0,0,1,1,0,1,1,0,1]

      if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue

      lhs_std = y[lhs].std()
      rhs_std = y[rhs].std()
      curr_score = lhs_std * lhs.sum() + rhs_std * rhs.sum()  # weighted average

      if curr_score < self.score:
        self.var_idx = var_idx
        self.score = curr_score
        self.split = x[r]

  @property
  def split_name(self):
    return self.x.columns[self.var_idx]

  @property
  def split_col(self):
    return self.x.values[self.idxs_native, self.var_idx]

  @property
  def is_leaf(self):
    return self.score == float('inf')

  def __repr__(self):
    s = f'n: {self.n}; val:{self.val}'
    if not self.is_leaf:
      s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
    return s

  def predict(self, x):
    return np.array([self.predict_row(xi) for xi in x])

  def predict_row(self, xi):
    if self.is_leaf: return self.val
    t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
    return t.predict_row(xi)


class DecisionTreeRegressor:
  def fit(self, X, y, idxs, min_leaf=5):
    self.dtree = Node(X, y, idxs, min_leaf)
    return self

  def predict(self, X):
    return self.dtree.predict(X.values)

# Random Forest
class RandomForest():
    def __init__(self, x, y, n_trees, sample_sz, depth=10, min_leaf=5):
        self.x = x
        self.y = y
        self.sample_sz = sample_sz
        self.depth = depth
        self.min_leaf = min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        tree = DecisionTreeRegressor()
        x = self.x.iloc[idxs]
        y = self.y[idxs]

        # print(x.shape)
        # print(len(idxs))
        # print(x)
        tree.fit(x, y, idxs, self.min_leaf)
        return tree
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)

forest = RandomForest(X, y, n_trees=150, sample_sz=399, depth=5, min_leaf=5)
preds = forest.predict(X)

# regressor = DecisionTreeRegressor().fit(X, y)
# preds = regressor.predict(X)

metrics.r2_score(y, preds)
err = rmse(preds, y)

print(err)

X_test = df_test[['OverallQual', 'GrLivArea', 'GarageCars']]
# pred_test = regressor.predict(X_test)
pred_test = forest.predict(X_test)
submission = pd.DataFrame({'Id': df_test.Id, 'SalePrice': pred_test})

script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
csv_path = os.path.join(script_dir, 'data/submission.csv')

submission.to_csv(csv_path, index=False)
