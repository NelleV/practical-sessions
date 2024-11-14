We are now going to investigate regularized models. We are going to use the
same data as before from eslevier articles. Here, I've done some preliminary
work to extract features from 1000 data

We are going to try to predict whether an article is labeled as PHYS versus
the rest.

# Data loading

- Load the features and labels. What is the size of the data?
```python

import pandas as pd

features = pd.read_csv("data/features.csv", index_col=0)
labels = pd.read_csv("data/labels.csv", index_col=0)
```


- Reduce the number of features to the firt 700.


```python

features = features[features.columns[:700]]
```


- Split the data in a train set containing 70% of the data and a test set
  containing the remaining 30%. We use scikit-learn's
  `model_selection.train_test_split` for this.

- Scale the data using a standard scaler (`preprocessing.StandardScaler`).
  Estimate the scaler on the train data (using `fit` / `fit_transform`), and
  apply the scaling on the test data (using `transform`). Keep both scaled and
  unscaled data at hand for tests.

# Logistic regression, not regularized

- Let us train a logistic regression *without regularization* on our
  *unscaled* train set, and evaluate it on the test set using
  `metrics.balanced_accuracy_score`
- Repeat on *scaled* training data. What do you observe in terms of
  performance?
- Plot the ROC curves.

# l2-Logistic regression, regularized

- What is the role of the l2 regularization?
- What is the role of C? How does it relate do the `lambda` regularization
  parameter we have seen in class?
- Train the l2-regularized logistic regression initialized below on the scaled
  training data, and evaluate it on the scaled test set (as above). How does
  the performance evolve?

## Effect of the l2-regularization on the logistic regression coefficients.

We will now look at how the regression coefficients have evolved between the
non-regularized and the regularized versions of the logistic regression.

- Use a scatter plot to plot the coefficient of the unscaled model with the
  scaled one

  ```python
  import matplotlib.pyplot as plt

  plt.scatter(scaled_unpenalized.coef_, scaled_l2.coef_)
  ```

  Add labels with `xlabel` and `ylabel`

## Optimization of the regularization parameter. 

We will now look at the effect of the regularization paramater on the training
and testing error.

- Create a range of values to test for the parameter C. Typically, we sample
  in logspace:

  ```python
  cvalues_list = np.logspace(-5, 1, 20)
  ```

- Estimate the model with the different C values, compute the training and
  testing error for each C using balanced accuracy, and plot the value of the
  training error as a function of C, and the testing error as a function of C.
  Comment.

- We will now use a 3-fold cross-validation on the training set to optimize
  the value of C. Scikit-learn makes it really easy to use a cross-validation
  to choose a good value for C among a grid of several choices. Check
  the [GridSearchCV
  class](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV).

  Now, find value for C using GridSearchCVCLass on the training data, and
  estimate the generalization of the best model on the testing data.
  
  Tip: Use the `.best_estimator_` attribute of a `GridSearchCV`. 

  Is it the same C value as found before? Comment.

# L1-regularized logistic regression

Going back to the small dataset.
 
- What is the rol of the L1-regularized logistic regression?
- Estimate a L1-regularized logistic regression. Compare the generalization
  error with the l2 regularized element.

## Effect of the regularization on the regression coefficients.

- Plot the weights given to each feature

  ```python
  plt.scatter(range(X.shape[1]), #TODO coefficients,
	      label="logistic regression")
  plt.scatter(range(X.shape[1]), #TODO coefficients,
	      label="logistic regression")
  plt.legend()
  ```

  Add labels as appropriate.

  What do you observe? How does this differ from l2-regularization?
  How many weights are different from zero? How many features are **not** used by the l1-regularized model?

- Extract the most important "features" for this task. Do they make sense to
  you?


## Optimization of the regularization parameter.

- Fit models with varying values of C. Plot the generalization error as a
  function of C. Plot the number of features selected as a function of C.
- Plot ROC curves of the several models in the range of the ones estimated,
  including the best performing model.
- What is the best model?

## Exploring further: the multitask lasso

We can move from a single classification problem (predicting whether the
article is in a specific class, such as PHYS) to a multitask problem: this
consists in predicting jointly the membership to all the classes. Try
estimating jointly the different tasks by using
[MultiTaskLasso](https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.MultiTaskLasso.html).
Does it improve the results?
