
# Feature extraction

Most feature extraction requires domain specific knowledge. Extracting useful
features from images (which can be represented as matrices of numbers) is very
different than extracting useful features from text (e.g., wikipedia
articles).

In many cases, feature extraction will require specific implementation. In
this practical session, we are going to extract features from a corpus of
scientific articles from elsevier. The original dataset is composed of
extracts of 40000 articles available as json format. We extracted a subset of
these, classified in either of two fields (MEDI, PHYS).

We are going to extract features from these series of json files as well as do
some summary statistics.

The data can be downloaded
[here](https://filesender.renater.fr/?s=download&token=7da273a2-5be7-4e90-900f-6c79a470f76f)

## Understanding the data at hand

- Start by loading the file `subset-elsevier/S000145751530083X.json`. Here's a
  code snippet to load a json file:

  ```python
      with open(filename, "rt") as f:
        article = json.load(f)
  ```

  Some fields of the json files are going to be particular relevant for our
  practical session:

    - The title of the article (`article["metadata"]["title"]`)
    - The abstract of the article (`article["abstract"]`)
    - A classification of the article into fields of research
      (`article["metadata"]["subjareas"]`)

  Print the information of those three fields for the first article.
  What can you observe about the categories of the text?

- Now looking at the entire dataset, how many articles are in the MEDI
  category? How many are in the PHYS category? And how many are in both?

## Basic feature extraction: bag-of-words

An obvious way to encode the text into features is to do a one-hot encoding of
the words in the text, i.e., create a big matrix where each row corresponds to
an article, each column to a word, and each entry of the matrix to the number
of times this word has been seen in the article.

- Let's start by estimating whether this is feasible. Just splitting the text
  by whitespace and considering "words" to be the remaining, how many unique
  words is there in this set of articles?

- Let's reduce a bit the number of "words" by lowercasing the text, removing
  punction and numerical data. Now, how many unique words is there in this set
  of article?

- Now, let's create the feature matrix X with this "reduced" set of words.

- How many words occur only in one article? Look at the distribution of number
  of articles per word (via a histogram). Do you think words occurring only in
  one article are meaningful for ML pipelines?

- Removing all words that occur only in one article.

- Do a PCA, marking in a different color the articles in the PHY category.
  Comment the plot.


## TF-IDF and sklearn's preprocessing module

As you may expect, using word count is not the best way to describe articles.
There are many improvements one can do, the most famous being probably the
**term frequency-inverse document frequency** one. This is an improvement over
the bag-of-word approach used in the previous section, that reiweights words
depending on their frequency on the rest of the corpus.

- Use sklearn's TfidfVectorizer normalize the feature matrix created
  previously.
- Perform again the PCA. Do you observe any changes?


## Predicting whether articles are part of the PHY category


- We are now going to fit a logistic regression onto this model using
  `sklearn.linear_model.LogisiticRegression`. By default, sklearn's logistic
  regression is penalized. Here, we are going to fit the unpenalized version
  of the logistic regression. Make sure you read the documentation carefully
  to fit an *unpenalized* logistic regression.

- When predicting on the data used for fitting the model (i.e., the feature
  martix X), compute the accuracy of the model: the proportion of correctly
  labeled samples. This is an estimation of the training error (the error of
  the model on the training data).

  Questions: do you think this is a good way to estimate how well the model is
  performing? Why?


  Look at the coefficient of the model, and in particular the coefficients
  associated with high coefficients. Can you conclude anything?

- Let's standardize the features. Use `sklearn.preprocessing.StandardScaler`
  to do this. Do the visualization again. Does it visually change? Why?

  Refit the model on the scaled data and estimate the training error. Does it
  change? Now look at the coefficients of the model, in particular high
  coefficients. Did they change? Why? Can you conclude anything?

## More on feature engineering, data transformation, etc

### Feature engineering

In this practical sessions, we've worked on feature extraction and engineering
of scientific articles, using fairly basic features. Here's a number of
elements you may want to investigate when doing feature engineering:


* __Encoding categorical features:__ if a K-categorical feature is not ordered
  (categorie 1 is as far to categorie 2 as to categorie 3 etc), then it must
  not be encoded by a single integer specifying the categorie. We can encode
  such feature by creating K-1 binary features encoding the belonging to k-th
  category. (see
  [link](http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features))

* __Feature binarization:__ some continuous features can gain predictive power
  when binarized. For exemple, in some prediction tasks, weekdays could be
  split into $working\ days$ and $not\ working\ days$. (see
  [link](http://scikit-learn.org/stable/modules/preprocessing.html#binarization))

* __Imputation of missing values:__ there are multiple strategies to input
  missing values when required (see
  [link](http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)).

* __Dealing with time features or other periodic features:__ when considering
  the hour of the day as a feature, we can't encode it by the an integer
  between 1 and 24 as midnigth is as close to 11pm to 1am. An easy strategy to
  encode periodic features is to apply this transformation $x \mapsto
  \sin(\frac{2\pi x}{T})$ (T is the period). In the case of the hour of the
  day, it is   $x \mapsto \sin(\frac{2\pi x}{24})$.

* __Generating new features:__ you might want to combine the existing features into new ones that seem informative to you. It can be useful for exemple, notably when working with linear models, to generate polynomial features from the original ones. You can also use external data to transform your features; for instance, if one feature is a date, adding a feature that qualifies whether the day is a working day, a weekday or a holiday can be useful.

* ...

In many practical cases, feature engineering is the key to obtaining a huge improvement in performance.

### Pre-processing data: standardization and rescaling

You might want to consider standardizing your data, or applying some other
form of transformation. `scikit-learn` has a number of pre-processing steps
that can be applied to the data: `preprocessing.MaxAbsScaler`,
`preprocessing.QuantileTransformer`, `preprocessing.RobustScaler`,
`preprocessing.StandardScaler`.

### Unsupervised projection

If your number of features is high or correlated, it may be useful to reduce
it with an unsupervised step prior to supervised steps. We have already worked
on a widly used dimentionality reduction method in `Lab 1`, the Principal
Component Analysis. There are other means of projecting data. Look at
scikit-learn's documentation to see possible options.
