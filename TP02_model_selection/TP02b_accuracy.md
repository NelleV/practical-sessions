# Accuracy, F1-score, ROC curves

In this practical session, we are going to implement several metrics to assess
how well a method performs in a classification setting.

The file `data/predictions.csv` contains three columns. The first column
contains the true labels. The second column corresponds to the predictions.
The third column corresponds to the probability of being True returned by the
model.

1. Compute the accuracy of the model. What is the minimum and maximum values
   of the accuracy? How well does the model perform in terms of accuracy?

2. Using the probabilities of belonging to the positive class, plot the ROC
   curve and the precision recall.

3. Compute other metrics (F1-score, Mathews Correlation Coefficient)

4. Can you conclude anything from this ensemble of results?
