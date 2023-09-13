# credit-risk-classification

Module 20 challenge on supervised machine learning.   

## Overview of the Analysis

This challenge used supervised machine learning to train and evaluate two different models designed to predict loan risk.  The project was designed for a "peer-to-peer" lending company, and attempted to identify borrowers who were, and were not, credit-worthy.  This should assist the company in making lending decisions, or perhaps purchasing decisions if it is considering purchasing a set of loans or lending to a peer company that holds the set of loans..

The models were developed using Python, jupyter notebooks, and the scikit learn Python package.  Each model used logistic regression to classify loans as either healthy (0), or at risk (1).

The code began by reading a csv file of borrower data provided by the lending company into a Pandas Dataframe.  The instructions were inconsistent as to whether the data file, called 
"lending_data.csv" should be placed in the main Credit_Risk subfolder with the jupyter notebook, or in a separate subfolder within that folder called "Resources."  To be consistent with the structure of other challenges, and to keep the data separate from the code, this project used the data stored in the separate subfolder.  The code expects that the jupyter notebook will be started from the Credit_Risk subfolder where the script is stored.

The available data for each borrower included the loan size, the interest_rate, the 	borrower's income and debt-to-income ratio, the number of accounts the borrower had, the borrower's total debt, and the loan status.  The loan status (healthy or at-risk) was used as the "target" or dependent variable in the logistic regression analysis, and all of the other fields were used as the features from which to train the model to predict a binary outcome of zero for healthy or one for at-risk loans.  

The data were first divided into training and testing data sets using scikit learn's "train_test_split" function.  A logistic regression model was then created using scikit learn's "LogisticRegression" function, and fit to the training data.  A set of predictions was made using the testing feature data and the fitted model using the model's "predict" function.  A balanced accuracy score (again from scikit learn) was then calculated using the target training data and the target predictions.   A confusion matrix (another scikit learn function) was generated from the training data and the training target predictions.  Finally, scikit learn's "classification_ report" function was used to create a training report from the target training data and the target predictions.

The results of the logistic regression were pretty good, because the model was excellent at identifying healthy loans, but the model was not great at identifying high risk loans.  See results section, below.  Examining the value_counts function for the raw data target values, it was obvious that the sample data set is very unbalanced, with approximately 75,036 healthy loans and 2,500 at-risk loans.  Therefore, scikit's "imbalanced-learn" package was used to resample the data using the "RandomOverSampler" method with one random state.  The resampled data was then split into training and testing datasets using the train-test-split function, and all analysis steps completed with the resampled data as was done with the raw data, including creating a logistic regression model, making predictions, and generating the balanced-accuracy report, the confusion matrix, and the training report

## Results

The results for each model were as follows:

* Machine Learning Model 1 (raw data):

    * Accuracy --  Balanced accuracy score of 95.2%

    * Precision scores --  
       *  1.00 for healthy loans
       *  0.86 for at risk loans
       *  0.93 macro average
       *  0.99 weighted average

    * Recall scores --
       *  1.00 for healthy loans
       *  0.90 for at risk loans
       *  0.95 macro average
       *  0.99 weighted average

* Machine Learning Model 2 (random oversampled data):

    * Balanced accuracy score --  99.5%

    * Precision scores --  
       *  0.99 for healthy loans
       *  0.86 for at risk loans
       *  0.93 macro average
       *  0.99 weighted average

    * Recall scores --
       *  0.99 for healthy loans
       *  0.90 for at risk loans
       *  0.99 macro average
       *  0.99 weighted average

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

The logistic regression model did an outstanding job on predicting the healthy loans in the raw training data.  It predicted with perfect precision and recall.   It did a quite good job of predicting the risky loans, but did nowhere near as well either in terms of precision or recall, with its worst score, .86, on precision for high risk loans.  The model did about 4 percent better at recall for high risk loans than at precision for those loans.   Not identifying 14% of the high risk loans might be an acceptable business model, depending on how high risk is defined and the amounts of those loans, but it does leave quite a bit of room for poor business decisions.   Falsely identifying 10% of the loans as healthy when they are not also could be a serious problem, or might be an acceptable business risk.

Overall, the logistic regression model fit with the oversampled data did a vastly much better job (9% to 13% improvement) at predicting the high risk loans, both for precision and recall. It did slightly worse at identifying healthy loans, by one percent, for both precision and recall, compared to the raw unbalanced data before resampling.  It is probably well-worth This slight reduction in identifying the healthy loans in order to achieve the 13% increase in ability to identify the high risk loans.  In particular, reducing by 9% falsely identifying risky loans as healthy seems well worth the slight loss of accuracy in identifying all healthy loans.  If the lender starts out being slightly more cautious with a few healthy loans, that is probably not a large business cost.  The lender can do a little more in-depth investigation of a suspected high risk loan and discover that it is actually healthy, with the only cost being the presumably short time spent on the investigation.  The resampling also evened out the differences in precision and recall, so that both are now equally good for the high risk loans.

Because any set of loan data is likely to be unbalanced, the lender should use the over resampling model for future data sets.   The results of 99% accuracy, precision, recall for all loan types seem really good, and the binary classification that the lender requires is perfect for a logistic regression model.   While there are some drawbacks to over resampling as a means to handle unbalanced data as far as outlier cases, it is simple and effective for the vast majority of the cases, and has demonstrated its usefulness during this examination.  As there likley will always be only a small number of at risk loans, so the data sets will always be unbalanced, using model 2 with the overresampler will provide highly accurate and reliable predictions given the lender's particular needs and the small number of features involved.  This model is unlikely to miss many outlier cases.  As the set of features being examined is relatively simple and small, the random overresampling is unlikely to produce an inaccurately weighted resampled dataset from a random repetition of a very unusual borrower profile.
