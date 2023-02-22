# ECO 395M: Exercises 2

## Group Members - Alina Khindanova, Anvit Sachdev, Shreya Kamble

## Problem 1: Saratoga house prices

Dataset: SaratogaHouses

First, we will convert the data into numeric form by doing label
encodng.

Next, we split the data into training and test data. The training data
comprises of 80% of overall data, and test data comprises of remaining
40% of the data.

Next, we standardize the data to account for large differences.

Now we start with creating the models.To evaluate the models, we compare
the out-of-sample mean-squared error.  
We will first check the performance of the “medium” model that we
considered in class.

    ## [1] "The out-of-sample Root Mean-Squared Error of the medium model that we considered in class is: 0.688741751724005"

*Best Linear Model*  
We create the best linear model by removing the features fireplaces,
pctCollege, fuel, sewer, waterfront, centralAir, and landValue. We also
create a new feature given by diving bathrooms/bedrooms; that measures
the proportion of bathrooms available per bathroom for the house.
Finally since the landValue of the house is a big factor in determing
the price, so we scale all these features by landValue.

    ## [1] "The out-of-sample Root Mean-Squared Error of the best linear model is: 0.623015130834669"

We can see that the out-of-sample Root Mean-Squared Error of the best
linear model is lower than the the medium model that we considered in
class. Thus it outperforms the medium model that we considered in
class.  
*The best K-nearest-neighbor regression model*  
We form the K-nearest-neighbor regression model by including the same
features as out best linear model, except that the model that we now use
is KNN regression model.  
![](Data-Mining-Assignment-2_files/figure-markdown_strict/KNN-1.png)

    ## [1] "The optimal value of K is: 13"

    ## [1] "The Root Mean Squared Error of the best K-nearest-neighbor regression model for the optimal value of K on test set is: 0.600410376500249"

We can see that the out-of-sample Root Mean-Squared Error of the best
K-nearest-neighbor regression model is lower than the best linear model.
Thus it outperforms the best linear model.  
In this model we try to emphasize on the most effective factors on
property prices. We observe that the prices depend more on the factors
like lot size,age,bedrooms, bathrooms, new construction, and living
area. We believe that all these variables depend on the land value, so
we took that under consideration. We came up with this model because
these can be considered as most common factors on pricing strategy. Also
land value plays an important role in determining the pricing so we have
included that as well in our predictive model in how each effect of
variables would change depending on the value of the property land. We
can predict the price of property house if we know only these data ,
this can be considered as factors having good price estimation. Adding
other variables might affect our model. So the tax authority could use
the data to know whether the market price is overvalued or undervalued
and use these information to create the best pricing estimation for
taxing purposes.  
\## Problem 2: Classification and retrospective sampling

Dataset: german\_credit.csv

![](Data-Mining-Assignment-2_files/figure-markdown_strict/unnamed-chunk-1-1.png)

Figure 1. A bar plot of default probability by credit history

We build a logistic regression model for predicting default probability,
using the variables duration + amount + installment + age + history +
purpose + foreign. The results shows that the probability of default is
higher for a person with good history in comparison to a person with
poor or terrible history. This result is counter intuitive, and we think
that this data set is not appropriate for building a predictive model of
defaults. That is because the sample is biased, and we could use
randomized sample to get correct model.

## Problem 3: Children and hotel reservations

Dataset: hotels\_dev.csv and hotels\_val.csv

First, we will convert the data into numeric form by doing label
encodng.

Next, we check the number of data points in each class (i.e., children=0
and children=1) in the data.

    ## 
    ##     0     1 
    ## 41365  3635

We can see that the data is imbalanced. It comprises of more than 90%
data points that have zero children.  
To resolve this problem, I formed training set with 6000 data points
with zero children, and 3000 data points with one children. So the
training set has two-thirds of the data points with zero children, and
one-third of the data points with at least one children. The test data
is comprised of 2000 data points with zero children, and 650 data points
with one children.

Now we start with creating the models.  
To evaluate the models, we calculate the out-sample confusion matrix,
accuracy, precision, recall and F1 Score, which are given as:-  
Accuracy = (True Positive+True Negative)/(True Positive+False
Positive+True Negative+False Negative)  
Precision = True Positive / (True Positive+False Positive)  
Recall = True Positive / (True Positive+False Negative)  
F1 Score = (2 \* Precision \* Recall)/(Precision + Recall)  
The baseline 1 model is the logistic regression model with
market\_segment, adults, customer\_type, and is\_repeated\_guest
variables as features.

Let’s look at the in-sample and out-sample performance of baseline 1
model.

    ## [1] "The in-sample confusion matrix of baseline 1 model is:-"

    ##    yhat
    ## y      0    1
    ##   0 5805  195
    ##   1 2932   68

    ## [1] "The in-sample accuracy of baseline 1 model is: 0.652555555555556"

    ## [1] "The out-of-sample confusion matrix of baseline 1 model is:-"

    ##    yhat
    ## y      0    1
    ##   0 1934   66
    ##   1  617   18

    ## [1] "The out-of-sample accuracy of baseline 1 model is: 0.740796963946869"

    ## [1] "The out-of-sample precision of baseline 1 model is: 0.214285714285714"

    ## [1] "The out-of-sample recall of baseline 1 model is: 0.0283464566929134"

    ## [1] "The out-of-sample F1 Score of baseline 1 model is: 0.0500695410292072"

We can see from the above result that the baseline 1 model has a decent
accuracy but very low precision, recall and F1 Score. This is because
this model is unable to accurately predict the data points with at least
one children.

The baseline 2 model is the logistic regression model with that uses all
the possible predictors except the arrival\_date variable. Let’s look at
the in-sample and out-sample performance of baseline 2 model.

    ## [1] "The in-sample confusion matrix of baseline 2 model is:-"

    ##    yhat
    ## y      0    1
    ##   0 5347  653
    ##   1 1417 1583

    ## [1] "The in-sample accuracy of baseline 2 model is: 0.77"

    ## [1] "The out-of-sample confusion matrix of baseline 2 model is:-"

    ##    yhat
    ## y      0    1
    ##   0 1766  234
    ##   1  310  325

    ## [1] "The out-of-sample accuracy of baseline 2 model is: 0.793548387096774"

    ## [1] "The out-of-sample precision of baseline 2 model is: 0.581395348837209"

    ## [1] "The out-of-sample recall of baseline 2 model is: 0.511811023622047"

    ## [1] "The out-of-sample F1 Score of baseline 2 model is: 0.544388609715243"

We can see that the baseline 2 model performs better than baseline 1
model.

    ## [1] "Its absolute improvement in terms of out-of-sample accuracy is approximately 0.0527514231499051"

    ## [1] "Its relative improvement in terms of out-of-sample accuracy is approximately 1.07120901639344"

    ## [1] "Its absolute improvement in terms of out-of-sample F1 Score is approximately 0.494319068686036"

    ## [1] "Its relative improvement in terms of out-of-sample F1 Score is approximately 10.8726502884794"

To create the best linear model, we derive a new feature (called as
is\_holiday) from the arrival\_date variable. This feature is a dummy
variable that takes the value 1 when the arrival date happens during
school summer break (i.e, during May, June, July and August); and
Christmas break (i.e, from Dec 25 to Dec 31). It makes sense to
introduce this feature as school going children usually make trips with
their family during this time.

    ## [1] "The in-sample confusion matrix of the best linear model is:-"

    ##    yhat
    ## y      0    1
    ##   0 5354  646
    ##   1 1394 1606

    ## [1] "The in-sample accuracy of the best linear model is: 0.773333333333333"

    ## [1] "The out-of-sample confusion matrix of the best linear model is:-"

    ##    yhat
    ## y      0    1
    ##   0 1816  184
    ##   1  286  349

    ## [1] "The out-of-sample accuracy of the best linear model is: 0.821631878557875"

    ## [1] "The out-of-sample precision of the best linear model is: 0.654784240150094"

    ## [1] "The out-of-sample recall of the best linear model is: 0.549606299212598"

    ## [1] "The out-of-sample F1 Score of the best linear model is: 0.597602739726027"

We can see that the best linear model performs better than the baseline
2 model.

    ## [1] "Its absolute improvement in terms of out-of-sample accuracy is approximately 0.0280834914611006"

    ## [1] "Its relative improvement in terms of out-of-sample accuracy is approximately 1.03538976566236"

    ## [1] "Its absolute improvement in terms of out-of-sample F1 Score is approximately 0.0532141300107846"

    ## [1] "Its relative improvement in terms of out-of-sample F1 Score is approximately 1.09775026343519"

### Model Validation Step 1: Performance of the Best Linear Model on hotel\_val data

    ## [1] "The confusion matrix of the best linear model is:-"

    ##    yhat
    ## y      0    1
    ##   0 4026  571
    ##   1  172  230

    ## [1] "The accuracy of the best linear model is: 0.851370274054811"

    ## [1] "The precision of the best linear model is: 0.287141073657928"

    ## [1] "The recall of the best linear model is: 0.572139303482587"

    ## [1] "The F1 Score of the best linear model is: 0.382377389858687"

We can see that the best linear model gives decent recall but has low
precision. That means, it is identifying many false positives.

The ROC curve for the best model, using the data in hotels\_val is:-  
![](Data-Mining-Assignment-2_files/figure-markdown_strict/ROC%20Curve-1.png)

We can see that the ROC curve is towards top-left of the 45-degree
diagonal of the ROC space. The closer the curve comes to the 45-degree
diagonal of the ROC space, the less accurate the test.

### Model Validation Step 2

    ##    fold_number actual_booking expected_booking
    ## 1            1             22         69.66802
    ## 2            2             17         67.49386
    ## 3            3             17         68.88840
    ## 4            4             19         64.00937
    ## 5            5             24         72.59252
    ## 6            6             18         64.01353
    ## 7            7             20         70.07587
    ## 8            8             15         64.02038
    ## 9            9             25         68.17613
    ## 10          10             21         63.34945
    ## 11          11             25         65.79570
    ## 12          12             23         67.82691
    ## 13          13             20         63.42374
    ## 14          14             24         65.38581
    ## 15          15             21         69.93728
    ## 16          16             24         73.60524
    ## 17          17             18         62.55803
    ## 18          18             15         63.87762
    ## 19          19             16         62.08709
    ## 20          20             18         68.38398

We can see that the expected number of bookings with children in a fold
is relatively higher than the actual number of bookings with children in
that fold. This indicates that there are too many false positives in the
model.

Note that the expected number of bookings with children in a fold can
become close to the actual number of bookings with children in that fold
by adding more data points with zero children in the training data.
However, this will increase the false negatives in the data. Thus there
exists a trade-off between the two. The optimal problem should be the
choice of total number of data points with zero children that should be
included in the data such that F1 score is maximized.
