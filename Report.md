# KDD Cup 98 Challenge

## Task

Description of the challenge
The context of this challenge is a charitable organisation is trying to target people effectively through
direct mail and would like to understand how to best target people who are most likely to donate to
charity.

From a data science point of view, it consists of two modelling exercises:
1. Building a binary classification model, predicting which people are more likely to donate to a
charity
2. Building a regression model, predicting the amount of money that each person is likely to donate

We are then interested in understanding how we could combine the results of the two models to
build a list of potential people to target. The goal is to maximise the net amount of money that
would be collected through a direct mail activity, where mail would be sent to people to collect their
donations.

Based on your solution could you recommend how many people we should target if each envelope
was costing us:
- $5 to send out.
- $1 to send out.

## Comments About The Project

My goal was to implement a system that would fullfil the objectives above and
therefore drive up the organisation’s revenue.

Often I write two types of reporting; one technical and the other more
high-level that caters for clients, business analysts or project managers. In
this report I describe the project in technical terms and thus its content,
format and emphasis are directed for a suitable audience.

Also sorry for not having graphs to support this report but I didn’t have
enough time to do some nice plots and tables. If I had more time to work on
this I would have done the following as well:

- Create graphical representation of the data.
- Do cross-validation grid-search for Model Selection.
- Do a business action plan supported by the results of this analysis.

The file `donors.py` and `README.md` are good starting points to understand
my solution and its steps.


## Dataset

The Dataset is divided into training data "cup98LRN.csv" and validation data "cup98VAL.csv"

#### Size

- 191779 records: 95412 training cases and 96367 test cases
- 481 attributes
- 236.2 MB: 117.2 MB training data and 119 MB test data

## My Solutions

They are structured around the following steps:

1. Data Importation
2. Exploratory Analysis
3. Data Munging
4. Feature Selection
5. Model Selection
6. Training
7. Testing
8. Model Evaluation and Comparison
9.Solution Validation using both models

In my solution to task 1 I follow this procedure.

In my solution to task 2 the regressor built using complete training data 
performs very poorly hence as a solution first I predict who is a donor, 
and then - using just those samples - I train a regressor that predicts 
how much the person donated.

I used only the training cases that were provided and made my training and test
sets out of that file. Thus my train and test sets together have 95412 cases.

Although I have created a third notebook using the validation dataset which 
will test the best model and predict donations, as it should work when both the 
models are integrated.

## System Architecture

```
.
├── README.html
├── README.md
├── data
│   ├── cup98LRN.csv
│   ├── cup98VAL.csv
│   ├── valtargt.csv
│
├── eda
│   ├──EDA_data.html
│   ├──Donors.html
│
├── Models
│   ├── donations_model.sav
│   ├── donations_model_features.csv
│   ├── donor_model.sav
│
├── lib
│   ├── __init__.py
│   ├── analyser.py
│   ├── trainer.py
│   ├── preprocessor.py
│   ├── utils.py
├── profits.ipynb
├── donors.ipynb
├── validation.ipynb
├── report.html
└── report.md

```

The main files are `donors.ipynb` and `profits.ipynb` and `validation.ipynb`. 
The EDA HTML’s generated are in EDA and all the auxiliary classes and 
their methods are in `lib`.

## Strategy, Comments, Decisions and Results

My comments to the steps mentioned in section My Solution are:

### Exploratory Analysis

#### What I Did

Utilized pandas.profiling to create a report of the data. 

- Checked the dimensionality of the raw data and how many missing values are
per variable.
- Looked at the data, analysed its variables, type and meaning.
- Looked at the distribution of the target variables.
- 
- Computed a set of descriptive statistics (mean, median, count, std, min,
max, percentiles, etc).
- Checked the documentation to understand how the variables are categorised.
- If I had more time I would had checked how donations are distributed among
age groups and per gender.

#### Comments

- The dataset contains only 5% of donors.
- The donations are usually smaller than $20.
- This data is quite noisy, high dimensional and with lots of missing values.
Feature selection and preprocessing will be vital for good modelling.
- There are records with formatting errors.
- There is an inverse relationship between the probability to donate and
the amount donated.

### Data Munging

#### What I did

- Identified redundant variables based on:
    - low variance,
    - low sparsity,
    - linear dependency to other variables, 
    - common sense.
- The previous step is made via a method at `Analyser.get_redundant_vars()`.
- Performed dimensionality reduction by dropping those vars/columns.
- Imputed the data by filling in the missing values with the mean if the
variable is a numeric type or with the most common term if the variable is
an object type. Made a method for this at `Preprocessor.fill_nans()`.
- Changed categorical variables into a numerical representation.
- applied Principal Component Analysis for dimensionality
reduction as the final step.


#### Comments

- Found several redundant variables but dropping them wasn’t enough for
significant dimensionality reduction.
- With all the missing values this step is quite sensitive. I hope the data
imputation doesn’t add too much noise to the variables.


### Feature Selection

#### What I did

- As a result of PCA, the features were reduced to 50 features
- Changed categorical variables into a numerical form.
- Made train and test sets, in its full form and in a balanced version.



#### Comments

- Even if I tried many methods for Feature Selection I could not spot an
optimal set of variables, and this is vital for good performance.
- If I had more time I would had fiddled more with Feature Selection.
- The balanced set is useful because some methods converge faster and better
if the training data is balanced.
- For this dataset this seems to be the most important part of the statistical
modelling.

### Data Imbalance

#### What I did
- The Data was imbalanced and its vital to deal with such imbalance to create a 
good model.
-I Balanced the dataset using three techniques 
    - Random Undersampling
    - Adasyn Oversamplng
    - Smote Oversampling


### Model Selection

#### What I did

- I tried manually several combinations of parameters of the training methods
and watched its impact. Then I chose the ones that seemed performing decently.
- I have implemented boosting and bagging algorithms along with Decision tree 
and Logistic regression.
-  An obvious choice for this is to do cross-validation grid search to find
optimal parameters,which is also implemented

#### Comments

- For Task 2, I did not have enough time to try multiple models.

### Training

#### What I did

##### Task 1

- Deployed 4 methods:
    - Method 1 | Logistic Regression Model.
    - Method 2 | Decision Trees Model
    - Method 3 | Random Forest Model 
    - Method 4 | XGBoost

##### Task 2

- First I predict who is a donor, and then - using just those samples -
I train a classifier that predicts how much the person donated.
- For predicting the donors I used the best model used/saved in `donors.ipynb` and for predicting the
donations I used Linear Regression.

#### Comments

- Would be nice to try as well a Naive Bayes Model and a Neural Network Model.
- Choose Method 3 as a baseline and
- tune in the parameters of the training methods of methods 1 and 2.
- Cherry pick the best 3 and build an optimal ensemble method.

- Should had used in both classifiers cross validation.
- Need to reevaluate my implementation and to make the training faster.

### Testing, Model Evaluation and Results

##### Task 1

- For each method computed the confusion matrix, accuracy, recall, precision
and F1. Made a method for this at `Performance.get_perf()`.

The Best values are below 
```
                   Model                            Accuracy     Recall      Precision      F1
  
    UNDERSAMPLE Logistic Regression Model          0.614159 	0.564499	0.073050	0.129360
    UNDERSAMPLE Decision Trees Model               0.569984 	0.524252	0.061553	0.110171
    UNDERSAMPLE Random Forest Model                0.614578 	0.541796	0.070602	0.124926
    UNDERSAMPLE XGBoost                            0.592412 	0.539732 	0.066582	0.118540
    ADASYN Logistic Regression Model               0.559241 	0.496388	0.057235	0.102635
    ADASYN Decision Trees Model                    0.659697 	0.444788	0.067481	0.117183
    ADASYN Random Forest Model                     0.570351 	0.600619	0.069335	0.124319
    ADASYN XGBoost                                 0.706493 	0.343653 	0.062854	0.106271
    SMOTE Logistic Regression Model                0.585600 	0.597523	0.071508	0.127730
    SMOTE Decision Trees Model                     0.556831 	0.496388	0.056923	0.102134
    SMOTE Random Forest Model                      0.672064 	0.430341	0.068104	0.117597
    SMOTE XGBoost                                  0.706231 	0.349845 	0.063782	0.107893
```
- All the methods don’t perform significantly well.
- The best one is SMOTE Random Forest Model.
- It would had been nice to display for each method its lift, AUC and ROC curves.
- With better Feature and Model Selection the results can be improved.

##### Task 2

- Uses Adjusted R- squared to evaluate the performance of my solution.
- I think the results can by improved by reevaluating my implementation
and the feature selection.


### Validation

-I have used the validation dataset, to check how the solution is working. For this
I have used the models developed using training data.
- The notebook first checks the probable doners, and then predict their donations. 
- At last it also checks on Donations missed, Donors Missed, Donors identified, and 
total donations collected, for stakeholder analysis


#### Comments

- As the models do not siginificantly well in training same is observed in the validation
data as well.
- Deeper analysis of the given dataset and better understanding is required to process 
as diverse data as given in order to improve model performance. 

## Author

[Purvi Jain](https://www.linkedin.com/in/purvinyx) 2022
