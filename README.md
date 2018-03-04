# Fraud Detection-using-Resampling

## Dependencies
- python3.6
- pandas 
- sklearn
- matplotlib 
- imblearn

## Running Steps
Update:
1. TRAINFILENAME: The CSV file of training data
2. TEST_FR: Fraction of training data to be kept aside for test data 
3. DISC_FEATURES_COL_TO_USE: Python List of Column names containing Discrete Features in TRAINFILENAME
4. CONT_FEATURES_COL_TO_USE: List of Column names containing Continuous Features in TRAINFILENAME
5. DISC_TARGET_COL_TO_USE: Name of Column names containing Target variable in TRAINFILENAME
6. Type arg passed to training function in main(). Type of model to be used from:
      - LR: Logistic Regression
      - SVM: Support Vector Classifier
      - RF: Random forest
      - GBC: Gradient Boosting Classifier
      - Default: Naive Bayes
<br />Run:
As ussual python script
