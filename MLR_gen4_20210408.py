# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:51:25 2020

@author: JMMEINHARDT

This script will generate multivariate linear regression models predicting a response variable using selected descriptors from a 
feature matrix. The script takes an input Excel workbook and returns models built using all descriptors, as well as subsets of
descriptors that are statistically most significant. Models are generated using scikit-learn linear models.

The user may specify the number of terms in the n-term linear regression model and the method of stepwise regression used for
feature selection to include only the n-most significant descriptors. The user may visualize the n-term and all-term models 
numerically and graphically. The user may also compare MLR predicted values (n-term, all-term) with the experimental response 
variable values for each model.

"""

# # # IMPORT PYTHON MODULES # # #

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import linear_model
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from textwrap import wrap
from math import sqrt

# # # DEFINE FUNCTIONS # # #

# define a function to normalize a feature dataframe using min-max normalization. the output is a new normalized dataframe.
def normalizer(dataframe):
    dataframe_normalized=(dataframe-dataframe.mean())/(dataframe.std())
    return(dataframe_normalized)

# define a function for data preprocessing. this function divides an input dataframe into response and feature matrix dataframes.
def preProcess(dataframe, prop, index):
    # drop NAN property values and separate dataframe into property dataframe (y) and feature matrix (x).
    dfwithoutNAN = dataframe.dropna(subset=[ycolumn])
    # remove columns from dataframe that contain only zeros. these features will not contribute to the MLR model.
    dfwithoutNAN = dfwithoutNAN.loc[:, (dfwithoutNAN != 0).any(axis = 0)]
    # separate dataframe into x (features) and y (response) dataframes.
    x = dfwithoutNAN.iloc[:, index:]
    x= x.fillna(dfwithoutNAN.mean())
    y = dfwithoutNAN[ycolumn]
    y_label = dfwithoutNAN['Name']
    x = normalizer(x)
    return(x, y, y_label)

# define a function that lets the user specify the property of interest.
def propertySelector(prop_of_interest):
    if prop_of_interest == 1:
        ycolumn = 'Tdec'
    elif prop_of_interest == 2:
        ycolumn = 'Impact Sensitivity / J (without < and >)'
    elif prop_of_interest == 3:
        ycolumn = 'LN(IS)'
    elif prop_of_interest == 4:
        ycolumn = 'Friction Sensitivity / N (without < and >)'
    elif prop_of_interest == 5:
        ycolumn = 'ESD / J (without < and >)'
    return(ycolumn)

# define a function to perform stepwise regression (SR) feature selection on feature matrix. inputtable value of 'step_reg'.
def sequentialForwardSelection(x, y, n, step_reg):
    if step_reg == 1:
        method = 1
    elif step_reg == 2:
        method = 0
    elif step_reg == 3:
        # bidirectional elimination, method will default to forward selection.
        method = 1
    # perform stepwise regression.
    sffs = SFS(linear_model.LinearRegression(), k_features = n, forward = method, floating = False, scoring = 'r2', cv = 0)
    forward_selected = sffs.fit(x, y)
    features_SR = list(forward_selected.k_feature_names_)
    # return stepwise regression selected feature matrix 'x_SR'.
    x_SR = x[features_SR]
    y_SR = y
    return(x_SR, y_SR)

# define a function to obtain an sklearn linear regression model fitting a property dataframe (y) using a feature matrix (x).
def getMLR(x, y):
    # generate MLR model object.
    lm = linear_model.LinearRegression()
    model = lm.fit(x, y)
    # write predictions array as a dataframe with ycolumn as header.
    y_predictions = pd.DataFrame(lm.predict(x))
    y_predictions.columns = [ycolumn]
    # score MLR model using r^2 and RMSE metrics.
    r2_score = float(round(lm.score(x, y),5))
    rmse = round(sqrt((mean_squared_error(y, y_predictions))),5)
    # get MLR coefficients.
    coef = lm.coef_
    coefficients = pd.DataFrame(coef)
    coefficients = coefficients.transpose()
    coefficients.columns = x.columns
    coefficients.index = ['Term Coefficient']
    # get MLR intercept.
    intercept = lm.intercept_
    return(lm, model, y_predictions, r2_score, rmse, coefficients, intercept)

# define a function to get an n-term MLR equation in text form.
def writeMLR(x_SR, coefficients_SR, n, intercept):
    MLR_equation = 'y = '
    for i in range(0, n):
        coefficient = coefficients_SR.iloc[0, i]
        coefficient = round(coefficient, 3)
        parameter = coefficients_SR.columns[i]
        term = str(coefficient) + '*(' + parameter + ')'
        MLR_equation = MLR_equation + ' + ' + term
    intercept = round(intercept,3)
    intercept = str(intercept)
    MLR_equation = MLR_equation + ' + ' + intercept 
    MLR_equation = MLR_equation[:3] + MLR_equation[6:]
    return(MLR_equation)

# define a function to generate x-y plot for MLR model using all descriptors.
def xyAllFeatures(y, y_predictions):
    fig, ax = plt.subplots()
    ax.scatter(x = y, y = y_predictions, color= 'blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=1)
    ax.set_xlabel('Experimental Value')
    ax.set_ylabel('MLR Predicted Value')
    y, y_predictions = y.values.reshape(-1,1), y_predictions.values.reshape(-1,1)
    ax.plot(y, linear_model.LinearRegression().fit(y, y_predictions).predict(y))
    r2 = str(r2_score(y, y_predictions))
    ax.set_title('MLR Using All Terms \n  R2: %s ' % r2)
    return()

# define a function to generate x-y plot for MLR model using the n most significant descriptors.
def xyNFeatures(y, y_predictions, n, MLR_equation):
    fig, ax = plt.subplots()
    ax.scatter(x = y, y = y_predictions, color= 'blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k-', lw=1)
    ax.set_xlabel('Experimental Value')
    ax.set_ylabel('MLR Predicted Value')
    y, y_predictions = y.values.reshape(-1,1), y_predictions.values.reshape(-1,1)
    ax.plot(y, linear_model.LinearRegression().fit(y, y_predictions).predict(y))
    r2 = str(r2_score(y, y_predictions))
    # this solution for the title is not super elegant but it was straightforward to implement.
    ax.set_title('\n'.join(wrap('MLR Using %d Terms                                         \n %s \n                        R2: %s ' % (n, MLR_equation, r2), 60)))
    plt.show()
    return()

# define a function to compute, display, and graph linear regression models that include 0 to n terms.
def modelPrinter(x, y, n, step_reg):
    # create an empty dataframe for storing predicted values from each model.
    y_predictions_SR_summary = pd.DataFrame()
    # iteratively generate regression models with numbers of terms up to and including n.
    for i in range(1,n+1):
        # perform stepwise regression on data for n-term model.
        x_SR, y_SR = sequentialForwardSelection(x, y, i, step_reg)
        # obtain an n-term MLR using SR selected features.
        lm_SR, model_SR, y_predictions_SR, r2_SR, rmse_SR, coefficients_SR, intercept_SR = getMLR(x_SR, y_SR)
        # obtain an n-term MLR equation for SR selected data.
        MLR_equation = writeMLR(x_SR, coefficients_SR, i, intercept)
        # obtain an MLR plot for n-term model.
        xyNFeatures(y_SR, y_predictions_SR, i, MLR_equation)
        # print statements summarizing each i-term equation.
        print('\nBEST %d-TERM MODEL EQUATION:' % (i))
        print(MLR_equation)
        print('Model R Squared: %s' % r2_SR)
        print('Model RMSE: %s' % rmse_SR)
        y_predictions_SR.columns = ['N = %d' % i]
        # add responses calculated by current model to summary dataframe.
        y_predictions_SR_summary = pd.concat([y_predictions_SR_summary, y_predictions_SR], axis = 1)
    return(y_predictions_SR_summary)

# define a function to create a dataframe containing actual and model-predicted response variable values.
def getComparison(y_label, y, y_predictions, y_predictions_SR):
    comparison_df = pd.concat([y_label, y, y_predictions], axis = 1)
    comparison_df.columns = ['Name', ycolumn, 'N = All']
    comparison_df = pd.concat([comparison_df, y_predictions_SR], axis = 1)
    return(comparison_df)

# # # MAIN JOB # # #

# USER: specify name of Excel workbook containing property of interest and descriptors.
workbook_name = 'Training-Set-Boltzmann.xlsx'
# USER: set the index value for the column in the Excel workbook where descriptors begin.
index = 13


# initalize workbook as a pandas dataframe.
df_input = pd.read_excel(workbook_name)
# specify the property of interest for prediction.
prop_of_interest = int(input('Enter the property of interest: \n (1) for Tdec \n (2) for LOG(IS)'))
ycolumn = propertySelector(prop_of_interest)
# specify the maximum number of terms for calculated linear regression model.
n = int(input('Enter the number of features that should be included in an n-term MLR equation: '))
# specify a stepwise regression method for feature reduction to n terms.
step_reg = 1


# run preprocessing function to yield cleaned property and feature matrix dataframes.
x, y, y_label = preProcess(df_input, ycolumn, index)

# obtain a MLR model using all descriptors.
lm, model, y_predictions_all_terms, r2, rmse, coefficients, intercept = getMLR(x, y)
# generate a plot for the MLR model containing all descriptors.
xyAllFeatures(y, y_predictions_all_terms)

# generate summary table for MLR models using n or fewer descriptors.
print("\n-----------------------------------------------------------------------------------")
print('DATASET: %s' % workbook_name)
print('PREDICTING: %s' % ycolumn)
y_predictions_SR_summary = modelPrinter(x, y, n, step_reg)
print("-----------------------------------------------------------------------------------")

# get comparison dataframe for user analysis.
comparison_df = getComparison(y_label, y, y_predictions_all_terms, y_predictions_SR_summary)

# print information for user on plots and comparison dataframe.
print('FIGURES: MLR Model (all terms), MLR Model (n selected terms).')
print('DATA: experimental, MLR predicted (all terms), and MLR predicted (n terms) property values can be accessed by calling the "comparison_df" dataframe from the variable explorer.')
print("-----------------------------------------------------------------------------------")