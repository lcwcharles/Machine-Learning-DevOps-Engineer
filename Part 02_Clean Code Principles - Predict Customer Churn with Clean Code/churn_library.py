'''
functions to find customers who are likely to churn
date: 10.26.2022
author: Charles
'''


# import libraries
import os
import logging
import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from libs.plots import plot_savefig
from libs.plots import plot_heatmap
from libs.plots import plot_total_trans_ct
from libs.plots import plot_normalize
from libs.plots import plot_customer_age
from libs.plots import plot_churn


# from yellowbrick.classifier import ClassificationReport


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def main():
    '''
    main function
    '''
    path = './data/bank_data.csv'
    bank_data_df = import_data(path)
    perform_eda(bank_data_df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(bank_data_df)
    train_models(X_train, X_test, y_train, y_test)


def import_data(path):
    '''
    returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(path)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    img_folder = './images/eda/'
    plot_churn(df, img_folder)
    plot_customer_age(df, img_folder)
    plot_normalize(df, img_folder)
    plot_total_trans_ct(df, img_folder)
    plot_heatmap(df, img_folder)


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be 
                     used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # encoded column
    for col in category_lst:
        # #old meathod
        # col_lst = []
        # col_groups = df.groupby(col).mean()['Churn']
        # for val in df[col]:
        #     col_lst.append(col_groups.loc[val])
        
        # more efficient by taking advantage of pandas' transform
        col_name = col + '_Churn'
        df[col_name] = df.groupby(col)["Churn"].transform("mean")
    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be 
                      used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    df = encoder_helper(df, category_lst)
    # print(df.head())
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    y = df['Churn']
    X = df[keep_cols]
    # X = df.drop(["Churn"], axis=1)
    # y = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''


    img_folder = './images/results/'

    name = 'Random Forest'
    plot_savefig(
        img_folder,
        name,
        y_test,
        y_train,
        y_test_preds_rf,
        y_train_preds_rf)

    name = 'Logistic Regression'
    plot_savefig(
        img_folder,
        name,
        y_test,
        y_train,
        y_test_preds_lr,
        y_train_preds_lr)


def feature_importance_plot(model, X_data, output_path):
    '''
    creates and stores the feature importances in path
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_path: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    img_name = "Feature_Importance.png"
    img_path = os.path.join(output_path, img_name)

    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # define grid search and fit random forest classifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # fit logistic regression model
    lrc.fit(X_train, y_train)

    # predictions for random forest
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # predictions for logistic regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # plot ROC for logistic regression
    lrc_plot = plot_roc_curve(
        lrc,
        X_test,
        y_test)
    plt.close()

    # plot ROC for logistic regression and random forest and save it
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    output_path = './images/results/'
    img_name = "ROC_result.png"
    img_path = os.path.join(output_path, img_name)

    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # save classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # save feature importances plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        output_path)


if __name__ == "__main__":
    main()
