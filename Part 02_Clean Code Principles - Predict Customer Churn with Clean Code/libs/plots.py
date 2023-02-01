'''
plot functions
date: 10.26.2022
author: Charles
'''

import logging
import os
import time
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.naive_bayes import GaussianNB
# from yellowbrick.classifier import ClassificationReport

logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def plot_churn(df, img_folder):
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    # sns.lineplot(data = df)
    img_name = 'Churn.png'
    img_path = os.path.join(img_folder, img_name)
    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))


def plot_customer_age(df, img_folder):
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    img_name = 'Customer_Age.png'
    img_path = os.path.join(img_folder, img_name)
    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))


def plot_normalize(df, img_folder):
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    img_name = 'Normalize.png'
    img_path = os.path.join(img_folder, img_name)
    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    

def plot_total_trans_ct(df, img_folder):
    plt.figure(figsize=(20, 10))
    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained
    # using a kernel density estimate
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    img_name = 'Total_Trans_Ct.png'
    img_path = os.path.join(img_folder, img_name)
    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))


def plot_heatmap(df, img_folder):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    img_name = 'Heatmap.png'
    img_path = os.path.join(img_folder, img_name)
    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))


def plot_savefig(
        img_folder,
        name,
        y_test,
        y_train,
        y_test_preds,
        y_train_preds):
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str(f'{name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    img_name = "_".join(name.split()) + '.png'
    img_path = os.path.join(img_folder, img_name)
    logging.info("Saving plot on %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()
    logging.info("Saved plot in %s - %s", img_path, time.strftime('%b_%d_%Y_%H_%M_%S'))
