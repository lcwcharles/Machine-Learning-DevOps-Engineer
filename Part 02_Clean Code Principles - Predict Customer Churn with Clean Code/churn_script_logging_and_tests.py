'''
Test churn library
date: 10.26.2022
author: Charles
'''

import os
import logging
import time
import churn_library as cl


logging.basicConfig(
    filename=f"./logs/churn_library_{time.strftime('%b_%d_%Y_%H_%M_%S')}.log",
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS - %s",
                     time.strftime('%b_%d_%Y_%H_%M_%S'))
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found - %s",
                      time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    else:
        try:
            assert df.shape[0] > 0
            assert df.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "Testing import_data: The file doesn't appear to have rows and columns - %s",
                time.strftime('%b_%d_%Y_%H_%M_%S'))
            raise err
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        img_folder = './images/eda/'
        # https://www.guru99.com/python-check-if-file-exists.html
        if not os.listdir(img_folder):
            # assert Path(image_folder).exists()
            logging.info(
                "SUCCESS: images exist - %s",
                time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error(
            "ERROR: empty folder - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    try:
        category_lst = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'
        ]
        df = encoder_helper(df, category_lst)
        for col in category_lst:
            assert col in df.columns
            logging.info("SUCCESS: encoded successfully the feature - %s - %s",
                         col, time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error("ERROR: not encoded the feature - %s - %s",
                      col, time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("SUCCESS: performed feature engineering - %s",
                     time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error(
            "ERROR: something went wrong in feature engineering - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    return X_train, X_test, y_train, y_test


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(X_train_bank, X_test_bank, y_train_bank, y_test_bank)
        list_files = os.listdir("./images/results/")
        assert len(list_files) > 0
        logging.info(
            "SUCCESS: images exist - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
    except AssertionError as err:
        logging.error(
            "ERROR: empty folder - %s",
            time.strftime('%b_%d_%Y_%H_%M_%S'))
        raise err
    else:
        try:
            list_files = os.listdir("./models/")
            assert len(list_files) > 0
            logging.info(
                "SUCCESS: models exist - %s",
                time.strftime('%b_%d_%Y_%H_%M_%S'))
        except AssertionError as err:
            logging.error(
                "ERROR: empty folder - %s",
                time.strftime('%b_%d_%Y_%H_%M_%S'))
            raise err


if __name__ == "__main__":
    df_bank = test_import(cl.import_data)
    test_eda(cl.perform_eda, df_bank)
    encoded_df = test_encoder_helper(cl.encoder_helper, df_bank)
    X_train_bank, X_test_bank, y_train_bank, y_test_bank = test_perform_feature_engineering(
        cl.perform_feature_engineering, encoded_df)
    test_train_models(cl.train_models)
    logging.info(
        "SUCCESS: all tests finished running - %s",
        time.strftime('%b_%d_%Y_%H_%M_%S'))
    print("Finished running testing script")
