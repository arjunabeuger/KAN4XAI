""" This file contains functions that can be applied rows of a pandas dataframe """

import pandas as pd


# Functions with progress tracking
def exclude_current_meanDest(row):
    """
    Function that calculates mean of dest account excluding the current transaction
    """
    num_transactions = row['num_transDest']
    mean = row['meanDest']

    if num_transactions == 1:
        return mean

    amount = row['amount']
    return ((mean * num_transactions) - amount) / (num_transactions - 1)


def exclude_current_meanOrig(row):
    """
    Function that calculates mean of orig account excluding the current transaction
    """
    num_transactions = row['num_transOrig']
    mean = row['meanOrig']

    if num_transactions == 1:
        return mean

    amount = row['amount']
    return ((mean * num_transactions) - amount) / (num_transactions - 1)


def exclude_current_maxDest(row, df):
    """
    Function that calculates max of dest account excluding the current transaction
    """
    current_max = row['maxDest']
    amount = row['amount']
    transactions = row['num_transDest']

    if current_max == amount:
        if transactions > 1:
            max_exclude_current = df.loc[(df['nameDest'] == row['nameDest']) & (df['amount'] != amount), 'amount'].max()
            if pd.notna(max_exclude_current):
                return max_exclude_current
    return current_max


def exclude_current_maxOrig(row, df):
    """
    Function that calculates max of orig account excluding the current transaction
    """
    current_max = row['maxOrig']
    amount = row['amount']
    transactions = row['num_transOrig']

    if current_max == amount:
        if transactions > 1:
            max_exclude_current = df.loc[(df['nameOrig'] == row['nameOrig']) & (df['amount'] != amount), 'amount'].max()
            if pd.notna(max_exclude_current):
                return max_exclude_current
    return current_max