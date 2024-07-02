import numpy as np
import pandas as pd
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import *
import seaborn as sns
from tqdm import tqdm

train_df = pd.read_csv('train_checkpoint.csv',)
val_df = pd.read_csv('val_checkpoint.csv', )
test_df = pd.read_csv('test_checkpoint.csv', )



def prepare_dataset(df, scaler=None):
    type_col = df.pop('type') if 'type' in df.columns else None
    num_transDest_col = df.pop('num_transDest') if 'num_transDest' in df.columns else None
    num_transOrig_col = df.pop('num_transOrig') if 'num_transOrig' in df.columns else None
    externalDest_col = df.pop('externalDest') if 'externalDest' in df.columns else None
    externalOrig_col = df.pop('externalOrig') if 'externalOrig' in df.columns else None
    isFraud_col = df.pop('isFraud')

    if scaler is None:
        scaler = StandardScaler()
        scaler = scaler.fit(df)
    
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    
    if type_col is not None:
        df = pd.concat([df, type_col], axis=1)
    if num_transDest_col is not None:
        df = pd.concat([df, num_transDest_col], axis=1)
    if num_transOrig_col is not None:
        df = pd.concat([df, num_transOrig_col], axis=1)
    if externalDest_col is not None:
        df = pd.concat([df, externalDest_col], axis=1)
    if externalOrig_col is not None:
        df = pd.concat([df, externalOrig_col], axis=1)
    df = pd.concat([df, isFraud_col], axis=1)
    
    return df, scaler

print("Preparing training set...")
train_df, scaler = prepare_dataset(train_df)

print("Preparing test set...")
test_df, _ = prepare_dataset(test_df, scaler=scaler)

print("Preparing validation set...")
val_df, _ = prepare_dataset(val_df, scaler=scaler)

# One-hot encode the 'type' column if it exists
if 'type' in train_df.columns:
    train_df = pd.get_dummies(train_df, columns=['type'])
if 'type' in test_df.columns:
    test_df = pd.get_dummies(test_df, columns=['type'])
if 'type' in val_df.columns:
    val_df = pd.get_dummies(val_df, columns=['type'])

# Move the 'isFraud' column to the end of the dataframe
def move_is_fraud(df):
    is_fraud_col = df.pop('isFraud')
    df['isFraud'] = is_fraud_col
    return df

train_df = move_is_fraud(train_df)
test_df = move_is_fraud(test_df)
val_df = move_is_fraud(val_df)
# Save the train, validation, and test set
print("Saving final datasets...")
# Save the train, validation, and test set
print("Saving datasets...")
train_df.to_csv("train_df.csv", )
test_df.to_csv("test_df.csv", )
val_df.to_csv("val_df.csv",)

print("Processing complete.")