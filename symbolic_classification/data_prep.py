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

sns.set_theme()

df = pd.read_csv('paysim.csv')
df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})

# Account for missing values
df['externalDest'] = ((df['oldbalanceDest'] == 0) & (df['newbalanceDest'] == 0)).astype(int)
df['externalOrig'] = ((df['oldbalanceOrig'] == 0) & (df['newbalanceOrig'] == 0)).astype(int)

# Update the values in the 'newbalanceDest' column to 'oldbalanceDest +- amount'
df['newbalanceDest'] = df['oldbalanceDest'] + df['amount']
df['oldbalanceOrig'] = df['newbalanceOrig'] + df['amount']

# Feature Engineering
num_transactions = df.shape[0]
std = 0.01 * (df['amount'].quantile(0.75) - df['amount'].min())

# Adding progress tracking for each step
print("Calculating transaction counts and means...")
df['num_transDest'] = df.groupby('nameDest')['nameDest'].transform('count')
df['meanDest'] = df.groupby('nameDest')['amount'].transform('mean')

# Progress bar for apply
tqdm.pandas(desc="Applying exclude_current_meanDest")
df['meanDest'] = df.progress_apply(exclude_current_meanDest, axis=1)
df['meanDest'] += np.random.normal(0, std, num_transactions)

df['num_transOrig'] = df.groupby('nameOrig')['nameOrig'].transform('count')
df['meanOrig'] = df.groupby('nameOrig')['amount'].transform('mean')

# Progress bar for apply
tqdm.pandas(desc="Applying exclude_current_meanOrig")
df['meanOrig'] = df.progress_apply(exclude_current_meanOrig, axis=1)
df['meanOrig'] += np.random.normal(0, std, num_transactions)

print("Swapping columns for consistency...")
num_transDest = df.pop('num_transDest')
df['num_transDest'] = num_transDest
num_transOrig = df.pop('num_transOrig')
df['num_transOrig'] = num_transOrig

print("Calculating maximum amounts...")

# Ensure maxDest and maxOrig columns exist before applying the exclusion functions
df['maxDest'] = df.groupby('nameDest')['amount'].transform('max')
df['maxOrig'] = df.groupby('nameOrig')['amount'].transform('max')

print("Precomputing maximum amounts excluding current transaction...")

# For destinations
tqdm.pandas(desc="For destinations")
max_dest_excl = df.groupby('nameDest')['amount'].progress_apply(lambda x: x[::-1].expanding(1).max()[::-1])
df['maxDestExcl'] = max_dest_excl.reset_index(level=0, drop=True)

# For origins
tqdm.pandas(desc="For origins")
max_orig_excl = df.groupby('nameOrig')['amount'].progress_apply(lambda x: x[::-1].expanding(1).max()[::-1])
df['maxOrigExcl'] = max_orig_excl.reset_index(level=0, drop=True)

def exclude_current_maxDest(row):
    if row['num_transDest'] > 1 and row['maxDest'] == row['amount']:
        return row['maxDestExcl']
    return row['maxDest']

def exclude_current_maxOrig(row):
    if row['num_transOrig'] > 1 and row['maxOrig'] == row['amount']:
        return row['maxOrigExcl']
    return row['maxOrig']

# Apply optimized functions with progress bar
tqdm.pandas(desc="Applying optimized exclude_current_maxDest")
df['maxDest'] = df.progress_apply(exclude_current_maxDest, axis=1)

tqdm.pandas(desc="Applying optimized exclude_current_maxOrig")
df['maxOrig'] = df.progress_apply(exclude_current_maxOrig, axis=1)

# Dropping the temporary columns
df.drop(columns=['maxDestExcl', 'maxOrigExcl'], inplace=True)

print("Sorting values by timestep...")
df = df.sort_values('step')

print("Calculating rolling averages and maxima...")
for window in [3, 7]:
    df[f'meanDest{window}'] = df.groupby('nameDest')['amount'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
    df[f'maxDest{window}'] = df.groupby('nameDest')['amount'].rolling(window=window, min_periods=1).max().reset_index(0, drop=True)

print("Rearranging columns...")
# Rearrange column order
df = df.reindex(columns=[col for col in df.columns if col not in ['num_transDest', 'num_transOrig', 'externalDest', 'externalOrig', 'isFraud']] + ['num_transDest', 'num_transOrig', 'externalDest', 'externalOrig', 'isFraud'])

# Prepare the training and test sets
df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, shuffle=True)
val_set_size = 0.10 / 0.25
test_df, val_df = train_test_split(test_df, test_size=val_set_size, random_state=42, shuffle=True)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Save a checkpoint before further processing
train_df.to_csv('train_checkpoint.csv', index=False)
test_df.to_csv('test_checkpoint.csv', index=False)
val_df.to_csv('val_checkpoint.csv', index=False)

