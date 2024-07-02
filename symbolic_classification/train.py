import sys
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from kan.KAN import *
from utils import *

# Add the path to your KAN module
sys.path.append(os.path.abspath('/Users/barager/Desktop/symbolic classifciation/kan'))

def main():
    # Load the datasets
    train_df = pd.read_csv('train_df.csv').iloc[:50_000]
    val_df = pd.read_csv('val_df.csv').iloc[:50_000]
    test_df = pd.read_csv('test_df.csv').iloc[:50_000]

    

    print("loaded data!")
    # Ensure the type columns are integer
    train_df[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']] = train_df[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']].astype(int)
    val_df[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']] = val_df[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']].astype(int)
    test_df[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']] = test_df[['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']].astype(int)

    # Extract features and labels
    feature_cols = ['step', 'amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'meanDest', 'meanOrig', 'maxDest', 'maxOrig', 'meanDest3', 'meanDest7', 'maxDest3', 'maxDest7', 'num_transDest', 'num_transOrig', 'externalDest', 'externalOrig', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']
    X_train, y_train = train_df[feature_cols], train_df['isFraud']
    X_val, y_val = val_df[feature_cols], val_df['isFraud']
    X_test, y_test = test_df[feature_cols], test_df['isFraud']
    
    print(y_train.value_counts())
    print(y_val.value_counts())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = {
    'train_input': torch.tensor(X_train.values, dtype=torch.float32).to(device),
    'train_label': torch.tensor(y_train.values, dtype=torch.float32).to(device),
    'test_input': torch.tensor(X_val.values, dtype=torch.float32).to(device),
    'test_label': torch.tensor(y_val.values, dtype=torch.float32).to(device)
    }

    # Define the function to count model parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # Ensure labels are in long format and on the same device
    dataset['train_label'] = dataset['train_label'].long().to(device)
    dataset['test_label'] = dataset['test_label'].long().to(device)

    # Create DataLoaders for batch processing
    train_dataset = TensorDataset(dataset['train_input'].to(device), dataset['train_label'])
    test_dataset = TensorDataset(dataset['test_input'].to(device), dataset['test_label'])

    batch_size = 10_000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

    # Define the model
    model = KAN(width=[X_train.shape[1], 2], grid=2, k=10, device=device)
    print(f"KAN parameters: {count_parameters(model)}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    
    targets = dataset['train_label']
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (2 * class_counts.float())
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device))
    

    # Functions to compute metrics
    def compute_metrics(y_true, y_pred):
        precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, f1, accuracy

    def evaluate_model(loader):
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return compute_metrics(all_labels, all_preds)

    # Training loop
    num_epochs = 500
    grid_update_freq = 10
    stop_grid_update_step = 100
    
    for epoch in tqdm(range(num_epochs)):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
        
    
                
        # Evaluate the model after each epoch
        train_precision, train_recall, train_f1, train_acc = evaluate_model(train_loader)
        test_precision, test_recall, test_f1, test_acc = evaluate_model(test_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | LR: {optimizer.param_groups[0]["lr"]:.4f}')
        print(f'Train - Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, Accuracy: {train_acc:.4f}')
        print(f'Test - Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}, Accuracy: {test_acc:.4f}')
        print()

    y_pred = torch.argmax(model(dataset['test_input'].to(device)), dim=1).cpu().numpy()
    y_true = dataset['test_label'].cpu().numpy()
    print(classification_report(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))

    # Configure classifiers
    rfc = RandomForestClassifier(n_estimators=20, max_depth=5)
    dtc = DecisionTreeClassifier(max_depth=5)
    logreg = LogisticRegression(max_iter=100, penalty='l2', solver='lbfgs')
    nb = GaussianNB()

    # Fit the models
    rfc.fit(X_train, y_train)
    dtc.fit(X_train, y_train)
    logreg.fit(X_train, y_train)
    nb.fit(X_train, y_train)

    # Store the models in a list
    classifiers = [rfc, dtc, logreg, nb]

    for classifier in classifiers:
        y_pred = classifier.predict(X_test)
        print(classifier.__class__.__name__)
        print(classification_report(y_test, y_pred))
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print()

if __name__ == '__main__':
    main()