# src/test_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')


def load_data():
    # Load the dataset
    df = pd.read_csv("../data/processed/DataProcessed.csv")
    
    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Drop the 'Date' column
    df = df.drop(columns=['Date'])
    
    # Identify categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Separate features and target variable
    X = df.drop(columns=['Class'])  # Replace 'Class' with the name of your target column
    Y = df['Class']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, Y

def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plot confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {title}')
    plt.show()

def logistic_regression(X_train, Y_train):
    """
    Train Logistic Regression model with GridSearchCV.
    """
    parameters = {"C":[0.01,0.1,1], 'penalty':['l2'], 'solver':['lbfgs']}
    lr = LogisticRegression()
    grid_search = GridSearchCV(lr, parameters, cv=10)
    grid_search.fit(X_train, Y_train)
    
    return grid_search

def support_vector_machine(X_train, Y_train):
    """
    Train Support Vector Machine model with GridSearchCV.
    """
    parameters = {
        'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
        'C': np.logspace(-3, 3, 5),
        'gamma':np.logspace(-3, 3, 5)
    }
    svm = SVC()
    grid_search = GridSearchCV(svm, parameters, cv=10)
    grid_search.fit(X_train, Y_train)
    
    return grid_search

def decision_tree(X_train, Y_train):
    """
    Train Decision Tree model with GridSearchCV.
    """
    parameters = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [2*n for n in range(1,10)],
        'max_features': ['auto', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, parameters, cv=10)
    grid_search.fit(X_train, Y_train)
    
    return grid_search

def k_nearest_neighbors(X_train, Y_train):
    """
    Train K-Nearest Neighbors model with GridSearchCV.
    """
    parameters = {
        'n_neighbors': range(1,11),
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1,2]
    }
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, parameters, cv=10)
    grid_search.fit(X_train, Y_train)
    
    return grid_search

def random_forest(X_train, Y_train):
    """
    Train Random Forest Classifier.
    """
    rf = RandomForestClassifier(
        criterion='gini',
        max_depth=18,
        n_estimators=200,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=200
    )
    rf.fit(X_train, Y_train)
    return rf

def xgboost_classifier(X_train, Y_train):
    """
    Train XGBoost Classifier.
    """
    xgb = XGBClassifier(
        max_depth=10,
        random_state=10,
        n_estimators=100,
        eval_metric='auc',
        min_child_weight=3,
        colsample_bytree=0.75,
        subsample=0.9
    )
    xgb.fit(X_train, Y_train)
    return xgb

def evaluate_model(model, X_test, Y_test, model_name):
    """
    Evaluate the model and plot confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    print(f'Accuracy of {model_name} on test data: {accuracy:.3f}')
    plot_confusion_matrix(Y_test, y_pred, model_name)
    return accuracy

def compare_models(results):
    """
    Compare model accuracies and plot results.
    """
    df = pd.DataFrame(results.items(), columns=['Model', 'Accuracy'])
    df = df.sort_values(by='Accuracy', ascending=False)
    
    fig = px.bar(
        df,
        x='Model',
        y='Accuracy',
        color='Model',
        title='Model Comparison on Test Data'
    )
    fig.show()

def main():
    # Load and split data
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )
    
    # Train models
    print("Training Logistic Regression...")
    lr_model = logistic_regression(X_train, Y_train)
    
    print("Training Support Vector Machine...")
    svm_model = support_vector_machine(X_train, Y_train)
    
    print("Training Decision Tree...")
    dt_model = decision_tree(X_train, Y_train)
    
    print("Training K-Nearest Neighbors...")
    knn_model = k_nearest_neighbors(X_train, Y_train)
    
    print("Training Random Forest...")
    rf_model = random_forest(X_train, Y_train)
    
    print("Training XGBoost...")
    xgb_model = xgboost_classifier(X_train, Y_train)
    
    # Evaluate models
    results = {}
    results['Logistic Regression'] = evaluate_model(lr_model, X_test, Y_test, 'Logistic Regression')
    results['Support Vector Machine'] = evaluate_model(svm_model, X_test, Y_test, 'Support Vector Machine')
    results['Decision Tree'] = evaluate_model(dt_model, X_test, Y_test, 'Decision Tree')
    results['K-Nearest Neighbors'] = evaluate_model(knn_model, X_test, Y_test, 'K-Nearest Neighbors')
    results['Random Forest'] = evaluate_model(rf_model, X_test, Y_test, 'Random Forest')
    results['XGBoost'] = evaluate_model(xgb_model, X_test, Y_test, 'XGBoost')
    
    # Compare and visualize model performance
    compare_models(results)
    
    # Save the best model
    best_model_name = max(results, key=results.get)
    best_model = {
        'Logistic Regression': lr_model,
        'Support Vector Machine': svm_model,
        'Decision Tree': dt_model,
        'K-Nearest Neighbors': knn_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }[best_model_name]
    
    joblib.dump(best_model, f'../models/{best_model_name.replace(" ", "_").lower()}.pkl')
    print(f'Best model ({best_model_name}) saved successfully.')

if __name__ == "__main__":
    main()
