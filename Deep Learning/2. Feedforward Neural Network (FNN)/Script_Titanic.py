# Imports  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
train_data = 'train.csv'
train = pd.read_csv(train_data)
test_data = '/test.csv'
test = pd.read_csv(test_data)


## EDA AND PREPROCESSING
########################
# Checking for missing values in train set
train.isnull().sum()
# Checking for missing values in test set
test.isnull().sum()
# Function to replace missing values with the median
def replace_with_median(df, column_name):
    median_value = df[column_name].median()
    df[column_name].fillna(median_value, inplace=True)
# Replacing empty strings with NaN in train
train.replace('', np.nan, inplace=True)
# Replacing empty strings with NaN in test
test.replace('', np.nan, inplace=True)
# Checking missing values in train set after replacing empty strings
train.isnull().sum()
# Replacing the missing 'Embarked' values with the most frequent 'S'
train["Embarked"].fillna("S", inplace=True)
test["Embarked"].fillna("S", inplace=True)
# Replacing missing values of 'Age' column with the median
replace_with_median(train, 'Age')
replace_with_median(test, 'Age')
# Dropping 'Cabin', 'Name', 'Ticket', and 'PassengerId' columns in train and test
train.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
# Replacing rows of 'Fare' column in test set where values are NaN with the median
replace_with_median(test, 'Fare')
# Replacing categorical values of 'Sex' column by dummy variables
train = pd.get_dummies(train, columns=['Sex'], drop_first=True)
test = pd.get_dummies(test, columns=['Sex'], drop_first=True)
# Replacing categorical values of 'Embarked' column by dummy variables
train = pd.get_dummies(train, columns=['Embarked'], prefix='Embarked', drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], prefix='Embarked', drop_first=True)
# Checking data types of all columns in train set
train_data_types = train.dtypes
print("\nData types in train set:")
print(train_data_types)
# Checking data types of all columns in test set
test_data_types = test.dtypes
print("\nData types in test set:")
print(test_data_types)
 

## SPLIT AND STANDARDIZE THE DATA
################################# 
X_train = train.iloc[:,1:].values
y_train = train.iloc[:,0].values
X_test = test.iloc[:,:]
X_test.isnull().sum()
# Initialize the StandardScaler
sc = StandardScaler()
# Feature Scaling for the full X_train and X_test
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# Further split X_train into X_train_train and X_train_test for gaining more insights
X_train_train, X_train_test, y_train_train, y_train_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


## BUILD THE FNN MODEL
######################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# Define the FNN model
class FNN(nn.Module):
    def __init__(self, input_size):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 14)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(14, 14)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(14, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
# Initialize the FNN model
input_size = X_train_train.shape[1]   
model = FNN(input_size)
# Define the optimizer  
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Define the loss function
criterion = nn.BCELoss()
# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(torch.tensor(X_train_train, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y_train_train, dtype=torch.float32).view_as(outputs))
    loss.backward()
    optimizer.step()
# Predicting the training set results
with torch.no_grad():
    y_pred_train = model(torch.tensor(X_train_train, dtype=torch.float32))
    y_pred_train = (y_pred_train > 0.5)
# Convert predictions to a NumPy array
y_pred_train = y_pred_train.numpy()

## EVALUATION OF THE MODEL
# ######################## 

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train_train, y_pred_train)
print("Accuracy of the model:", accuracy)
# Filter training data for male survivors
male_survivors = train[(train['Sex_male'] == 1) & (train['Survived'] == 1)]
# Filter training data for female survivors
female_survivors = train[(train['Sex_male'] == 0) & (train['Survived'] == 1)]
# Calculate the percentage of male survivors
percentage_male_survived = (len(male_survivors) / len(train[train['Sex_male'] == 1])) * 100
# Calculate the percentage of female survivors
percentage_female_survived = (len(female_survivors) / len(train[train['Sex_male'] == 0])) * 100

# RESULTS
# Accuracy of the model 0.74
# Percentage of male survivors: 18.89%
#Percentage of female survivors: 74.20%