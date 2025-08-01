# # Untitled-1.py
# # Untitled-1.p
import pandas as pd
import numpy as np
#Data visualation 
import seaborn as sns
import matplotlib.pyplot as plt
#machine learnng 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
df = pd.read_csv("Titanic Dataset.csv")  # Use your actual filename
df.head()
#data Handle missing values
#Data cleaning
df['age'].fillna(df['age'].median(), inplace=True)
df['fare'].fillna(df['fare'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop non-useful or heavily missing columns
df.drop(['cabin', 'ticket', 'name', 'boat', 'body', 'home.dest'], axis=1, inplace=True)
#Feature Engineering
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
#family_size
df['family_size'] = df['sibsp'] + df['parch'] + 1
#Exploratory Data Analysis (EDA)
#Survival rate by gender and class

sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x='pclass', y='survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()
#Train/Test Split

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#Model Building (Random Forest)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, preds))
print(confusion_matrix(y_val, preds))
print(classification_report(y_val, preds))