# 🚢 Titanic Survival Prediction (Advanced Dataset)

This project is a machine learning solution to the classic Titanic survival prediction problem — but with a twist. Instead of using the basic dataset, we explore an **enriched version** containing extra fields like `lifeboat number`, `body ID`, and `home destination`. The goal is to train a model that can predict whether a passenger survived or not based on various personal and travel details.


## 💡 Project Overview

The sinking of the Titanic is one of the most infamous shipwrecks in history. The tragedy offers a rich dataset to explore classification techniques and understand how factors like **age, gender, passenger class, and fare** may have affected a person’s chance of survival.

We take it further by including **additional real-world features** to test model performance.


## 📊 Features Used

| Feature         | Description |
|----------------|-------------|
| `pclass`        | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `sex`           | Gender (male/female) |
| `age`           | Age of the passenger |
| `sibsp`         | # of siblings/spouses aboard |
| `parch`         | # of parents/children aboard |
| `fare`          | Fare paid |
| `embarked`      | Port of embarkation (C, Q, S) |
| `family_size`   | Engineered feature = `sibsp + parch + 1` |
| _Dropped_: `ticket`, `name`, `cabin`, `boat`, `body`, `home.dest` (too sparse or redundant)

## 🧠 Technologies Used

- **Python**
- **pandas**, **numpy** for data handling
- **seaborn**, **matplotlib** for visualization
- **scikit-learn** for machine learning (Random Forest)
- **joblib** for model saving


## ⚙️ How It Works

### 1. Data Cleaning & Preprocessing
- Missing values in age, fare, and embarked handled
- Text fields like `sex` and `embarked` converted to numbers
- Unnecessary columns dropped for simplicity

### 2. Feature Engineering
- Created a new feature: `family_size`
- Removed features with too many missing values

### 3. Model Training
- Split dataset into training and validation sets (80/20)
- Trained a **RandomForestClassifier**
- Achieved ~80–85% accuracy on validation set

### 4. Final Predictions
- Trained model saved as `titanic_model.pkl`
- Submission file created as `submission.csv`


## 🛠 Project Structure

# Titanic-Survival-Prediction
