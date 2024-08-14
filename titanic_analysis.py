import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Display basic information (including data sample)
print(df.head())
print(df.tail())
print(df.info())

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing Age with median
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode
df.drop(columns=['Cabin'], inplace=True)  # Drop Cabin due to too many missing values

# Verify missing values have been handled
print("Missing values after handling:")
print(df.isnull().sum())

# Feature Engineering
# Create a new feature 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Create a new feature 'IsAlone'
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Convert 'Sex' to numeric (binary)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Display the updated dataframe
print(df.head())

# Exploratory Data Analysis (EDA)
# Plot survival rates by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Rates by Gender')
plt.show()

# Plot survival rates by class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Rates by Class')
plt.show()

# Plot survival rate by FamilySize
sns.catplot(x='FamilySize', y='Survived', data=df, kind='point')
plt.title('Survival Rate by Family Size')
plt.show()

# Plot survival rate by IsAlone
sns.barplot(x='IsAlone', y='Survived', data=df)
plt.title('Survival Rate by IsAlone')
plt.show()

# Plot distribution of age by survival
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True, label='Survived', color='green')
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True, label='Not Survived', color='red')
plt.legend()
plt.title('Age Distribution by Survival')
plt.show()
