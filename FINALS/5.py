import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    def __init__(self):
        self.prior = {}
        self.conditional = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            self.prior[c] = np.mean(y == c)
        for feature in X.columns:
            self.conditional[feature] = {
                c: {'mean': np.mean(X[feature][y == c]), 'std': np.std(X[feature][y == c])} 
                for c in self.classes
            }

    def predict(self, X):
        def _gaussian_pdf(x, mean, std):
            exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
            return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

        y_pred = []
        for _, sample in X.iterrows():
            probabilities = {c: self.prior[c] for c in self.classes}
            for feature in X.columns:
                for c in self.classes:
                    mean = self.conditional[feature][c]['mean']
                    std = self.conditional[feature][c]['std']
                    probabilities[c] *= _gaussian_pdf(sample[feature], mean, std)
            y_pred.append(max(probabilities, key=probabilities.get))
        return y_pred

# Load and preprocess dataset
df = pd.read_csv('./dataset/titanic.csv')[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df.fillna({'Age': df['Age'].median(), 'Fare': df['Fare'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived', axis=1), df['Survived'], test_size=0.2)

# Train Naive Bayes Classifier
classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = classifier.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", np.mean(y_pred == y_test))
