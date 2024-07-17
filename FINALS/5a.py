import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

class NaiveBayesClassifier:
    def _init_(self):
        self.prior = {}
        self.conditional = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            self.prior[c] = np.mean(y == c)
        for feature in X.columns:
            self.conditional[feature] = {}
            for c in self.classes:
                feature_values = X[feature][y == c]
                self.conditional[feature][c] = {'mean': np.mean(feature_values), 'std': np.std(feature_values)}

    def predict(self, X):
        y_pred = []
        for _, sample in X.iterrows():
            probabilities = {c: self.prior[c] for c in self.classes}
            for c in self.classes:
                for feature in X.columns:
                    mean = self.conditional[feature][c]['mean']
                    std = self.conditional[feature][c]['std']
                    probabilities[c] *= self._gaussian_pdf(sample[feature], mean, std)
            y_pred.append(max(probabilities, key=probabilities.get))
        return y_pred

    def _gaussian_pdf(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

# Load and preprocess data
df = pd.read_csv('dataset/titanic.csv')
df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Split data
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train, y_train = train.drop('Survived', axis=1), train['Survived']
X_test, y_test = test.drop('Survived', axis=1), test['Survived']

# Train and evaluate model
classifier = NaiveBayesClassifier()
classifier._init_()  # Correctly initialize the classifier
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Display results
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)
