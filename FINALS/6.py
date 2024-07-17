import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

df = pd.read_csv('./dataset/glass.csv')
X = df.drop('Type', axis=1).values
y = df['Type'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)
accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)