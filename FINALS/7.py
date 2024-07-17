import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder

# Load the weather dataset from the local file
file_path = r"dataset/weather_forecast.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())

# Preprocessing: convert categorical variables to numerical using one-hot encoding
encoder = OneHotEncoder(drop='first')
X_encoded = encoder.fit_transform(df.drop('Play', axis=1)).toarray()
y = df['Play']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize and fit the decision tree classifier with ID3 algorithm
clf_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_id3.fit(X_train, y_train)

# Visualize the ID3 decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf_id3, filled=True, feature_names=encoder.get_feature_names_out(['Outlook', 'Temperature', 'Humidity', 'Windy']), class_names=['No', 'Yes'])
plt.show()

# Predict and evaluate the ID3 model
y_pred_id3 = clf_id3.predict(X_test)
print("ID3 Algorithm Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_id3)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_id3)}")
print("Cross-Validation Scores (ID3):", cross_val_score(clf_id3, X_encoded, y, cv=5))
print("Mean CV Accuracy (ID3):", cross_val_score(clf_id3, X_encoded, y, cv=5).mean())

# Initialize and fit the decision tree classifier with CART algorithm
clf_cart = DecisionTreeClassifier(criterion='gini', random_state=42)
clf_cart.fit(X_train, y_train)

# Visualize the CART decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf_cart, filled=True, feature_names=encoder.get_feature_names_out(['Outlook', 'Temperature', 'Humidity', 'Windy']), class_names=['No', 'Yes'])
plt.show()

# Predict and evaluate the CART model
y_pred_cart = clf_cart.predict(X_test)
print("CART Algorithm Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_cart)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_cart)}")
print("Cross-Validation Scores (CART):", cross_val_score(clf_cart, X_encoded, y, cv=5))
print("Mean CV Accuracy (CART):", cross_val_score(clf_cart, X_encoded, y, cv=5).mean())

