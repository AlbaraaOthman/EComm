import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#Following this tutorial https://www.datacamp.com/tutorial/decision-tree-classification-python

headings = ['Searching for Item(s','Times add to basket was clicked','Time spent on individual product page','How many times add to basket was clicked from individual product page','How many times go to basket was clicked','How long spent on the checkout','Purchase Completed?']
data = pd.read_csv("Dataset.csv", header=1, names=headings)
features = ['Searching for Item(s','Times add to basket was clicked','Time spent on individual product page','How many times add to basket was clicked from individual product page','How many times go to basket was clicked','How long spent on the checkout']
X = data[features]
target = ['Purchase Completed?']
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test


decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(X_train,y_train)

ypred = decisionTree.predict(X_test)

importances = decisionTree.feature_importances_

importance_match = {}
for i in range(len(importances)):
    importance_match[features[i]] = importances[i]

print(importance_match)

fig = plt.figure(figsize=(10,10))

plt.bar(features, importances, width=0.4)
plt.xlabel("Survey Questions")
plt.ylabel("Importance Values")
plt.title("The Importances of Different Sections of Our Website")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
print("Accuracy:",metrics.accuracy_score(y_test, ypred))



