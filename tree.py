import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Following this tutorial https://www.datacamp.com/tutorial/decision-tree-classification-python

headings = ['Searching for Item(s','Times add to basket was clicked from search page','Time spent on individual product page','How many times add to basket was clicked from individual product page','How many times go to basket was clicked','How long spent on the checkout','Purchase Completed?']
data = pd.read_csv("Dataset_Less_Time_on_Search.csv", header=1, names=headings)
features = ['Searching for Item(s','Times add to basket was clicked from search page','Time spent on individual product page','How many times add to basket was clicked from individual product page','How many times go to basket was clicked','How long spent on the checkout']
X = data[features]
target = ['Purchase Completed?']
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test


decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(X_train,y_train)

ypred = decisionTree.predict(X_test)

importances = decisionTree.feature_importances_

importance_match = {}

#https://www.geeksforgeeks.org/donut-chart-using-matplotlib-in-python/

for i in range(len(importances)):
    importance_match[features[i]] = importances[i]

#Pick important features


take = round(len(importance_match) * .7)
imp_sorted = sorted(importance_match, key=importance_match.get, reverse=True)[:take]
print(imp_sorted)
values = []
for i in range(len(imp_sorted)):
    values.append(importance_match[imp_sorted[i]])

#Train with top features
decisionTree_best = DecisionTreeClassifier()
Xbesttrain= X_train[imp_sorted]
Xbesttest = X_test[imp_sorted]

decisionTree_best = decisionTree_best.fit(Xbesttrain,y_train)
ypred_best = decisionTree_best.predict(Xbesttest)

print("Accuracy:",metrics.accuracy_score(y_test, ypred))
print("Top 4 Accuracy",metrics.accuracy_score(y_test,ypred_best))
confusion = confusion_matrix(y_test,ypred_best)
print(confusion)
fig = plt.figure(figsize=(5,5))

plt.pie(importances, startangle=90,autopct='%1.1f%%',pctdistance=0.85,  textprops={'fontsize': 16})

centre_circle = plt.Circle((0, 0), 0.70, fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)
plt.text(0, 0, 'Survey Questions Importance', horizontalalignment='center', verticalalignment='center', fontsize=18)
plt.legend(features, loc="center left",bbox_to_anchor=[1, 0.5], fontsize=10)
plt.show()




