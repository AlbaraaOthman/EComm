import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.tree import plot_tree
#Following this tutorial https://www.datacamp.com/tutorial/decision-tree-classification-python

#List of headings of data
headings = ['Time spent on searching for item (s)','How many times add to basket was clicked (from search page)','Time spent on individual product page','How many times add to basket was clicked (from individual product page)','How many times go to basket was clicked','How long spent on checkout','Still buying?','Basket Value']
#Read the data
data = pd.read_csv("Dataset_Less_Time_on_Search.csv", header=1, names=headings)
#Features we are looking at
features = ['Time spent on searching for item (s)','How many times add to basket was clicked (from search page)','Time spent on individual product page','How many times add to basket was clicked (from individual product page)','How many times go to basket was clicked','How long spent on checkout','Basket Value']
X = data[features]
#Our Churn
target = 'Still buying?'
y = data[target]

#Split into an 80 20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% training and 20% test, random state 42 means the same data will be split each time

#Run the original decision tree
decisionTree = DecisionTreeClassifier()
decisionTree = decisionTree.fit(X_train,y_train)

#Predict y values for test case
ypred = decisionTree.predict(X_test)

#Use forrest regression to find the importance of each section
importances = decisionTree.feature_importances_

importance_match = {}

#https://www.geeksforgeeks.org/donut-chart-using-matplotlib-in-python/

#Store the importances in a dictionary
for i in range(len(importances)):
    importance_match[features[i]] = importances[i]

#Pick important features
take = round(len(importance_match) * .7) #Take top 70%
imp_sorted = sorted(importance_match, key=importance_match.get, reverse=True)[:take] #sort list names
values = []
#Get the values for those names
for i in range(len(imp_sorted)):
    values.append(importance_match[imp_sorted[i]])

#Train with top features
decisionTree_best = DecisionTreeClassifier()
Xbesttrain= X_train[imp_sorted]
Xbesttest = X_test[imp_sorted]

decisionTree_best = decisionTree_best.fit(Xbesttrain,y_train)
ypred_best = decisionTree_best.predict(Xbesttest)
print("Top Accuracy",metrics.accuracy_score(y_test,ypred_best)) #Show Accuracy
print(classification_report(y_test,ypred_best)) # Show classification report
print(confusion_matrix(y_test,ypred_best)) # Show confusion matrix

plt.figure(figsize=(20,10))
plot_tree(decisionTree_best, feature_names=imp_sorted, class_names=['Not Buying', 'Buying'], filled=True)
plt.show()


# fig = plt.figure(figsize=(5,5)) #Donut graph of size 5 x 5
#
#
# #Create the pi chart
# plt.pie(importances, startangle=90,autopct='%1.1f%%',pctdistance=0.85,  textprops={'fontsize': 16})
#
# #Add a centre circle which is blank (to make it donut)
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
#
#
# fig = plt.gcf()
#
# #add the circle
# fig.gca().add_artist(centre_circle)
# plt.text(0, 0, 'Survey Questions Importance', horizontalalignment='center', verticalalignment='center', fontsize=18)
# plt.legend(features, loc="center left",bbox_to_anchor=[1, 0.5], fontsize=10) #position nicely
# plt.show()




