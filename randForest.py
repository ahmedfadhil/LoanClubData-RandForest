import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pydot
from sklearn.tree import export_graphviz

loans = pd.read_csv('loan_data.csv')

loans.head()
loans.info()
loans.describe()
plt.figure(figsize=(10, 6))
loans[loans['credit.policy'] == 1]['fico'].hist(bins=35, color='blue',
                                                label='Credit Policy = 1',
                                                alpha=0.6)
loans[loans['credit.policy'] == 0]['fico'].hist(bins=35, color='red',
                                                label='Credit Policy = 0',
                                                alpha=0.6)
plt.xlabel('FICO')

plt.legend()

# Not fully paid the load back

plt.figure(figsize=(10, 6))
loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=35, color='blue',
                                                 label='not fully paid = 1',
                                                 alpha=0.6)
loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=35, color='red',
                                                 label='not fully paid = 0',
                                                 alpha=0.6)
plt.xlabel('FICO')

plt.legend()
plt.figure(figsize=(11, 7))
sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')

sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')

plt.figure(figsize=(11, 7))
sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')

# Decision tree and random forest model

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

final_data.head()

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Random Forest Model

rfc = RandomForestClassifier(n_estimators=300)

rfc.fit(X_train, y_train)

predictions = rfc.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
























