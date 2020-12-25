import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics

features = pd.read_csv('arash_zhdr_1k.csv')
features.head()

X = features[['bl_num', 'bl_size', 'he_num', 'he_size']].astype(float)
Y = features['dr'].astype(int)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)

clf = RandomForestClassifier(n_estimators=150)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)



confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()
