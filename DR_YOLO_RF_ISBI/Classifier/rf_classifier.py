import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics

features = pd.read_csv('../arash_yolo_isbi_ts.csv')
features.head()

X = features[['bl_num', 'bl_size', 'he_num', 'he_size', 'laser_num', 'laser_size']].astype(float)
Y = features['level'].astype(int)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=1400)

clf = RandomForestClassifier(n_estimators=95)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print(' Accuracy: ', metrics.accuracy_score(y_test, y_pred))
print(' QKappa Score: ', metrics.cohen_kappa_score(y_test, y_pred, weights='quadratic'))
plt.show()
