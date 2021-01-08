import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageOps

# import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics

from flask import Flask, send_file, jsonify
from flask_restx import Api, Resource, reqparse, fields
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
api = Api(app)
parser = reqparse.RequestParser()
# parser.add_argument('rate', type=int, help='Rate cannot be converted')
parser.add_argument('metric')
parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


# todo = api.model('Process', {
#     'metric': fields.String(required=True, description='The task details')
# })

features = pd.read_csv('../arash_yolo_isbi_ts.csv')
features.head()

X = features[['bl_num', 'bl_size', 'he_num', 'he_size', 'laser_num', 'laser_size']].astype(float)
Y = features['dr'].astype(int)

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=1400)

clf = RandomForestClassifier(n_estimators=95)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
# sn.heatmap(confusion_matrix, annot=True)

# print(' Accuracy: ', metrics.accuracy_score(y_test, y_pred))
# print(' QKappa Score: ', metrics.cohen_kappa_score(y_test, y_pred, weights='quadratic'))
# plt.show()
clf = RandomForestClassifier(n_estimators=95)
clf.fit(X_train, y_train)


@api.expect(parser)
class Process(Resource):

    def get(self):
        args = parser.parse_args()
        uploaded_file = args['file']

        y_pred = clf.predict(X_test)
        if args['metric'] == "acc":
            acc = metrics.accuracy_score(y_test, y_pred)
        else:
            acc = 0.0
        return {'accuracy': str(acc)}


    def post(self):
        args = parser.parse_args()
        uploaded_file = args['file']
        # img = Image.open(uploaded_file.stream)
        # g_img=ImageOps.grayscale(img)
        y_pred = clf.predict(X_test[30].reshape(1, -1))

        return jsonify({'dr':str(y_pred[0])})#jsonify({'msg': 'success', 'size': [img.width, img.height]})





api.add_resource(Process, '/process')

if __name__ == "__main__":
    app.run(debug=True)
