from flask import Flask
from flask_restx import Api, Resource, reqparse, fields
from werkzeug.datastructures import FileStorage
from PIL import Image

app = Flask(__name__)
api = Api(app)

upload_parser = api.parser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)


@api.route('/upload/')
@api.expect(upload_parser)
class Upload(Resource):
    def post(self):
        args = upload_parser.parse_args()
        uploaded_file = args['file']  # This is FileStorage instance
        img = Image.open(uploaded_file)
        url = img.size()

        return {'url': url}, 201


if __name__ == "__main__":
    app.run(debug=True)
