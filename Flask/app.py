# app.py - a minimal flask api using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('x', type=float, default=False, required=False)
parser.add_argument('y', type=float, default=False, required=False)


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class Prediction(Resource):
    def get(self):
        args = parser.parse_args()
        x = args['x']
        x = float(x)
        result = x * x
        return {'result': result}

    def post(self):
        json_data = request.get_json(force=True)
        x = float(json_data['x'])
        result = x * x
        return {"result": result}, 200


class Add(Resource):
    def get(self):
        args = parser.parse_args()
        x = args['x']
        y = args['y']
        result = float(x) + float(y)
        return {"result": result}

    def post(self):
        json_data = request.get_json(force=True)
        x = json_data['x']
        y = json_data['y']
        result = float(x) + float(y)
        return {"result": result}, 200


api.add_resource(HelloWorld, '/hello')
api.add_resource(Prediction, '/prediction')
api.add_resource(Add, '/add')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
