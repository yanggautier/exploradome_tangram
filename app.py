# app.py - a minimal flask api using flask_restful
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from detect_video import * 

app = Flask(__name__)
api = Api(app)

# Define parser and request args
parser = reqparse.RequestParser()
parser.add_argument('link', type=str, default=False, required=False)

class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

class Detectvideo(Resource):
    def get(self):
        json_data = request.get_json(force=True)
        link = float(json_data['link'])
        detect_video(link)
        return {"result": "OK"}, 200

api.add_resource(HelloWorld, '/hello')
api.add_resource(Detectvideo, '/detection')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
