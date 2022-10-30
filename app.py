from flask import Flask,jsonify
from flask_restful import Resource, Api, reqparse
import pickle

#load the model
model = pickle.load(open('model.pkl','rb'))

labels = ['Rockstar', '2K', 'Zynga']

app = Flask(__name__)
api = Api(app)

class e_shop(Resource):
    def get(self):

        parser = reqparse.RequestParser()
        parser.add_argument('key1', type=int)
        parser.add_argument('key2', type=int)
        parser.add_argument('key3', type=int)
        parser.add_argument('key4', type=int)
        
        params = parser.parse_args()
        data = list(params.values())
        print([data])

        prediction = model.predict([data])
        class_prediction = labels[prediction[0]]

        return jsonify(prediction=class_prediction)

api.add_resource(e_shop, '/predict', endpoint='predict')

if __name__ == '__main__':
    app.run(debug=True)
