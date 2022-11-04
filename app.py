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
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('feat1', type=int)
            parser.add_argument('feat2', type=int)
            parser.add_argument('feat3', type=int)
            parser.add_argument('feat4', type=int)
            
            params = parser.parse_args()
            data = list(params.values())
            print([data])

            prediction = model.predict([data])
            class_prediction = labels[prediction[0]]

            return jsonify(prediction=class_prediction)

        except ValueError as e: #Error handling
        # return jsonify({"Error":str(e)})
            if('RandomForestClassifier does not accept missing values' in str(e)):
                return jsonify({"Error":"Invaid Input! Model is expecting 4 features as input"})
            if('could not convert string to float' in str(e)):
                return jsonify({"Error":"Invaid Input! Model is expecting Interger features as input"})

api.add_resource(e_shop, '/predict', endpoint='predict')

if __name__ == '__main__':
    app.run(debug=True)

# api - http://127.0.0.1:5000/predict?feat1=-10&feat2=20&feat3=4&feat4=-12




