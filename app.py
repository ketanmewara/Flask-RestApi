from flask import Flask, request, jsonify
import pickle

#load the model
model = pickle.load(open('model.pkl','rb'))

labels = ['Rockstar', '2K', 'Zynga']

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Welcome"

# endpoint 
@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        feat1 = request.args.get('feat1')
        feat2 = request.args.get('feat2')
        feat3 = request.args.get('feat3')
        feat4 = request.args.get('feat4')

        data = [[feat1,feat2,feat3,feat4]]
        
        prediction = model.predict(data)
        class_prediction = labels[prediction[0]]
        return {"prediction" : class_prediction}
            
    except ValueError as e: #Error handling
        # return jsonify({"Error":str(e)})
        if('RandomForestClassifier does not accept missing values' in str(e)):
            return jsonify({"Error":"Invaid Input! Model is expecting 4 features as input"})
        if('could not convert string to float' in str(e)):
            return jsonify({"Error":"Invaid Input! Model is expecting Interger features as input"})
    

if __name__ == '__main__':
    app.run(debug=True)