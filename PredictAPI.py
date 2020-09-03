from flask import Flask, request, jsonify # loading in Flask
import pandas as pd # loading pandas for reading csv
from gevent.pywsgi import WSGIServer
import pickle
# creating a Flask application
app = Flask(__name__)
model_path = '\Desktop\MachineLearning'
diabetesmodel = pickle.load(open(model_path + 'diabetes_prediction2.pkl', 'rb'))
kidneymodel = pickle.load(open(model_path + 'kidney_prediction2.pkl', 'rb'))
manmodel = pickle.load(open(model_path + 'man_prediction.pkl', 'rb'))
femalemodel = pickle.load(open(model_path + 'female_prediction.pkl', 'rb'))

# creating predict url and only allowing post requests.
@app.route('/diabetespredict', methods=['POST'])
def diabetespredict():   
    data = request.get_json(force=True)
    prediction = diabetesmodel.predict([list(data.values())])
    output = str(prediction[0])
    return output

# creating predict url and only allowing post requests.
@app.route('/kidneypredict', methods=['POST'])
def kidneypredict():   
    data = request.get_json(force=True)
    prediction = kidneymodel.predict([list(data.values())])
    output = str(prediction[0])
    return output


@app.route('/manpredict', methods=['POST'])
def manpredict():   
    data = request.get_json(force=True)
    prediction = manmodel.predict([list(data.values())])
    output = str(prediction[0])
    return output

@app.route('/femalepredict', methods=['POST'])
def femalepredict():   
    data = request.get_json(force=True)
    prediction = femalemodel.predict([list(data.values())])
    output = str(prediction[0])
    return output
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000) # worked
    # also worked
    #from werkzeug.serving import run_simple
    #run_simple('localhost', 5000, app)
