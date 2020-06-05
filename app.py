from flask import Flask
from flask.wrappers import Response
from flask import Flask, jsonify, request
import json
from json import JSONEncoder
import pandas
import numpy
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
import tensorflow as tf
import asyncio
# First Initilize the Flask Applicaiton


app = Flask(__name__)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


async def predict_model(sample_pred_text):   
    
    maxlen = 400
    reqJson =  json.loads(sample_pred_text, object_pairs_hook=OrderedDict)

    instance = json.dumps(reqJson, separators=(',', ':'))

    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(sample_pred_text)
    instance = tokenizer.texts_to_sequences(sample_pred_text)

    flat_list = []
    for sublist in instance:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    instance = pad_sequences(flat_list, padding='post', maxlen=maxlen)

    model = await tf.keras.models.load_model('model.h5',compile=False)
    await model.load_weights('model-weights.h5')
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) 
    

    instance = sequence.pad_sequences(flat_list, padding='post', maxlen=maxlen)
    prediction = await  model.predict(instance)
    # encodedNumpyData = json.dumps(prediction, cls=NumpyArrayEncoder)
    return prediction.tolist()

# Create some test data for our catalog in the form of a list of dictionaries.
instance = '{"timestamp":1502738411514,"method":"post","query":{},"path":"/login","statusCode":401,"source":{"remoteAddress":"100.44.63.104","referer":"http://localhost:8002/enter"},"route":"/login","headers":{"host":"localhost:8002","accept-language":"en-us","accept-encoding":"gzip, deflate","connection":"keep-alive","accept":"*/*","referer":"http://localhost:8002/enter","cache-control":"no-cache","x-requested-with":"XMLHttpRequest","content-type":"application/json","content-length":"56"},"requestPayload":{"username":"OR 1=1; # limit 1","password":"pizzal0v32"},"responsePayload":{"statusCode":401,"error":"Unauthorized","message":"Invalid Login"}}'
movies = [
    {
        "name": "The Shawshank Redemption",
        "casts": ["Tim Robbins", "Morgan Freeman", "Bob Gunton", "William Sadler"],
        "genres": ["Drama"]
    },
    {
       "name": "The Godfather ",
       "casts": ["Marlon Brando", "Al Pacino", "James Caan", "Diane Keaton"],
      "genres": ["Crime", "Drama"]
   }
]

@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route("/supper", methods=["GET"])
def heath():
    return Response(json.dumps({"status":"Uber Super Cemal"}), status=200, mimetype='application/json')

@app.route('/movies')
def hello():
    return jsonify(movies)

@app.route('/movies', methods=['POST'])
def add_movie():
    movie = request.get_json()
    movies.append(movie)
    return {'id': len(movies)}, 200

@app.route('/predict', methods=['POST'])
async def keras_predict():
    text = request.get_json()
    res = json.dumps(text)
    result = await predict_model(res)
    return 'OK'

@app.route('/testt', methods=['GET'])
def test_keras_predict():    
    res = json.dumps(instance)
    result = predict_model(res)
    return {'Result': result[0][0]}, 200

@app.route('/movies/<int:index>', methods=['DELETE'])
def delete_movie(index):
    movies.pop(index)
    return 'None', 200





if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)


