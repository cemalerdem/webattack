from flask import Flask
from flask.wrappers import Response
from flask import Flask, jsonify, request
from predict import predict_model
import json
# First Initilize the Flask Applicaiton

app = Flask(__name__)

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
@app.route('/movies')
def hello():
    return jsonify(movies)

@app.route('/movies', methods=['POST'])
def add_movie():
    movie = request.get_json()
    movies.append(movie)
    return {'id': len(movies)}, 200

@app.route('/predict', methods=['POST'])
def keras_predict():
    text = request.get_json()
    res = json.dumps(text)
    result = predict_model(res)
    return {'Result': result[0][0]}, 200

@app.route('/testt', methods=['GET'])
def test_keras_predict():    
    res = json.dumps(instance)
    result = predict_model(res)
    return {'Result': result[0][0]}, 200

@app.route('/movies/<int:index>', methods=['DELETE'])
def delete_movie(index):
    movies.pop(index)
    return 'None', 200

@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"

@app.route("/supper", methods=["GET"])
def heath():
    return Response(json.dumps({"status":"Uber Super Cemal"}), status=200, mimetype='application/json')

@app.route("/predict", methods=["GET","POST"])
def predict():
    data = {"success": False}
    # get the request parameters
    params = request.json
    if (params == None):
        params = request.args
    # if parameters are found, echo the msg parameter 
    if (params != None):
        data["response"] = params.get("msg")
        data["success"] = True
    # return a response in json format 
    return jsonify(data)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5001)


