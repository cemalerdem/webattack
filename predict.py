import sys
import os
import json
import pandas
import numpy
import optparse
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def predict_model(sample_pred_text):   
    
    maxlen = 400
    reqJson = json.loads(sample_pred_text, object_pairs_hook=OrderedDict)

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

    model = tf.keras.models.load_model('saved_model/securitai-lstm-model.h5',compile=False)
    model.load_weights('saved_model/securitai-lstm-weights.h5')
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy']) 
    

    instance = sequence.pad_sequences(flat_list, padding='post', maxlen=maxlen)
    prediction = model.predict(instance)
    # encodedNumpyData = json.dumps(prediction, cls=NumpyArrayEncoder)
    return prediction.tolist()


