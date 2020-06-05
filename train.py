import os
import json
import pandas
from optparse import OptionParser
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


def train(sample_pred_text):
    max_features = 26773
    dataframe = pandas.read_csv(csv_file, engine='python', quotechar='|', header=None)
    dataset = dataframe.sample(frac=1).values
    
  
    # Preprocess dataset  
    X = dataset[:,0]
    Y = dataset[:,1]

    for index, item in enumerate(X):
        # Quick hack to space out json elements
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        del reqJson['timestamp']
        del reqJson['headers']
        del reqJson['source']
        del reqJson['route']
        del reqJson['responsePayload']
        X[index] = json.dumps(reqJson, separators=(',', ':'))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    
    
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(X_train)
    
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    # Extract and save word dictionary
    word_dict_file = 'build/word-dictionary.json'

    if not os.path.exists(os.path.dirname(word_dict_file)):
        os.makedirs(os.path.dirname(word_dict_file))

    with open(word_dict_file, 'w') as outfile:
        json.dump(tokenizer.word_index, outfile, ensure_ascii=False)

   
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 400
    batch_size = 32
    
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
    from numpy import asarray
    from numpy import zeros
    
    embeddings_dictionary = dict()
    glove_file = open('data/glove.6B.100d.txt', encoding="utf8")
    
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()
    
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
            
        
    from keras.layers import Bidirectional
    import tensorflow as tf

    model = tf.keras.Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    
 
    history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=3,
          validation_data=[X_test, y_test])
    score = model.evaluate(X_test, y_test, verbose=1)

    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])  
    
    import matplotlib.pyplot as plt
    print(history.history)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    
    # Save model
    model.save_weights('model/model-weights.h5')
    model.save('model/model.h5')
    with open('model/model.json', 'w') as outfile:
        outfile.write(model.to_json())
        
    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    
    # #TEST    
    # instance = '{"timestamp":1502738715779,"method":"get","query":{"query":"Girls' Clothing"},"path":"/search","statusCode":200,"source":{"remoteAddress":"74.248.215.37","referer":"http://localhost:8002/enter"},"route":"/search","headers":{"host":"localhost:8002","accept-language":"en-us","accept-encoding":"gzip, deflate","connection":"keep-alive","accept":"*/*","referer":"http://localhost:8002/enter","cache-control":"no-cache","x-requested-with":"XMLHttpRequest"},"requestPayload":null,"responsePayload":"SEARCH"}'
    # reqJson = json.loads(instance, object_pairs_hook=OrderedDict)
    # del reqJson['timestamp']
    # del reqJson['headers']
    # del reqJson['source']
    # del reqJson['route']
    # del reqJson['responsePayload']
    # instance = json.dumps(reqJson, separators=(',', ':'))
    # tokenizer.fit_on_texts(instance)
    # instance = tokenizer.texts_to_sequences(instance)
    # flat_list = []
    # for sublist in instance:
    #     for item in sublist:
    #         flat_list.append(item)

    # flat_list = [flat_list]

    # instance = sequence.pad_sequences(flat_list, padding='post', maxlen=maxlen)

    # ress = model.predict(instance)
    # print(ress)
    


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--file', action="store", dest="file", help="data file")
    (options, args) = parser.parse_args()

    if options.file is not None:
        csv_file = options.file
    else:
        csv_file = 'db/dev-access.csv'
    train(csv_file)

      