import random
import json
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
import nltk
from nltk.stem import WordNetLemmatizer
import os


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('models/words.pkl', 'rb'))
classes = pickle.load(open('models/classes.pkl', 'rb'))
model = load_model('models/ecoturismo_model.model')

# create app object using Flask class
app = Flask(__name__)






print("Bot is running")

@app.route('/')
def home():
    print('render')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # get string inputs from browser and converts to numerical values

    print('a')
    message = [str(x) for x in request.form.values()]
    print(message)

    print("predicting class..")
    ints = predict_class(message[0].lower())

    print("prediction ready")
    
    res = get_response(ints, intents)

    print("response ready")

    return render_template('index.html', prediction_text=res["name"], image=res["img"], description=res['dsc'],tag1=res["tag1"], tag2=res["tag2"])

def predict_class(inp):
    print("start predict")
    
    bag = bag_of_words(inp)
    res = model.predict(np.array([bag]))[0]
    
    print("got res")
    
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# extract words from the input
def abstraction(inp):
    inp_words = nltk.word_tokenize(inp)
    inp_words = [lemmatizer.lemmatize(word) for word in inp_words]
    return inp_words

def bag_of_words(inp):
    inp_words = abstraction(inp)
    bag = [0] * len(words)
    for x in inp_words:
        for i, word in enumerate(words):
            if word == x:
                bag[i] = 1

    return np.array(bag)

if __name__ == "__main__":
    app.run()
