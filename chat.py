import json
import random

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "NEU-CHATBOT"

def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    print(X)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    tag = tags[predicted.item()]
    print(tag)
    probs = torch.softmax(output, dim=1)
    print(probs)
    prob = probs[0][predicted.item()]
    print(prob)
    if prob.item() > 0.99:
       
        for intent in intents['intents']:
            if tag == intent["tag"]: 
                print(tag)
                
                print(intent['responses'])
                return random.choice(intent['responses'])
            
                               
    else:
        return "I do not understand..."
