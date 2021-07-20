from os import error
from nltk import metrics
import numpy as np
import random
import json
import torch
import torch.nn as nn
import tensorflow as tf
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
set(stopwords.words("english"))

from torch.utils.data import Dataset, DataLoader


from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []



for intent in intents['intents']:
    tag = intent['tag']
    
    tags.append(tag)
    for pattern in intent['patterns']:
        
        w = tokenize(pattern)
        
        print("Tokenized Words :", (w))
        all_words.extend(w)
        
        xy.append((w, tag))

filtered_sentence = []
stop_words = set(stopwords.words("english"))
for w in all_words:
    if w not in stop_words:
        filtered_sentence.append(w)
print("Filtered Tokenized Words :", (filtered_sentence))



ignore_words = ['?', '.', '!', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words ]
print("Stemmed Words :", (all_words))

all_words = sorted(set(all_words))
tags = sorted(set(tags))



X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
   
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)


num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print(X_train)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

   
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    
    def __len__(self):
        return self.n_samples
    

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

correct = 0
running_loss=0
total=0


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
     
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)

    running_loss += loss.item()
     
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
       
    train_loss=running_loss/len(train_loader)
    accu=100.*correct/total

    if (epoch+1) % 100 == 0:
        
         print (f'Epoch [{epoch+1}/{num_epochs}]'), print('Train Loss: %.4f | Accuracy: %.4f'%(train_loss,accu))




data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')

