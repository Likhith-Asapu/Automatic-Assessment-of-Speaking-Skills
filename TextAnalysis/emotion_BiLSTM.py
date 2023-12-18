
import torch
import torch.nn as nn
import torch.optim as optim
import fasttext.util
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from datasets import load_dataset
import fasttext
import fasttext.util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import RepeatedStratifiedKFold
import re

from tqdm import tqdm

# Load FastText word embeddings
import fasttext.util
# fasttext.util.download_model('en', if_exists='ignore')  # English
# ft = fasttext.load_model('/content/new_hing_emb')

dataset = load_dataset("Ar4ikov/iemocap_audio_text")

from google.colab import drive
drive.mount('/content/drive')

ft = fasttext.load_model('/content/drive/MyDrive/cc.en.100.bin')
def train_svm_classifier_for_emotion():

    # X,y = shuffle(X,y, random_state=42)
    X = []
    y = []

    def create_training_data(example):
        labels_encoded = {'ang':0, 'hap':1, 'sad':2, 'neu':3, 'dis':4, 'fea':5, 'sur':6, 'fru':7, 'exc':8, 'oth':9}
        if example['emotion'] == 'neu' or example['emotion'] == 'hap' or example['emotion'] == 'sad' or example['emotion'] == 'ang':
            sentence = example['to_translate']
            sentence = re.sub(r"[,.;@#?!&$-]",'',sentence)
            words = sentence.split()
            word_vectors = [ft.get_word_vector(word) for word in words]

            X.append(word_vectors)
            y.append(labels_encoded[example['emotion']])
    dataset.map(
        create_training_data
    )

    # Train_SVM(X,y,"IEMOCAP text")
    return X,y

X,y = train_svm_classifier_for_emotion()

max_sequence_length = max(len(sentence) for sentence in X)
# r = [torch.tensor(sentence).shape for sentence in X]
# print(r)
X = [torch.cat((torch.tensor(sentence), torch.zeros(max_sequence_length - len(sentence), 100))) for sentence in X]
X = torch.stack(X)
y = torch.tensor(y, dtype=torch.float)

# # Convert sentences to FastText word embeddings
# def sentence_to_embeddings(sentence):
#     embeddings = [ft[word] if word in ft else ft['<unk>'] for word in sentence]
#     return torch.tensor(embeddings, dtype=torch.float32)

# train_indexed_sentences = [sentence_to_embeddings(sentence.split()) for sentence in tqdm(train_sentences)]
# test_indexed_sentences = [sentence_to_embeddings(sentence.split()) for sentence in tqdm(test_sentences)]
# validation_indexed_sentences = [sentence_to_embeddings(sentence.split()) for sentence in tqdm(validation_sentences)]

# # Padding sequences to a fixed length
# max_sequence_length = max(len(sentence) for sentence in train_indexed_sentences + test_indexed_sentences + validation_indexed_sentences)

# train_padded_sequences = [torch.cat((sentence, torch.zeros(max_sequence_length - len(sentence), 300))) for sentence in train_indexed_sentences]
# test_padded_sequences = [torch.cat((sentence, torch.zeros(max_sequence_length - len(sentence), 300))) for sentence in test_indexed_sentences]
# validation_padded_sequences = [torch.cat((sentence, torch.zeros(max_sequence_length - len(sentence), 300))) for sentence in validation_indexed_sentences]

# # Split data into training and testing sets
# X_train = torch.stack(train_padded_sequences)
# y_train = torch.tensor(train_labels, dtype=torch.float)

# X_test = torch.stack(test_padded_sequences)
# y_test = torch.tensor(test_labels, dtype=torch.float)

# X_validation = torch.stack(validation_padded_sequences)
# y_validation = torch.tensor(validation_labels, dtype=torch.float)

# Create a PyTorch Dataset and DataLoader
class EmotionDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Build a BiLSTM model
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        lstm_out, (hn,cn) = self.bilstm(text)
        out_tensor = torch.cat([t for t in hn], dim=1)
        return self.fc(out_tensor)

# Training loop
def train_model(model, iterator, optimizer, criterion):
    model.train()

    for batch in iterator:
        text, labels = batch
        optimizer.zero_grad()
        predictions = model(text.to(device)).squeeze(1)
        loss = criterion(predictions.to(device), labels.long().to(device))
        loss.backward()
        optimizer.step()

# Evaluation
def evaluate_model(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            predictions = model(text.to(device)).squeeze(1)
            loss = criterion(predictions, labels.long().to(device))
            predicted_class = torch.argmax(predictions,dim = 1)
            correct += torch.sum(predicted_class.to(device) == labels.to(device)).item()
            total += len(predicted_class)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator), correct / total

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics


# Train the model
N_EPOCHS = 10

input_dim = 100  # FastText embedding dimension
hidden_dim = 64
output_dim = 4
lr = 0.01
X,y = shuffle(X,y, random_state=27)
best_f1 = 0
best_lr = 0
# Define loss and optimizer
for lr in [0.1,0.01,0.001,0.0001]:
  scores = []
  rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1,random_state=36851234)
  for train, test in rskf.split(X, y):
    model = BiLSTMModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_dataset = EmotionDataset(X[train], y[train])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = EmotionDataset(X[test], y[test])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    for epoch in tqdm(range(N_EPOCHS)):
      train_model(model, train_loader, optimizer, criterion)
      validation_loss,validation_accuracy = evaluate_model(model, test_loader, criterion)
      # print(f'Validation Loss: {validation_loss:.4f}')
      # print(f'Validation Accuracy: {validation_accuracy:.4f}')

    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X[test]).to(device))
        y_pred = torch.argmax(outputs, dim=1)
    try:
        f1score = metrics.f1_score(y[test], y_pred.cpu())
        scores.append(f1score)
    except:
        f1score = metrics.accuracy_score(y[test], y_pred.cpu())
        scores.append(f1score)

  f1mean = np.array(scores).mean()
  if f1mean > best_f1:
    best_f1 = f1mean
    best_lr = lr
  print(f"{lr} - {f1mean}")

print(f"FINISHED Training on {dataset}")
print(f"Best lr value = {best_lr}, f1 = {best_f1}")

