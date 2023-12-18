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
from TrainANN import *
from TrainSVM import *
dataset = load_dataset("Ar4ikov/iemocap_audio_text")
ft = fasttext.load_model('cc.en.100.bin')

    
def train_svm_classifier_for_emotion():
   
    # X,y = shuffle(X,y, random_state=42)
    X = []
    y = []

    def create_training_data(example):
        sentence = example['to_translate']
        sentence = re.sub(r"[,.;@#?!&$-]",'',sentence)
        X.append(ft.get_sentence_vector(sentence))
        
        if example['activation'] < 2.53:
            y.append(0)
        elif example['activation'] < 3.6:
            y.append(1)
        else:
            y.append(2)
        
    dataset.map(
        create_training_data
    )

    # Train_SVM(X,y,"IEMOCAP text")
    Train_ANN(X,y,"IEMOCAP text",3)
    

train_svm_classifier_for_emotion()