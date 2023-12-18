from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 
import numpy as np
from os import listdir
from os.path import isfile, join
from sklearn.svm import SVC
import plotly.graph_objs as go 
import plotly
import glob
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from TrainSVM import *
from TrainANN import *

def emoDB_get_labels(labels):
    '''
    LABELS :-
    ANGER   - W - 0
    HAPPY   - F - 2
    SAD     - T - 0
    Neutral - N - 1
    BORED   - L - 0
    DISGUST - E - 0
    ANXIETY - A - 0
    '''
    labels_encoded = {'W':0, 'F':2, 'T':0, 'N':1, 'L':0, 'E':0, 'A':0}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes

def emovo_get_labels(labels):
    '''
    LABELS :-
    ANGER   - rab
    HAPPY   - gio
    SAD     - tri
    Neutral - neu
    DISGUST - dis   
    FEAR    - pau   
    SURPRISE- sor  
    '''
    labels_encoded = {'rab':0, 'gio':2, 'tri':0, 'neu':1, 'dis':0, 'pau':0, 'sor':2}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes

def ravdess_get_labels(labels):
    '''
    LABELS :-
    ANGER   - 05
    HAPPY   - 03
    SAD     - 04
    Neutral - 01
    DISGUST - 07    
    FEAR    - 06    
    SURPRISE- 08    
    CALM    - 02    
    '''
    labels_encoded = {'05':0, '03':2, '04':0, '01':1, '07':0, '06':0, '08':2, '02' : 2}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes


def savee_get_labels(labels):
    '''
    LABELS :-
    ANGER   - a
    HAPPY   - h
    SAD     - sa
    Neutral - n
    DISGUST - d     
    FEAR    - f     
    SURPRISE- su   
    '''
    labels_encoded = {'a':0, 'h':2, 'sa':0, 'n':1, 'd':0, 'f':0, 'su':2}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes

def iemocap_get_labels(labels):
    
    classes = []
    for label in labels:
        if label < 2.5:
            classes.append(0)
        elif label < 4:
            classes.append(1)
        else:
            classes.append(2)
    return classes

def check_label(label,db_type):
    if db_type == 'emoDB':
        labels_encoded = {'W':0, 'F':2, 'T':0, 'N':1, 'L':0, 'E':0, 'A':0}
    elif db_type == 'emovo':
        labels_encoded = {'rab':0, 'gio':2, 'tri':0, 'neu':1, 'dis':0, 'pau':0, 'sor':2}
    elif db_type == 'ravdess':
        labels_encoded = {'05':0, '03':2, '04':0, '01':1, '07':0, '06':0, '08':2, '02' : 2}
    elif db_type == 'savee':
        labels_encoded = {'a':0, 'h':2, 'sa':0, 'n':1, 'd':0, 'f':0, 'su':2}
    elif db_type == 'iemocap':
        return True
    if labels_encoded[label] == None:
        return False
    return True

# get label/class of the file from the file path
# (returns label as an integer)
def get_label_numbers(labels,db_type):
    if db_type == 'emoDB':
        return emoDB_get_labels(labels)
    elif db_type == 'emovo':
        return emovo_get_labels(labels)
    elif db_type == 'ravdess':
        return ravdess_get_labels(labels)
    elif db_type == 'savee':
        return savee_get_labels(labels)
    elif db_type == 'iemocap':
        return iemocap_get_labels(labels)

# get label/class of the file from the file path
# (returns label as an integer)
def get_label(file_path,db_type):
    if db_type == 'emoDB':
        return file_path[-6]
    elif db_type == 'emovo':
        return file_path[6:9]
    elif db_type == 'ravdess':
        return file_path[-18:-16]
    elif db_type == 'savee':
        if file_path[-8:-7] == 's':
            return file_path[-8:-6]
        else:
            return file_path[-7:-6]
        

# read audio data from file 
# (returns segement feature vectors as a numpy array)
def get_features(file_path):
    fs, s = aIO.read_audio_file(file_path)

    if len(s.shape) > 1:
        s = np.mean(s, axis=1)
    mt, st, mt_n = aF.mid_feature_extraction(s, fs, 3 * fs, 3 * fs, 
                                             0.05 * fs, 0.05 * fs)
    return mt.T

def get_features_iemocap(fs,s):
    
    if len(s.shape) > 1:
        s = s[:,0]
    mt, st, mt_n = aF.mid_feature_extraction(s, fs, 3 * fs, 3 * fs, 
                                            0.05 * fs, 0.05 * fs)
    return mt.T

# read data from folder
# (returns feature vectors and labels as numpy arrays)
def read_folder(folder_path):
    features = []
    labels = []
    if folder_path == "iemocap":
        dataset = load_dataset('Ar4ikov/iemocap_audio_text')
        files = dataset['train']
        for file in tqdm(files,total=len(files)):
            fs = file['audio']['sampling_rate']
            s  = file['audio']['array']
            segment_features = get_features_iemocap(fs,s)
            l = file['valence']
            if check_label(l,folder_path):
                for segment_feature in segment_features:
                    features.append(segment_feature)
                    labels.append(l)
    else:
        files = glob.glob(folder_path+"/*.wav")
        for file in tqdm(files,total=len(files)):
            segment_features = get_features(file)
            l = get_label(file,folder_path)
            if check_label(l,folder_path):
                for segment_feature in segment_features:
                    features.append(segment_feature)
                    labels.append(l)
    return np.array(features),np.array(get_label_numbers(labels,folder_path))

# train the svm classifier with cross validation
def train_svm_classifier_for_valence(folder_path):
    print(f"Training on {folder_path} ...")
    if folder_path != "all":
        f,y = read_folder(folder_path)
    else:
        f1,y1 = read_folder("emoDB")
        f2,y2 = read_folder("emovo")
        f3,y3 = read_folder("ravdess")
        f4,y4 = read_folder("savee")
        f5,y5 = read_folder("iemocap")
        f = np.concatenate((f1,f2,f3,f4,f5),axis=0)
        y = np.concatenate((y1,y2,y3,y4,y5),axis=0)
    
    Train_SVM(f,y,folder_path)
    Train_ANN(f,y,folder_path,3)
    
train_svm_classifier_for_valence("emoDB")
train_svm_classifier_for_valence("emovo")
train_svm_classifier_for_valence("ravdess")
train_svm_classifier_for_valence("savee")
# train_svm_classifier_for_valence("all")