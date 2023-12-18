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
    0 - ANGER   - W
    1 - HAPPY   - F
    2 - SAD     - T
    3 - Neutral - N
    4 - BORED   - L
    5 - DISGUST - E
    6 - ANXIETY - A
    '''
    labels_encoded = {'W':2, 'F':2, 'T':0, 'N':1, 'L':0, 'E':2, 'A':2}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes

def emovo_get_labels(labels):
    '''
    LABELS :-
    0 - ANGER   - rab
    1 - HAPPY   - gio
    2 - SAD     - tri
    3 - Neutral - neu
    4 - DISGUST - dis   
    5 - FEAR    - pau   
    6 - SURPRISE- sor   
    '''
    labels_encoded = {'rab':2, 'gio':2, 'tri':0, 'neu':1, 'dis':0, 'pau':2, 'sor':2}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes

def ravdess_get_labels(labels):
    '''
    LABELS :-
    0 - ANGER   - 05
    1 - HAPPY   - 03
    2 - SAD     - 04
    3 - Neutral - 01
    4 - DISGUST - 07    
    5 - FEAR    - 06    
    6 - SURPRISE- 08    
    7 - CALM    - 02    
    '''
    labels_encoded = {'05':2, '03':2, '04':0, '01':1, '07':2, '06':2, '08':2, '02' : 0}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes


def savee_get_labels(labels):
    '''
    LABELS :-
    0 - ANGER   - a
    1 - HAPPY   - h
    2 - SAD     - sa
    3 - Neutral - n
    4 - DISGUST - d    
    5 - FEAR    - f    
    6 - SURPRISE- su   
    '''
    labels_encoded = {'a':2, 'h':2, 'sa':0, 'n':1, 'd':2, 'f':2, 'su':2}
    classes = []
    for label in labels:
        classes.append(labels_encoded[label])
    return classes

def iemocap_get_labels(labels):
    
    classes = []
    for label in labels:
        if label < 2.53:
            classes.append(0)
        elif label < 3.6:
            classes.append(1)
        else:
            classes.append(2)
    return classes

def check_label(label,db_type):
    if db_type == 'emoDB':
        labels_encoded = {'W':2, 'F':2, 'T':0, 'N':1, 'L':0, 'E':2, 'A':2}
    elif db_type == 'emovo':
        labels_encoded = {'rab':2, 'gio':2, 'tri':0, 'neu':1, 'dis':0, 'pau':2, 'sor':2}
    elif db_type == 'ravdess':
        labels_encoded = {'05':2, '03':2, '04':0, '01':1, '07':2, '06':2, '08':2, '02' : 0}
    elif db_type == 'savee':
        labels_encoded = {'a':2, 'h':2, 'sa':0, 'n':1, 'd':2, 'f':2, 'su':2}
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
        s = s[:,0]
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
            l = file['activation']
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
def train_svm_classifier_for_arousal(folder_path):
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
    
train_svm_classifier_for_arousal("emoDB")
train_svm_classifier_for_arousal("emovo")
train_svm_classifier_for_arousal("ravdess")
train_svm_classifier_for_arousal("savee")
train_svm_classifier_for_arousal("iemocap")
# train_svm_classifier_for_arousal("all")