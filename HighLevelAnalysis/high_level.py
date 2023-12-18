from pydub import AudioSegment,silence
import numpy as np
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
from collections import Counter
import pandas as pd
from scipy import stats
from TrainSVM import *
from TrainANN import *

# myaudio = AudioSegment.from_mp3("../emovo/dis-f1-l3.wav")
# dBFS=myaudio.dBFS
# silence = silence.detect_silence(myaudio, min_silence_len=500, silence_thresh=dBFS-16, seek_step=250)

# silence = [((start/1000),(stop/1000)) for start,stop in silence] #in sec
# print(silence)


# b = np.load('PuSQ/features_data/1_speaker1_female_Text.npz',allow_pickle=True)
# print(b)
# print(b['feature_names'])
# print(b['features'])

LowLevelAudio = Counter()
Text = Counter()
MetaAudio = Counter()
Expressiveness = Counter()
Enjoyment = Counter()
Annotators = Counter()
Expressiveness_Deviation = Counter()
Enjoyment_Deviation = Counter()
# read data from folder
def read_folder(type):
    if type == "LowLevelAudio" or type == "MetaAudio" or type == "Text":
        files = glob.glob("PuSQ/features_data/*_"+type+".npz")
    
    for file in tqdm(files,total=len(files)):
        b = np.load(file,allow_pickle=True)
        
        if type == "LowLevelAudio":
            name = file[19:-18]
            LowLevelAudio[name] = b['features']
        elif type == "MetaAudio":
            name = file[19:-14]
            MetaAudio[name] = b['features']
        elif type == "Text":
            name = file[19:-9]
            Text[name] = b['features']
        else:
            print("Wrong type")
            return
    # return np.array(features),np.array(get_label_numbers(labels,folder_path))


# extract average Expressiveness and Enjoyment scores for each speaker across all annotators
def extract_labels():
    df = pd.read_csv('PuSQ/annotations_database.csv')
    group = 1
    for index, row in df.iterrows():
        Annotators[row['Sample_name']] += 1
        if Annotators[row['Sample_name']] == 1:
            Expressiveness[row['Sample_name']] = []
            group += 1
        Expressiveness[row['Sample_name']].append(row['Class1'])
        if Annotators[row['Sample_name']] == 1:
            Enjoyment[row['Sample_name']] = []
        Enjoyment[row['Sample_name']].append(row['Class2'])
        
    
    for key in Annotators:
        Expressiveness_Deviation[key] = stats.median_abs_deviation(Expressiveness[key])
        Expressiveness[key] = np.mean(Expressiveness[key])
        Enjoyment_Deviation[key] = stats.median_abs_deviation(Enjoyment[key])
        Enjoyment[key] = np.mean(Enjoyment[key])
        # print(Expressiveness_Deviation[key],Expressiveness[key])
        # if Expressiveness_Deviation[key] <= 0.75 and (Expressiveness[key] >= 4 or Expressiveness[key] <= 2) and key.find('_female') != -1 and Annotators[key] > 2:
        #     X1.append()
        # print(key,Expressiveness[key],Enjoyment[key])
def find_group(key):
    return re.search("speaker[0-9]+", key).group()

def train(feature,task,gender):
    gender = gender.lower()
    X = []
    y = []
    groups = []
    count1 = 0
    count2 = 0
    feature_keys = Counter()
    if feature == "LowLevelAudio":
        feature_keys = LowLevelAudio
    elif feature == "Text":
        for key,value in Text.items():
            if len(Text[key]) > 0:
                feature_keys[key] = Text[key]
    elif feature == "MetaAudio":
        feature_keys = MetaAudio
    
    if task == "Expressiveness":
        task_features = Expressiveness
        task_deviation = Expressiveness_Deviation
    elif task == "Enjoyment":
        task_features = Enjoyment
        task_deviation = Enjoyment_Deviation
    
    for key,value in feature_keys.items(): 
        if task_deviation[key] <= 0.75 and key.find("_"+gender) != -1 and Annotators[key] > 2:
            if gender == "female":
                if task_features[key] <= 2: 
                    y.append(0)
                    X.append(feature_keys[key])
                    groups.append(find_group(key))
                    count1 += 1 
                if task_features[key] >= 4:
                    y.append(1)
                    X.append(feature_keys[key]) 
                    groups.append(find_group(key))
                    count2 += 1
            else:
                if task_features[key] <= 2: 
                    y.append(0)
                    X.append(feature_keys[key])
                    groups.append(find_group(key))
                    count1 += 1 
                if task_features[key] >= 3.1:
                    y.append(1)
                    X.append(feature_keys[key]) 
                    groups.append(find_group(key))
                    count2 += 1  
            
    print(count1,count2,count1+count2)
    Train_SVM(X,y,groups,f"{feature},{task},{gender}")
    Train_ANN(X,y,groups,f"{feature},{task},{gender}")
    print("=============================================")

def train_early_fusion(feature1,feature2,task,gender):
    gender = gender.lower()
    X = []
    y = []
    groups = []
    count1 = 0
    count2 = 0
    feature_keys1 = Counter()
    feature_keys2 = Counter()
    if feature1 == "LowLevelAudio":
        feature_keys1 = LowLevelAudio
    elif feature1 == "Text":
        for key,value in Text.items():
            if len(Text[key]) > 0:
                feature_keys1[key] = Text[key]
            else:
                feature_keys1[key] = np.zeros(22)
    elif feature1 == "MetaAudio":
        feature_keys1 = MetaAudio
    elif feature1 == "MetaAudio+Text":
        for key,value in MetaAudio.items():
            if len(MetaAudio[key]) > 0 and len(Text[key]) > 0:
                feature_keys1[key] = np.concatenate((MetaAudio[key],Text[key]),axis=0)
            else:
                feature_keys1[key] = np.concatenate((MetaAudio[key],np.zeros(22)),axis=0)
    
    if feature2 == "LowLevelAudio":
        feature_keys2 = LowLevelAudio
    elif feature2 == "Text":
        for key,value in Text.items():
            if len(Text[key]) > 0:
                feature_keys2[key] = Text[key]
            else:
                feature_keys2[key] = np.zeros(22)
    elif feature2 == "MetaAudio":
        feature_keys2 = MetaAudio
    
    if task == "Expressiveness":
        task_features = Expressiveness
        task_deviation = Expressiveness_Deviation
    elif task == "Enjoyment":
        task_features = Enjoyment
        task_deviation = Enjoyment_Deviation
    
    for key,value in feature_keys1.items(): 
        if task_deviation[key] <= 0.75 and key.find("_"+gender) != -1 and Annotators[key] > 2:
            if gender == "female":
                if task_features[key] <= 2: 
                    y.append(0)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0))
                    groups.append(find_group(key))
                    count1 += 1 
                if task_features[key] >= 4:
                    y.append(1)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0))
                    groups.append(find_group(key))
                    count2 += 1
            else:
                if task_features[key] <= 2: 
                    y.append(0)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0))
                    groups.append(find_group(key))
                    count1 += 1 
                if task_features[key] >= 3.1:
                    y.append(1)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0)) 
                    groups.append(find_group(key))
                    count2 += 1  
            
    print("count1 - {}, count2 - {}, count3 - {}".format(count1,count2,count1+count2))
    Train_SVM(X,y,groups,f"Early fusion {feature1} and {feature2},{task},{gender}")
    Train_ANN(X,y,groups,f"Early fusion {feature1} and {feature2},{task},{gender}")
    print("=============================================")

def train_late_fusion(feature1,feature2,task,gender):
    gender = gender.lower()
    X = []
    y = []
    groups = []
    count1 = 0
    count2 = 0
    feature_keys1 = Counter()
    if feature1 == "LowLevelAudio":
        feature_keys1 = LowLevelAudio
    elif feature1 == "Text":
        for key,value in Text.items():
            if len(Text[key]) > 0:
                feature_keys1[key] = Text[key]
    elif feature1 == "MetaAudio":
        feature_keys1 = MetaAudio
    elif feature1 == "MetaAudio+Text":
        for key,value in MetaAudio.items():
            if len(MetaAudio[key]) > 0 and len(Text[key]) > 0:
                feature_keys1[key] = np.concatenate((MetaAudio[key],Text[key]),axis=0)
    
    if feature2 == "LowLevelAudio":
        feature_keys2 = LowLevelAudio
    elif feature2 == "Text":
        feature_keys2 = Text
    elif feature2 == "MetaAudio":
        feature_keys2 = MetaAudio
    
    if task == "Expressiveness":
        task_features = Expressiveness
        task_deviation = Expressiveness_Deviation
    elif task == "Enjoyment":
        task_features = Enjoyment
        task_deviation = Enjoyment_Deviation
    print(len(feature_keys1))
    size = 0
    for key,value in feature_keys1.items(): 
        if task_deviation[key] <= 0.75 and key.find("_"+gender) != -1 and Annotators[key] > 2:
            size = len(feature_keys1[key])
            if gender == "female":
                if task_features[key] <= 2: 
                    y.append(0)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0))
                    groups.append(find_group(key))
                    count1 += 1 
                if task_features[key] >= 4:
                    y.append(1)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0))
                    groups.append(find_group(key))
                    count2 += 1
            else:
                if task_features[key] <= 2: 
                    y.append(0)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0))
                    groups.append(find_group(key))
                    count1 += 1 
                if task_features[key] >= 3.1:
                    y.append(1)
                    X.append(np.concatenate((feature_keys1[key],feature_keys2[key]),axis=0)) 
                    groups.append(find_group(key))
                    count2 += 1  
            
    print("negative - {}, positive - {}, total - {}".format(count1,count2,count1+count2))
    Train_2_SVM(X,y,groups,size,f"Late fusion {feature1} and {feature2},{task},{gender}")
    Train_2_ANN(X,y,groups,size,f"Late fusion {feature1} and {feature2},{task},{gender}")
    print("=============================================")


read_folder("LowLevelAudio")
read_folder("MetaAudio")
read_folder("Text")
extract_labels()
# ###############################
# train("MetaAudio","Expressiveness","female")
# train("Text","Expressiveness","female")
# train("LowLevelAudio","Expressiveness","female")
# print("--------------------------------------------------------\n\n")
# train("MetaAudio","Expressiveness","male")
# train("Text","Expressiveness","male")
# train("LowLevelAudio","Expressiveness","male")
# print("--------------------------------------------------------\n\n")
# # ################################
# train("MetaAudio","Enjoyment","female")
# train("Text","Enjoyment","female")
# train("LowLevelAudio","Enjoyment","female")
# print("--------------------------------------------------------\n\n")

# train("MetaAudio","Enjoyment","male")
# train("Text","Enjoyment","male")
# train("LowLevelAudio","Enjoyment","male")
# print("--------------------------------------------------------\n\n")
# # # ################################
# train_early_fusion("MetaAudio","Text","Expressiveness","female")
# train_late_fusion("MetaAudio","LowLevelAudio","Expressiveness","female")
# train_early_fusion("MetaAudio","LowLevelAudio","Expressiveness","female")
train_late_fusion("MetaAudio+Text","LowLevelAudio","Expressiveness","female")
# train_early_fusion("MetaAudio+Text","LowLevelAudio","Expressiveness","female")
# print("--------------------------------------------------------\n\n")

# train_early_fusion("MetaAudio","Text","Expressiveness","male")
# train_late_fusion("MetaAudio","LowLevelAudio","Expressiveness","male")
# train_early_fusion("MetaAudio","LowLevelAudio","Expressiveness","male")
train_late_fusion("MetaAudio+Text","LowLevelAudio","Expressiveness","male")
# train_early_fusion("MetaAudio+Text","LowLevelAudio","Expressiveness","male")
# print("--------------------------------------------------------\n\n")

# train_early_fusion("MetaAudio","Text","Enjoyment","female")
# train_late_fusion("MetaAudio","LowLevelAudio","Enjoyment","female")
# train_early_fusion("MetaAudio","LowLevelAudio","Enjoyment","female")
# train_late_fusion("MetaAudio+Text","LowLevelAudio","Enjoyment","female")
# train_early_fusion("MetaAudio+Text","LowLevelAudio","Enjoyment","female")
# print("--------------------------------------------------------\n\n")

# train_early_fusion("MetaAudio","Text","Enjoyment","male")
# train_late_fusion("MetaAudio","LowLevelAudio","Enjoyment","male")
# train_early_fusion("MetaAudio","LowLevelAudio","Enjoyment","male")
# train_late_fusion("MetaAudio+Text","LowLevelAudio","Enjoyment","male")
# train_early_fusion("MetaAudio+Text","LowLevelAudio","Enjoyment","male")
# print("--------------------------------------------------------\n\n")
# # ##################################################################