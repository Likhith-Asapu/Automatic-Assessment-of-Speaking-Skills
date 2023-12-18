from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO 
import librosa
from huggingface_hub import hf_hub_download
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
import speech_recognition as sr
from collections import Counter
from joblib import dump, load
import fasttext
import fasttext.util
import re
from scipy.special import softmax
from pydub import AudioSegment,silence
import warnings
warnings.filterwarnings("ignore")

ft = fasttext.load_model('cc.en.100.bin')

def get_text(wav_file_path):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(wav_file_path) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        # Record the audio
        audio_data = recognizer.record(source)
        text = ""
        try:
            # Use the recognizer to convert speech to text
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand the audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
    return text

def get_low_level_feature(file_path):
    fs, s = aIO.read_audio_file(file_path)
    segment_length = len(s)/fs
    if len(s.shape) > 1:
        s = s[:,0]
    mt, st, mt_n = aF.mid_feature_extraction(s, fs, 3 * fs, 3 * fs, 
                                             0.05 * fs, 0.05 * fs)
    segment_features= mt.T
    segment_features_mean = np.mean(segment_features,axis=0)
    return segment_features_mean,segment_length

def get_text_feature(file_path,segment_length):
    text = get_text(file_path)
    words = text.split()
    word_rate = len(words) * 60 /segment_length
    unique_words = set(words)
    unique_word_rate = len(unique_words)/segment_length
    
    word_freq = Counter(words)
    for key,value in word_freq.items():
        word_freq[key] = value/len(words)
    
    bins = np.linspace(0, 0.1, 10 + 1)

    # Calculate histogram
    histogram, _ = np.histogram(list(word_freq.values()), bins=bins)

    text = re.sub(r"[,.;@#?!&$-]",'',text)
    X = [ft.get_sentence_vector(text)]
    X = np.array(X)
    
    clf = load('svm_text_emotion.joblib')     
    emotions = softmax(clf.decision_function(X),axis=1) * 100
    emotions = emotions[0]
    emotion_features = [emotions[0],emotions[1],emotions[3],emotions[2]]
    
    # valence
    clf = load('svm_text_valence.joblib') 
    valence = softmax(clf.decision_function(X),axis=1) * 100
    valence = valence[0]
    valence_features = [valence[0],valence[1],valence[2]]
    
    # arousal
    clf = load('svm_text_arousal.joblib') 
    arousal = softmax(clf.decision_function(X),axis=1) * 100
    arousal = arousal[0]
    arousal_features = [arousal[2],arousal[0],arousal[1]]
    hist = []
    for num in histogram:
        hist.append(num)
    
    features = [word_rate, unique_word_rate] + hist + valence_features + arousal_features + emotion_features

    return features,len(words)

def get_audio_features(file_path,segment_length,no_of_words):
    
    myaudio = AudioSegment.from_wav(file_path)
    dBFS=myaudio.dBFS
    
    # short segments
    silence2 = silence.detect_silence(myaudio, min_silence_len=500, silence_thresh=dBFS-16, seek_step = 250)
    silence2 = [((start/1000),(stop/1000)) for start,stop in silence2] #in sec
    arr = []
    total = 0
    if len(silence2) == 0:
        silence2.append((0,1))
    for s in silence2:
        total += s[1] - s[0]
        arr.append(s[1] - s[0])
    if len(silence2) == 0:
        silence2.append(0,1)
    silence_short =  total/len(silence2)
    avg_short_segments = len(silence2) * 60/segment_length
    std_short = np.std(arr)
    speech_ratio_short = (segment_length - total)/segment_length
    word_rate_short = no_of_words/(segment_length - total)
    
    # long segments
    silence2 = silence.detect_silence(myaudio, min_silence_len=1000, silence_thresh=dBFS-16, seek_step = 250)
    silence2 = [((start/1000),(stop/1000)) for start,stop in silence2] #in sec
    arr = []
    total = 0
    if len(silence2) == 0:
        silence2.append((0,1))
    for s in silence2:
        total += s[1] - s[0]
        arr.append(s[1] - s[0])
    if len(silence2) == 0:
        silence2.append((0,1))
    silence_long =  total/len(silence2)
    avg_long_segments = len(silence2) * 60/segment_length
    std_long = np.std(arr)
    speech_ratio_long = (segment_length - total)/segment_length
    word_rate_long = no_of_words/(segment_length - total)
    
    fs, s = aIO.read_audio_file(file_path)
    segment_length = len(s)/fs
    if len(s.shape) > 1:
        s = s[:,0]
    mt, st, mt_n = aF.mid_feature_extraction(s, fs, 3 * fs, 3 * fs, 
                                             0.05 * fs, 0.05 * fs)
    segment_features= mt.T
    X = np.array(segment_features)
    
    clf = load('all_emotion.joblib')     
    emotions = softmax(clf.decision_function(X),axis=1) * 100
    emotions = np.mean(emotions,axis=0)
    emotion_features = [emotions[0],emotions[1],emotions[3],emotions[2]]
    
    # valence
    clf = load('all_valence.joblib') 
    valence = softmax(clf.decision_function(X),axis=1) * 100
    valence = np.mean(valence,axis=0)
    valence_features = [valence[0],valence[1],valence[2]]
    
    # arousal
    clf = load('all_arousal.joblib') 
    arousal = softmax(clf.decision_function(X),axis=1) * 100
    arousal = np.mean(arousal,axis=0)
    arousal_features = [arousal[2],arousal[0],arousal[1]]
    
    feature = [silence_short,silence_long,avg_short_segments,avg_long_segments,
               std_short,std_long,speech_ratio_short,speech_ratio_long,word_rate_short,word_rate_long] + emotion_features + arousal_features + valence_features
    
    return feature
def main():
    
    print("Enter Audio File Name: ",end = "")
    file_path = input()
    low_level_feature,segment_length = get_low_level_feature(file_path)
    text_feature,no_of_words = get_text_feature(file_path,segment_length)
    audio_feature = get_audio_features(file_path,segment_length,no_of_words)
    
    gender = "male"
    if gender == "male":
        clf1 = load('Late fusion MetaAudio+Text and LowLevelAudio,Expressiveness,male-1.joblib')
        clf2 = load('Late fusion MetaAudio+Text and LowLevelAudio,Expressiveness,male-2.joblib') 
    else:
        clf1 = load('Late fusion MetaAudio+Text and LowLevelAudio,Expressiveness,female-1.joblib')
        clf2 = load('Late fusion MetaAudio+Text and LowLevelAudio,Expressiveness,female-2.joblib')
    
    X1 = np.array([audio_feature + text_feature])
    X2 = np.array([low_level_feature])
    
    y1 = clf1.predict_proba(X1)
    y2 = clf2.predict_proba(X2)
    y_pred = ((y1 * y2)[:,1] > (y1 * y2)[:,0]).astype(int)
    print(f"Expressiveness Absent score - {(y1 * y2)[:,0][0]}, Expressiveness present score - {(y1 * y2)[:,1][0]}")
    if y_pred == 1:
        print("Expressiveness present")
    else:
        print("Expressiveness absent")
        
    
    gender = "male"
    if gender == "male":
        clf1 = load('Late fusion MetaAudio+Text and LowLevelAudio,Enjoyment,male-1.joblib')
        clf2 = load('Late fusion MetaAudio+Text and LowLevelAudio,Enjoyment,male-2.joblib') 
    else:
        clf1 = load('Late fusion MetaAudio+Text and LowLevelAudio,Enjoyment,female-1.joblib')
        clf2 = load('Late fusion MetaAudio+Text and LowLevelAudio,Enjoyment,female-2.joblib')
    
    X1 = np.array([audio_feature + text_feature])
    X2 = np.array([low_level_feature])
    
    y1 = clf1.predict_proba(X1)
    y2 = clf2.predict_proba(X2)
    y_pred = ((y1 * y2)[:,1] > (y1 * y2)[:,0]).astype(int)
    print(f"Enjoyment Absent score - {(y1 * y2)[:,0][0]}, Enjoyment present score - {(y1 * y2)[:,1][0]}")
    if y_pred == 1:
        print("Enjoyment present")
    else:
        print("Enjoyment absent")

main()