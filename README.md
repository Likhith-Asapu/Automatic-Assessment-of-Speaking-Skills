# Automatic-Assessment-of-Speaking-Skills

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)  ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

This Repository contains the implementation of *Automatic Assessment of Speaking Skills Using Aural and Textual
Information*. The details of the models implemented in this repo can be found in this paper: [https://aclanthology.org/2021.icnlsp-1.19/](https://aclanthology.org/2021.icnlsp-1.19/) 

# Requirements
- Python3
- PyTorch
- scikit-learn
- numpy
- pyAudioAnalysis
- SpeechRecognition
- librosa
- fasttext
- tqdm

Run `pip install -r requirements.txt
` to install all the required packages.

# Folder Structure

```
Automatic-Assessment-of-Speaking-Skills
├───AudioAnalysis
│   ├───arousal.py
│   ├───emotion.py
│   ├───valence.py
│   ├───TrainANN.py
│   ├───TrainSVM.py
│   └───Saved models
├───HighLevelAnalysis
│   ├───highLevel.py
│   ├───TrainANN.py
│   ├───TrainSVM.py
│   └───Saved models
├───Test
│   ├───Test.py
│   ├───sample.wav
│   └───Models saved
├───TextAnalysis
│   ├───arousal_BiLSTM.py
│   ├───arousal_FastText.py
│   ├───emotion_BiLSTM.py
│   ├───emotion_FastText.py
│   ├───valence_BiLSTM.py
│   ├───valence_FastText.py
│   ├───TrainANN.py
│   ├───TrainSVM.py
│   └───Saved models
├───README.md
├───requirements.txt
└───Presentation.pdf
```

# Dataset

The Datasets used in this paper are:
- IEMOCAP
- Savee
- Emovo
- Emo-db
- Ravdess
- PuSQ

# Usage

To test the final model, run the following commands:

```zsh
cd Test
python test.py
```

Enter the name of the audio file which needs to be tested
```
Enter Audio File Name: sample.txt
```

The output will two boolean values based on perceived Enjoyment and Expressiveness.


