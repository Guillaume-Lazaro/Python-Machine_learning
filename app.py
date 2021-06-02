import warnings
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

path='./dataset/'
model_filename = "hatespeech.joblib.z"

st.title("Appli de détection d'injure Twitter")
st.header("Etes-vous un troll ?")

# df = pd.read_csv(path+"labeled_data.csv")
# lb = preprocessing.LabelEncoder()
# y = df['class']
# X = df.drop('class', axis=1)

# X_train, X_test, y_train, y_test = train_test_split(X,y)


clf = joblib.load(model_filename)
text = "I hate you, please die!"
st.write(clf.predict_proba([text])[0])

user_text = st.text_input("Phrase à tester")
score = clf.predict_proba([user_text])[0]
st.write("Probabilité de merdes")
st.write(score)