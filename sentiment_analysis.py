from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pydub import AudioSegment

import os
import nltk
import speech_recognition as sr
import sys
import pickle

# nltk.download('punkt')
#moviedir = r'movie_reviews'
#movie_train = load_files(moviedir,shuffle=True)


# def train():
#global movie_train
#movie_train = load_files(moviedir,shuffle=True)
#movie_vec = CountVectorizer(min_df=2,tokenizer=nltk.word_tokenize)
#movie_counts = movie_vec.fit_transform(movie_train.data)
#tfidf_transformer = TfidfTransformer()

#movie_tfidf = tfidf_transformer.fit_transform(movie_counts)

#docs_train, docs_test, y_train, y_test = train_test_split(movie_tfidf, movie_train.target, test_size = 0.20, random_state=12)
#clf = MultinomialNB().fit(docs_train,y_train)
#y_pred = clf.predict(docs_test)

# return clf,movie_vec,tfidf_transformer

pkl_filename = "sentiment_analysis.pkl"
vec_filename = "vector.pkl"
trans_filename = "transformer.pkl"
trained = True
# if trained == False:
#    print("Training module...")
#    pred,vec,transformer = train()
#    with open(pkl_filename,'wb') as model:
#        pickle.dump(pred,model)
#    with open(vec_filename,'wb') as vec_file:
#        pickle.dump(vec,vec_file)
#    with open(trans_filename,'wb') as tran_file:
#        pickle.dump(transformer,tran_file)
# else:
with open(pkl_filename, 'rb') as model:
    pred = pickle.load(model)
with open(vec_filename, 'rb') as vec_file:
    movie_vec = pickle.load(vec_file)
with open(trans_filename, 'rb') as tran_file:
    tfidf_transformer = pickle.load(tran_file)

recog = sr.Recognizer()


def predict_audio(audio):
    file_list = list()
    audio_file = "NAN"
    audio_file = audio
    if audio_file.endswith(".wav") and audio_file != "nil":
        sample_audio = sr.AudioFile(audio_file)
        with sample_audio as source:
            audio = recog.record(source)
        review = recog.recognize_google(audio)
        file_list.append(review)

    elif audio_file != "nil":
        file_split = os.path.splitext(audio_file)
        file_extension = file_split[1]
        file_format = file_extension[1:]
        raw_file = AudioSegment.from_file(audio_file, format=file_format)
        raw_file.export(file_split[0]+".wav", format="wav")
        wav_file = file_split[0] + ".wav"

        sample_audio = sr.AudioFile(wav_file)
        with sample_audio as source:
            audio = recog.record(source)
        review = recog.recognize_google(audio)
        file_list.append(review)

    file_list_count = movie_vec.transform(file_list)
    global file_list_tfidf
    file_list_tfdif = tfidf_transformer.transform(file_list_count)
    predict = pred.predict(file_list_tfdif)
    sentiment = []
    for review, category in zip(file_list, predict):
        sa = (review, ["pos", "neg"][category])
        sentiment.append(sa)
    return sentiment
