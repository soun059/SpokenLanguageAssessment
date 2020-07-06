import warnings
import csv
import _thread
from pandas import read_csv
from scipy.stats import ttest_ind
from scipy.stats import ks_2samp
from scipy.stats import binom
import pickle
import soundfile as sf
import sounddevice as sd
import queue
from sklearn import preprocessing
from subprocess import check_output
import os
import time
import pandas as pd
import numpy as np
import errno
import glob
from parselmouth.praat import call, run_file
import parselmouth
import sys
import sentiment_analysis as sa
import librosa
import soundfile
import os
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import nltk
nltk.download('wordnet')
nltk.download('stopwords')


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(
            X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result


def my_except_hook(exctype, value, traceback):
    print('There has been an error in the system')


#sys.excepthook = my_except_hook
if not sys.warnoptions:
    warnings.simplefilter("ignore")

op = int(input("Enter the option 1.Record 2.use recorded:"))
if(op != 1 and op != 2):
    exit()

pathy = input("Enter the path to the Auto-Speech_Rater directory: ")
name = input("Filename (no extension please):    ")
if(op == 1):
    t0 = int(input("Your desired Recording time in seconds:    "))
levvel = int(input("Pick degree of difficulties between 0 to 100:    "))
topic_to_speak = input("Enter the topic to speak:")
if topic_to_speak == "":
    print("please enter a relevant topic to speak")
    exit()

pa00 = pathy+"\\"+"dataset"+"\\"+"audioFiles"+"\\"
pa0 = pathy+"\\"+"dataset"+"\\"+"audioFiles"+"\\"+name+".wav"
pa1 = pathy+"\\"+"dataset"+"\\"+"datanewchi22.csv"
pa2 = pathy+"\\"+"dataset"+"\\"+"stats.csv"
pa3 = pathy+"\\"+"dataset"+"\\"+"datacorrP.csv"
pa4 = pathy+"\\"+"dataset"+"\\"+"datanewchi.csv"
pa5 = pathy+"\\"+"dataset"+"\\"+"datanewchi33.csv"
pa6 = pathy+"\\"+"dataset"+"\\"+"datanewchi33.csv"
pa7 = pathy+"\\"+"dataset"+"\\"+"datanewchi44.csv"
pa8 = pathy+"\\"+"dataset"+"\\"+"essen"+"\\"+"MLTRNL.praat"
pa9 = pathy+"\\"+"dataset"+"\\"+"essen"+"\\"+"myspsolution.praat"
pa10 = pathy+"\\"+"dataset"+"\\"+"audioFiles"+"\\done.wav"

rere = pa0

RECORD_TIME = 0
if(op == 1):
    RECORD_TIME = t0


def countdown(p, q, w):
    i = p
    j = q
    z = w
    k = 0
    while True:
        if(j == -1):
            j = 59
            i -= 1
        if(j > 9):
            print(str(k)+str(i) + " : " + str(j), "\t", end="\r")
        else:
            print(str(k)+str(i)+" : " + str(k)+str(j), "\t", end="\r")
        time.sleep(1)
        j -= 1
        if(i == 0 and j == -1):
            break
    if(i == 0 and j == -1):
        if z == 0:
            huf = "Go ahead!"
            print(huf)
        if z == 1:
            huf = "Time up!"
        # time.sleep(1)


print("===========================================")
print("HOLD ON!! get ready, 5 seconds to go!")
print("===========================================")
countdown(0, 5, 0)  # countdown(min,sec)


q = queue.Queue()
rec_start = int(time.time())

dev_info = sd.query_devices(2, 'input')

samplerate = 16000

if(op == 1):
    _thread.start_new_thread(countdown, (0, t0, 1))
    myrecording = sd.rec(
        t0 * samplerate, samplerate=samplerate, channels=2, blocking=True)
    sf.write(pa0, myrecording, samplerate)


result_array = np.empty((0, 100))
path = pa0
files = glob.glob(path)
result_array = np.empty((0, 27))

try:
    def mysppron(m, p, q):
        sound = m
        sourcerun = p
        path = q
        objects = run_file(sourcerun, -20, 2, 0.3, "yes",
                           sound, path, 80, 400, 0.01, capture_output=True)

        z1 = str(objects[1])
        z2 = z1.strip().split()
        z3 = int(z2[13])
        z4 = float(z2[14])
        db = binom.rvs(n=10, p=z4, size=10000)
        a = np.array(db)
        b = np.mean(a)*100/10
        print("Pronunciation_posteriori_probability_score_percentage= :%.2f" % (b))
        return

    def myspp(m, p, q):
        sound = m
        sourcerun = p
        path = q
        objects = run_file(sourcerun, -20, 2, 0.3, "yes",
                           sound, path, 80, 400, 0.01, capture_output=True)

        z1 = str(objects[1])
        z2 = z1.strip().split()
        z3 = int(z2[13])
        z4 = float(z2[14])
        db = binom.rvs(n=10, p=z4, size=10000)
        a = np.array(db)
        b = np.mean(a)*100/10
        return b

    def myspgend(m, p, q):
        sound = m
        sourcerun = p
        path = q
        objects = run_file(sourcerun, -20, 2, 0.3, "yes",
                           sound, path, 80, 400, 0.01, capture_output=True)

        z1 = str(objects[1])
        z2 = z1.strip().split()
        z3 = float(z2[8])
        z4 = float(z2[7])

        if z4 <= 114:
            g = 101
            j = 3.4
        elif z4 > 114 and z4 <= 135:
            g = 128
            j = 4.35
        elif z4 > 135 and z4 <= 163:
            g = 142
            j = 4.85
        elif z4 > 163 and z4 <= 197:
            g = 182
            j = 2.7
        elif z4 > 197 and z4 <= 226:
            g = 213
            j = 4.5
        elif z4 > 226:
            g = 239
            j = 5.3
        else:
            print("Voice not recognized")
            exit()

        def teset(a, b, c, d):
            d1 = np.random.wald(a, 1, 1000)
            d2 = np.random.wald(b, 1, 1000)
            d3 = ks_2samp(d1, d2)
            c1 = np.random.normal(a, c, 1000)
            c2 = np.random.normal(b, d, 1000)
            c3 = ttest_ind(c1, c2)
            y = ([d3[0], d3[1], abs(c3[0]), c3[1]])
            return y
        nn = 0
        mm = teset(g, j, z4, z3)
        while (mm[3] > 0.05 and mm[0] > 0.04 or nn < 5):
            mm = teset(g, j, z4, z3)
            nn = nn+1
        nnn = nn
        if mm[3] <= 0.09:
            mmm = mm[3]
        else:
            mmm = 0.35
        if z4 > 97 and z4 <= 114:
            print(
                "a Male, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % (mmm), (nnn))
        elif z4 > 114 and z4 <= 135:
            print(
                "a Male, mood of speech: Reading, p-value/sample size= :%.2f" % (mmm), (nnn))
        elif z4 > 135 and z4 <= 163:
            print(
                "a Male, mood of speech: speaking passionately, p-value/sample size= :%.2f" % (mmm), (nnn))
        elif z4 > 163 and z4 <= 197:
            print("a female, mood of speech: Showing no emotion, normal, p-value/sample size= :%.2f" % (mmm), (nnn))
        elif z4 > 197 and z4 <= 226:
            print(
                "a female, mood of speech: Reading, p-value/sample size= :%.2f" % (mmm), (nnn))
        elif z4 > 226 and z4 <= 245:
            print(
                "a female, mood of speech: speaking passionately, p-value/sample size= :%.2f" % (mmm), (nnn))
        else:
            print("Voice not recognized")

    for soundi in files:
        objects = run_file(pa8, -20, 2, 0.3, "yes", soundi,
                           pa00, 80, 400, 0.01, capture_output=True)

        z1 = (objects[1])
        z3 = z1.strip().split()
        z2 = np.array([z3])
        result_array = np.append(result_array, [z3], axis=0)

    np.savetxt(pa1, result_array, fmt='%s', delimiter=',')

    df = pd.read_csv(pa1,
                     names=['avepauseduratin', 'avelongpause', 'speakingtot', 'avenumberofwords', 'articulationrate', 'inpro', 'f1norm', 'mr', 'q25',
                            'q50', 'q75', 'std', 'fmax', 'fmin', 'vowelinx1', 'vowelinx2', 'formantmean', 'formantstd', 'nuofwrds', 'npause', 'ins',
                            'fillerratio', 'xx', 'xxx', 'totsco', 'xxban', 'speakingrate'], na_values='?')

    scoreMLdataset = df.drop(['xxx', 'xxban'], axis=1)
    scoreMLdataset.to_csv(pa7, header=False, index=False)
    newMLdataset = df.drop(['avenumberofwords', 'f1norm', 'inpro', 'q25', 'q75', 'vowelinx1',
                            'nuofwrds', 'npause', 'xx', 'totsco', 'xxban', 'speakingrate', 'fillerratio'], axis=1)
    newMLdataset.to_csv(pa5, header=False, index=False)
    namess = nms = ['avepauseduratin', 'avelongpause', 'speakingtot', 'articulationrate', 'mr',
                    'q50', 'std', 'fmax', 'fmin', 'vowelinx2', 'formantmean', 'formantstd', 'ins',
                    'xxx']
    df1 = pd.read_csv(pa5,
                      names=namess)
    df33 = df1.drop(['xxx'], axis=1)
    array = df33.values
    array = np.log(array)
    x = array[:, 0:13]

    print(" ")
    print(" ")
    print("===========================================")
    p = pa0
    c = pa9
    a = pa00
    bi = myspp(p, c, a)
    print(bi)
    if bi < levvel:
        mysppron(p, c, a)
        input("Try again, unnatural-sounding speech detected. No further result. Press any key to exit.")
        exit()

    mysppron(p, c, a)
    myspgend(p, c, a)

    print(" ")
    print(" ")
    print("====================================================================================================")
    print("HERE ARE THE RESULTS, your spoken language level (speaking skills).")
    print("a: just started, a1: beginner, a2: elementary, b1: intermediate, b2: upper intermediate, c: master")
    print("====================================================================================================")

    filename = pathy+"/"+"dataset"+"/"+"essen"+"/"+"CART_model.sav"
    model = pickle.load(open(filename, 'rb'))
    predictions = model.predict(x)
    print("58% accuracy    ", predictions, "CART model")

    filename = pathy+"/"+"dataset"+"/"+"essen"+"/"+"LDA_model.sav"

    model = pickle.load(open(filename, 'rb'))
    predictions = model.predict(x)
    print("70% accuracy    ", predictions, "LDA model")

    filename = pathy+"/"+"dataset"+"/"+"essen"+"/"+"LR_model.sav"

    model = pickle.load(open(filename, 'rb'))
    predictions = model.predict(x)
    print("67% accuracy    ", predictions, "LR model")

    filename = pathy+"/"+"dataset"+"/"+"essen"+"/"+"NB_model.sav"

    model = pickle.load(open(filename, 'rb'))
    predictions = model.predict(x)
    print("64% accuracy    ", predictions, "NB model")

    print("==============================================")
    print("Sentiment Analysis")
    res = sa.predict_audio(pa0)
    print("Sentiment found out to be :", res)
    print("=============================================")
    print("Emotion analysis")
    os.system(
        f"ffmpeg -i {pa0} -ac 1 -ar 16000 {pa10} -y -hide_banner -loglevel warning")
    rec = extract_feature(pa10, mfcc=True, chroma=True,
                          mel=True).reshape(1, -1)
    model = pickle.load(open("mlp_classifier.model", "rb"))
    result = model.predict(rec)[0]
    print("result:", result)
    print("=============================================")
    print("Grammatical Errors")
    import language_check
    tool = language_check.LanguageTool('en-US')
    i = 0
    fin = nltk.sent_tokenize(res[0][0])
    for line in fin:
        matches = tool.check(line)
        i = i + len(matches)

    print("No. of mistakes found in the speech is ", i)
    print("Accuracy per sentence:", i / len(fin))
    print("==============================================")
    print("Relevancy")
    from nltk.tokenize import word_tokenize
    from collections import Counter

    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    import gensim
    import string
    from gensim import corpora
    from gensim.corpora.dictionary import Dictionary
    from nltk.tokenize import word_tokenize
    compileddoc = [x for x in nltk.sent_tokenize(res[0][0])]
    stopwords = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(document):
        stopwordremoval = " ".join(
            [i for i in document.lower().split() if i not in stopwords])
        punctuationremoval = ''.join(
            ch for ch in stopwordremoval if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word)
                              for word in punctuationremoval.split())
        return normalized

    topic = clean(topic_to_speak).split()
    total_presence = 0.0
    final_doc = [clean(document).split() for document in compileddoc]
    dictionary = corpora.Dictionary(final_doc)

    DT_matrix = [dictionary.doc2bow(doc) for doc in final_doc]

    Lda_object = gensim.models.ldamodel.LdaModel
    lda_model_1 = Lda_object(DT_matrix, num_topics=2, id2word=dictionary)

    for i in lda_model_1.print_topics(num_topics=len(topic), num_words=5):
        for j in i[1].split("+"):
            p = j.split("*")
            st = p[1].replace('\"', "")
            st = st.replace(" ", "")
            #print(st, topic)
            if st in topic:
                total_presence += float(p[0])
    print("Total Presence of topic:", total_presence * 100)


except Exception as e:
    print(e)
    print(" ")
    print(" ")
    print("===========================================")
    print("Try again, noisy background or unnatural-sounding speech detected. No result.")
print("===========================================")
input("RECORDING PROCESS IS DONE, press any key to terminate the programe")
