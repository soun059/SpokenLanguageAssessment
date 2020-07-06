# SpokenLanguageAssessment
A spoken language assessment tool by which you can use your speech to determine how better are you in your english speaking capabalities.
Here you can use pipenv and install all the dependencies. Once done, if you are in windows install ffmpeg and please add its bin to your environment variable or if in linux or mac you can use your package manager to install them.
Once done .. execute the program in two ways
1. Record your voice for a specific duration and please give difficulty rating as 0 due to unpredictability of the recording device.
2. Use prerecorded voice to analyse and provide the name of the file to be used

##################
A quick example 

Use Hobbies as the file name and provide the path of the project on the given parameters. Once done just wait and this output has been produced.

Terminal -----
Enter the option 1.Record 2.use recorded:2
Enter the path to the Auto-Speech_Rater directory: C:\Users\souja\Desktop\SpokenLanguageAssessment
Filename (no extension please):    Hobbies
Pick degree of difficulties between 0 to 100:    10
Enter the topic to speak:Tell me about your hobbies
===========================================
HOLD ON!! get ready, 5 seconds to go!
===========================================
Go ahead!


===========================================
90.074
Pronunciation_posteriori_probability_score_percentage= :90.03
a Male, mood of speech: speaking passionately, p-value/sample size= :0.00 5


====================================================================================================
HERE ARE THE RESULTS, your spoken language level (speaking skills).
a: just started, a1: beginner, a2: elementary, b1: intermediate, b2: upper intermediate, c: master
====================================================================================================
58% accuracy     ['a'] CART model
70% accuracy     ['b2'] LDA model
67% accuracy     ['b2'] LR model
64% accuracy     ['b1'] NB model
==============================================
Sentiment Analysis
Sentiment found out to be : [('my hobbies is playing games especially like video games and also playing other puzzle games along with study books', 'pos')]
=============================================
Emotion analysis
Guessed Channel Layout for Input Stream #0.0 : stereo
result: angry
=============================================
Grammatical Errors
No. of mistakes found in the speech is  1
Accuracy per sentence: 1.0
==============================================
Relevancy
Total Presence of topic: 8.200000000000001
===========================================
RECORDING PROCESS IS DONE, press any key to terminate the programe

Hope you will find this useful.
