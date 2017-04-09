import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import scipy
import warnings

warnings.filterwarnings("ignore")
import json
from watson_developer_cloud import ToneAnalyzerV3

from os import system
api_data = pd.read_csv(
'toneanalysisDepression.csv',
                           sep= ',', header= None)                  #change csv path
#print api_data
attrList = list(api_data.columns.values)
api_data = pd.read_csv('toneanalysisDepression.csv',
                           sep= ',',skiprows=[0], header= None)
#print attrList
X=api_data.values[:,0:6]
Y=api_data.values[:,6]
#print X
#print Y

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=8, min_samples_leaf=6)
 
clf_entropy.fit(X_train, y_train)
 
dotfile = open("tree.dot", 'w')
tree.export_graphviz(clf_entropy, out_file = dotfile, feature_names = attrList)
dotfile.close()
    
system("dot -Tpng tree.dot -o tree.png")
system("dot -Tps tree.dot -o tree.ps")
y_pred=[]
i=0
for a in X_test:
    y_pred.append((list)(clf_entropy.predict(a)))
 
#X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.5, random_state = 100)
#print ('accuracy is :',(accuracy_score(y_test, y_pred, normalize=True)))    




tone_analyzer = ToneAnalyzerV3(
   username='fc63d7cf-48af-4b43-9284-949648280610',
   password='V0iycgGI00QJ',
   version='2016-05-19')
enteredtext=raw_input('Enter the text to be analysed \n')
response=(json.loads(json.dumps((tone_analyzer.tone(text=enteredtext)),indent=2)))
#data=response.json()
# we will take input from the gui and put in the text part, try implementing that
apival=[0 for i in range(6)]
 
anger=response["document_tone"]["tone_categories"][0]["tones"][0]["score"]
apival[0]=anger
disgust=response["document_tone"]["tone_categories"][0]["tones"][1]["score"]
apival[1]=disgust
fear=response["document_tone"]["tone_categories"][0]["tones"][2]["score"]
apival[2]=(fear)
joy=response["document_tone"]["tone_categories"][0]["tones"][3]["score"]
apival[3]=(joy)
sadness=response["document_tone"]["tone_categories"][0]["tones"][4]["score"]
apival[4]=(sadness)
emotional_range=response["document_tone"]["tone_categories"][2]["tones"][4]["score"]
apival[5]=(emotional_range)

confidence=((sadness+fear+joy)/3.0)  #choose these three as these are the splitting attributes
 
print ('Anger,Disgust,Fear,Joy,Sadness,Emotional Range')
print apival
pred=clf_entropy.predict([apival])
print ("Prediction is ",pred[0])
#print(DecisionTreeClassifier.score(d,y_test, sample_weight=None))
#if(pred=='Yes'):
   
print ("Confidence of emotional analysis ",confidence)

#predict_proba(X, check_input=True)


 
