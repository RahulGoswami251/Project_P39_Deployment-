#import the libraries
import pandas as pd
from pickle import dump
from pickle import load
import pickle

import pandas as pd
df = pd.read_csv("Model_data.csv")

## Applying CountVectorizer 
# Extracting features by CountVectorizer from reviews 
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
reviews_feature_cv = count_vect.fit_transform(df.JwMarriot_reviews).toarray()
reviews_feature_cv.shape

pickle.dump(count_vect,open('trnsform.pkl','wb'))

## Aplying SMOTE oversampling technique for removing class imbalance 
from imblearn.over_sampling import SMOTE
over_sample = SMOTE(random_state = 50, sampling_strategy = "all")

X_oversample, y_oversample = over_sample.fit_sample(reviews_feature_cv, df['Sentiments']) 

# Count of Sentiments or target classes 
import collections, numpy
collections.Counter(y_oversample)

# Split data into train & test 
def split_into_words(i):
    return (i.split(" "))

seed = 7

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_oversample,y_oversample,test_size=0.2)

## Neutral Network  
# MLPClassifier
from sklearn.neural_network import MLPClassifier
model5=   MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 2), random_state=50)
MLP_clf= model5.fit(x_train, y_train)

# save the model to disk
dump(MLP_clf, open('Neural_Network_P39.sav', 'wb'))

# load the model from disk
model = load(open('Neural_Network_P39.sav', 'rb'))

pickle.dump(model5,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
