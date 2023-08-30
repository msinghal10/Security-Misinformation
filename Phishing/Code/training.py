import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import *
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from ast import literal_eval
from collections import Counter 


Corpus = pd.read_csv(r"Train-no.csv", encoding='utf-8')


# Step - 2: Split the model into Train and Test Data set

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['ID'], Corpus['Label'],
                                                                    test_size=0.3)


Encoder = LabelEncoder()

Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

temp=[]
temp1=[]
Train_X=Train_X.reset_index(drop=True)
Test_X=Test_X.reset_index(drop=True)
Corpus=Corpus.reset_index(drop=True)

for x in range(len(Train_X)):
    tmp = []
    count1=0
    found = Corpus.loc[Corpus['ID']==Train_X[x]]
    tmp.append(found.iloc[0]['Tweets'])
    tmp.append(found.iloc[0]['Friends'])
    tmp.append(found.iloc[0]['Followers'])
    tmp.append(found.iloc[0]['Verified'])
    tmp.append(found.iloc[0]['Listed'])
    tmp.append(found.iloc[0]['Description'])
    tmp.append(found.iloc[0]['Age'])
    tmp.append(found.iloc[0]['Image'])
    tmp.append(found.iloc[0]['HasURL-TW'])
    tmp.append(found.iloc[0]['HasMedia-Tw'])
    tmp.append(found.iloc[0]['User-URL'])
    tmp.append(found.iloc[0]['VT'])
    f=found.iloc[0]['Tweet-Text']
    for a in f:
        if a.isupper():
            count1+=1
    tmp.append(count1)
    f=len(Train_X[x])
    tmp.append(f)
    
    temp.append(tmp)


for x in range(len(Test_X)):
    tmp1=[]
    count1=0
    found = Corpus.loc[Corpus['ID']==Test_X[x]]

    tmp1.append(found.iloc[0]['Tweets'])
    tmp1.append(found.iloc[0]['Friends'])
    tmp1.append(found.iloc[0]['Followers'])
    tmp1.append(found.iloc[0]['Verified'])
    tmp1.append(found.iloc[0]['Listed'])
    tmp1.append(found.iloc[0]['Description'])
    tmp1.append(found.iloc[0]['Age'])
    tmp1.append(found.iloc[0]['Image'])
    tmp1.append(found.iloc[0]['HasURL-TW'])
    tmp1.append(found.iloc[0]['HasMedia-Tw'])
    tmp1.append(found.iloc[0]['User-URL'])
    tmp1.append(found.iloc[0]['VT'])
    f=found.iloc[0]['Tweet-Text']
    for a in f:
        if a.isupper():
            count1+=1
    tmp1.append(count1)
    f=len(Test_X[x])
    tmp1.append(f)
    temp1.append(tmp1)




# # Step - 5: Now we run oversample the minority class so that there are no imbalances
temp = np.array(temp)
temp1 = np.array(temp1)

temp = np.reshape(temp, (len(temp), temp.shape[1]))
temp1 = np.reshape(temp1, (len(temp1), temp1.shape[1]))



X_train=np.array(Train_X.values.astype('U'))
X_test=np.array(Test_X.values.astype('U'))
X_train=np.reshape(X_train,(-1, 1))
X_test=np.reshape(X_test,(-1, 1))

X_train=np.append(X_train,temp,axis=1)
X_test=np.append(X_test,temp1,axis=1)



print(X_train.shape)
print(X_test.shape)


# Step - 6: Now we can run different algorithms to classify out data check for accuracy


# pipe1 = Pipeline([('classifier2' ,RandomForestClassifier())])
# param_grid1 = [
#      {'classifier2' : [RandomForestClassifier()],
#      'classifier2__n_estimators' : range(100,1000,100),
#     'classifier2__random_state' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}]

regressor = GridSearchCV(pipe1, param_grid = param_grid1, cv = 3, verbose=1, n_jobs=-1)
regressor = RandomForestClassifier(n_estimators=100, random_state=1)
regressor.fit(X_train, Train_Y)
y_pred = regressor.predict(X_test)

print("Random Forrest Accuracy Score -> ", accuracy_score(y_pred.round(), Test_Y) * 100)
print("Random Forrest Report ->", classification_report(Test_Y,y_pred.round(), zero_division=1))

# print(regressor.best_params_)

#To get the feature importance uncomment from lines XXX to XXX

# a=dict(zip(regressor.feature_importances_,X_resample))
# # print(a)
# a_keys=list(a.keys())
# a_values=list(a.values())
# ll=[]
# for i, x in enumerate(a_values):
#     temp=[]
#     temp.append(abc[x])
#     temp.append(a_keys[i])
#     ll.append(temp)
# df5 = pd.DataFrame(ll)
# df5.to_csv(r'final-features-twitter-1.csv',index=False)



# filename = 'labelencoder_fitted-new.pkl'
# pickle.dump(Encoder, open(filename, 'wb'))

# filename = 'rf_trained_model-new.sav'
# pickle.dump(regressor, open(filename, 'wb'))

