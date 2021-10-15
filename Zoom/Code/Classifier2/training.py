import pandas as pd
import numpy as np
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


Corpus = pd.read_csv(r"facebook.csv", encoding='utf-8')

# Step - 1: Data Pre-processing
# C1= Corpus[Corpus['Label'] == '__label3__']
# C1.to_csv(r'FP.csv',index=False)
C2=Corpus.loc[(Corpus['Label'] == '__label1__') | (Corpus['Label'] == '__label2__')]
# C2.to_csv(r'TP.csv',index=False)
# input('enter enter when ready')
# C2.loc[C2['Label'] == '__label2__','Label'] = '__label1__'
# C2.to_csv(r'TP.csv',index=False)
# input('enter enter when ready')
# C1.loc[C1['Label'] == '__label3__','Label'] = '__label2__'
# C1.to_csv(r'FP-1.csv',index=False)
# Corpus = pd.concat([C1,C2])
# Corpus.to_csv(r't.csv',index=False)
# input('enter enter when ready')
C2['Message']=C2['Message'].dropna()
# print(Corpus)
# input('enter enter when ready')
def csleaner(text):
    text = re.sub(r"@[A-Za-z0-9]+","",str(text)) #Remove @ sign
    text=re.sub(r"[RT]+","",str(text))
    text=text.strip().replace(':', '')
    text=re.sub(r'\\n'," ",text)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", str(text)) #Remove http links
    text = " ".join(str(text).split())
    # tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    text = text.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
    #      if w.lower() in words or not w.isalpha())
    text=text.replace(r"/[^a-zA-Z ]+/g", "")
    return str(text)

C2['Message'] = C2['Message'].map(csleaner)


C2['Message'].map(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",str(x)).split())

#Following removes any traces of emojis
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
         u"\U0001F600-\U0001F64F"  # emoticons
         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
         u"\U0001F680-\U0001F6FF"  # transport & map symbols
         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
         u"\U00002500-\U00002BEF"  # chinese char
         u"\U00002702-\U000027B0"
         u"\U00002702-\U000027B0"
         u"\U000024C2-\U0001F251"
         u"\U0001f926-\U0001f937"
         u"\U00010000-\U0010ffff"
         u"\u2640-\u2642"
         u"\u2600-\u2B55"
         u"\u200d"
         u"\u23cf"
         u"\u23e9"
         u"\u231a"
         u"\ufe0f"  # dingbats
         u"\u3030"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',str(text))

C2['Message'] = C2['Message'].map(deEmojify)

C2['Message']=C2['Message'].dropna()
C2['Label']=C2['Label'].dropna()
# print(Corpus)
# def csleaner(text):
#     text = re.sub(r'@[A-Za-z0-9]+',"",text) #Remove @ sign
#     text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) #Remove http links
#     text = " ".join(text.split())
#     # tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
#     text = text.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
#     # tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
#     #      if w.lower() in words or not w.isalpha())
#     return text

# Corpus['text'] = Corpus['text'].map(csleaner)

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun

tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):
    # Change all the text to lower case
    text = str(text).lower()

    #Tokenization
    text_words_list = word_tokenize(text)

    #Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    Final_words = []
    word_Lemmatized = WordNetLemmatizer()
    for word, tag in pos_tag(text_words_list):
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    return str(Final_words)


C2['text_final'] = C2['Message'].map(text_preprocessing)

# Corpus.to_csv(r'finaldata.csv',index=False)
# input('enter enter when ready')
# opening a csv,
# conatinatying the csv with the corpus text final 
# conctenate or + ?



# Step - 2: Split the model into Train and Test Data set

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(C2['text_final'], C2['Label'],
                                                                    test_size=0.3)

# print(Train_X)

# Test_Y.to_csv(r'test2.csv',index=False)
# input('enter enter when ready')
# Train_Y = pd.read_csv(r"test1.csv")
# Test_Y = pd.read_csv(r"test2.csv")

# # Step - 3: Label encode the target variable
Encoder = LabelEncoder()
# # Train_Y = np_utils.to_categorical(lambda i : lb.fit_transform(Train_Y))
# # Encoder.fit(Train_Y)
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

temp=[]
temp1=[]
Train_X=Train_X.reset_index(drop=True)
Test_X=Test_X.reset_index(drop=True)
# C2=C2.reset_index(drop=True)
# Corpus.reset_index()
for x in range(len(Train_X)):
    tmp = []
    found = C2.loc[C2['text_final']==Train_X[x]]
    # input('enter enter when ready')
    # if len(found) >0:
    tmp.append(found.iloc[0]['Likes'])
    tmp.append(found.iloc[0]['Comments'])
    tmp.append(found.iloc[0]['Shares'])
    if (len(found.iloc[0]['Link'])>0):
        tmp.append(1)
    else:
        temp.append(0)
    temp.append(tmp)
    # temp.append(found.iloc[0]['Likes'])
    # print(temp)

for x in range(len(Test_X)):
    tmp=[]
    found = C2.loc[C2['text_final']==Test_X[x]]
    # print(found)
    # input('enter enter when ready')
    # if len(found) >0:
    tmp.append(found.iloc[0]['Likes'])
    tmp.append(found.iloc[0]['Comments'])
    tmp.append(found.iloc[0]['Shares'])
    if (len(found.iloc[0]['Link'])>0):
        tmp.append(1)
    else:
        temp.append(0)
    temp1.append(tmp)
    # print(temp)



# type(found['Likes'])
# np.savetxt("GFG.csv", temp,delimiter =",")
# input('enter enter when ready')
# Step - 4: Vectorize the words by using TF-IDF Vectorizer

# Tfidf_vect = TfidfVectorizer(analyzer='char',token_pattern=r'\w{1,}',max_features=500,ngram_range=(2, 2))
# Tfidf_vect.fit(Train_X.values.ravel())

# Train_X_Tfidf = Tfidf_vect.transform(Train_X.values.ravel())
# Test_X_Tfidf = Tfidf_vect.transform(Test_X.values.ravel())

# pipe11 = Pipeline([('tfidf' ,TfidfVectorizer())])
# param_grid = [
#     {'tfidf' : [TfidfVectorizer()],
#      'tfidf_analyzer' : ['char','word','char_wb'],
#     'tfidf__max_features' : range(100,20000,100),
#      'tfidf_ngram_range': ((1,1),(1,2),(2,2),(3,3))}  
# ]
# Tfidf_vect = GridSearchCV(pipe11, param_grid = param_grid, verbose=5, n_jobs=-1)
Tfidf_vect = TfidfVectorizer(analyzer='char',token_pattern=r'\w{1,}',max_features=100,ngram_range=(1,2))
# Corpus.to_csv(r'lol.csv',index=False)
# # input('enter enter when ready')
# Corpus = pd.read_csv(r"lol.csv")
# Corpus['text_final']=Corpus['text_final'].str[0:-1]+','+Corpus['F'].astype(str)+Corpus['V'].astype(str)+']'
# Corpus['text_final']=Corpus['text_final'].str[0:-1]+','+Corpus['F'].astype(str)+Corpus['S'].astype(str)+Corpus['F'].astype(str)+Corpus['V'].astype(str)+']'
# Corpus['text_final'] = Corpus['text_final'] + ' '  + Corpus['F'].astype(str) + ' '  + Corpus['L'].astype(str)+ ' '  + Corpus['C'].astype(str)+ ' '  + Corpus['V'].astype(str)
# Corpus['text_final']=Corpus['text_final'].str.insert(-2,','+Corpus['F'].astype(str))

# Corpus['text_final'] = Corpus['text_final'] + ' '  + Corpus['F'].astype(str)
# Corpus.to_csv(r'lol1.csv',index=False)
# input('enter enter when ready')
Tfidf_vect.fit(C2['text_final'].apply(lambda x: np.str_(Tfidf_vect)))
# print(Tfidf_vect.best_params_)
# input('enter enter when ready')
Train_X_Tfidf = Tfidf_vect.transform(Train_X.values.astype('U'))
Test_X_Tfidf = Tfidf_vect.transform(Test_X.values.astype('U'))

# Step - 5: Now we run oversample the minority class so that there are no imbalances
temp = np.array(temp)
temp1 = np.array(temp1)

temp = np.reshape(temp, (len(temp), 3))
temp1 = np.reshape(temp1, (len(temp1), 3))

# print(Train_X_Tfidf.shape)
# print(temp.shape)
# print(Test_X_Tfidf.shape)
# print(temp1.shape)

X_train=Train_X_Tfidf.toarray()
X_test=Test_X_Tfidf.toarray()
# print(X_train)
# print(temp)
# input('enter enter when ready')
X_train=np.append(X_train,temp,axis=1)
X_test=np.append(X_test,temp1,axis=1)

# print(X_train.shape)
# print(X_test.shape)
# input('tt')

x_train = pd.DataFrame(X_train)
X_resample, y_resampled = SMOTE().fit_resample(x_train, Train_Y)



# Step - 6: Now we can run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(X_resample, y_resampled)

Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf, Train_Y)
# predictions_NB = Naive.predict(Test_X_Tfidf)
#Naive.fit(X_resample, y_resampled)
#predictions_NB = Naive.predict(X_test)

# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)
#print("Naive Bayes Report ->", classification_report(Test_Y,predictions_NB,zero_division=1))

# pipe = Pipeline([('classifier1' ,svm.SVC())])
# param_grid = [
#     {'classifier1' : [svm.SVC()],
#      'classifier1__kernel' : ['linear','rbf'],
#     'classifier1__C' : np.logspace(-4, 4, 20),
#     'classifier1__degree' : [1,2,3,4,5],
#      'classifier1__gamma': ['scale', 'auto']}  
# ]

pipe1 = Pipeline([('classifier2' ,RandomForestClassifier())])
param_grid1 = [
     {'classifier2' : [RandomForestClassifier()],
     'classifier2__n_estimators' : range(100,1000,100),
    'classifier2__random_state' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}]


pipe2 = Pipeline([('classifier3' ,KNeighborsClassifier())])
param_grid2 = [
     {'classifier3' : [KNeighborsClassifier()],
     'classifier3__n_neighbors' : range(3, 10)}]
# pipe3 = Pipeline([('classifier4' ,MLPClassifier())])
# param_grid3 = [
#      {'classifier4' : [MLPClassifier()],
#      'classifier4__activation' : ['logistic', 'tanh', 'relu'],
#     'classifier4__solver' : ['adam'],
#     'classifier4__alpha' : [1e-5,1e-3,1e-4,1e-2,1e-1,1e-6,1e-7],
#      'classifier4__hidden_layer_sizes': (5,2),
#      'classifier4__random_state' : [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}]

# SVM = GridSearchCV(pipe, param_grid = param_grid, cv = 3, verbose=1, n_jobs=-1)


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
# SVM = svm.SVC(C=1.0, kernel='linear', degree=2, gamma='auto')

# SVM.fit(Train_X_Tfidf, Train_Y)
# predictions_SVM = SVM.predict(Test_X_Tfidf)

# #SVM.fit(X_resample, y_resampled)
# #predictions_SVM = SVM.predict(X_test)

# # Use accuracy_score function to get the accuracy
# print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
# print("SVM Report ->", classification_report(Test_Y,predictions_SVM,zero_division=1))
# print(SVM.best_params_)


regressor = RandomForestClassifier(n_estimators=500, random_state=3)
# regressor = GridSearchCV(pipe1, param_grid = param_grid1, cv = 3, verbose=1, n_jobs=-1)
# regressor.fit(Train_X_Tfidf, Train_Y)
# y_pred = regressor.predict(Test_X_Tfidf)
regressor.fit(X_resample, y_resampled)
y_pred = regressor.predict(X_test)
print("Random Forrest Accuracy Score -> ", accuracy_score(y_pred.round(), Test_Y) * 100)
print("Random Forrest Report ->", classification_report(Test_Y,y_pred.round(), zero_division=1))
# with open('y_instagram-RF.csv', 'w') as f:
#     for x in Test_Y:
#         f.write(str(x)+'\n')
# with open('preds_instagram-RF.csv', 'w') as f:
#     for x in y_pred:
#         f.write(str(x)+'\n')
print(regressor.feature_importances_)
# print(regressor.best_params_)
input('press enter')
# neigh = KNeighborsClassifier(n_neighbors=10)

neigh = GridSearchCV(pipe2, param_grid = param_grid2, cv = 3,verbose=1, n_jobs=-1)
# neigh.fit(Train_X_Tfidf, Train_Y)
# y_pred = neigh.predict(Test_X_Tfidf)
neigh.fit(X_resample, y_resampled)
y_pred = neigh.predict(X_test)
print("K-Nearest Accuracy Score -> ", accuracy_score(y_pred.round(), Test_Y) * 100)
print("K-Nearest Report ->", classification_report(Test_Y,y_pred.round(), zero_division=1))
print(neigh.best_params_)

clf = MLPClassifier(activation='logistic',solver='adam', alpha=1e-5,hidden_layer_sizes=(8, 2), random_state=1)

# clf = GridSearchCV(pipe3, param_grid = param_grid3, cv = 3, verbose=1, n_jobs=-1)
# clf.fit(Train_X_Tfidf, Train_Y)
# y_pred = clf.predict(Test_X_Tfidf)
clf.fit(X_resample, y_resampled)
y_pred = clf.predict(X_test)
print("Multi-layer Neural Network Accuracy Score -> ", accuracy_score(y_pred.round(), Test_Y) * 100)
print("Multi-layer Neural Network Report ->", classification_report(Test_Y,y_pred.round(), zero_division=1))
# print(clf.best_params_)
# # Saving Encdoer, TFIDF Vectorizer and the trained model for future infrerencing/prediction

# saving encoder to disk
filename = 'labelencoder_fitted.pkl'
pickle.dump(Encoder, open(filename, 'wb'))

# saving TFIDF Vectorizer to disk
filename = 'Tfidf_vect_fitted.pkl'
pickle.dump(Tfidf_vect, open(filename, 'wb'))

# saving the both models to disk
# filename = 'svm_trained_model.sav'
# pickle.dump(SVM, open(filename, 'wb'))

# filename = 'nb_trained_model.sav'
# pickle.dump(Naive, open(filename, 'wb'))

filename = 'rf_trained_model.sav'
pickle.dump(regressor, open(filename, 'wb'))

# filename = 'k_near_trained_model.sav'
# pickle.dump(neigh, open(filename, 'wb'))

# filename = 'nn_trained_model.sav'
# pickle.dump(clf, open(filename, 'wb'))

print("Files saved to disk! Proceed to inference.py")
