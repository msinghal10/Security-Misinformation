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
import ast
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from imblearn.over_sampling import *
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):

    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


# Loading Label encoder
labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))

# Loading TF-IDF Vectorizer
Tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))

# Loading models
# SVM = pickle.load(open('svm_trained_model.sav', 'rb'))
# Naive = pickle.load(open('nb_trained_model.sav', 'rb'))
Forrest=pickle.load(open('rf_trained_model.sav','rb'))
# KN=pickle.load(open('k_near_trained_model.sav','rb'))
# NN=pickle.load(open('nn_trained_model.sav','rb'))
# Inference
pd.set_option('mode.use_inf_as_na', True)
Corpus = pd.read_csv(r"twitter.csv", encoding='utf-8')

C1= Corpus[Corpus['Label'].isna()]
C1.to_csv(r'classifier1-output-nolabels.csv',index=False)
# print(C1)
# input('h')
# C1['Description']=C1['Description'].dropna()
# Corpus['Description'] = Corpus['Description'].map(csleaner)




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

C1['full_text'] = C1['full_text'].map(csleaner)

# Corpus['full_text'] = Corpus['full_text'].map(csleaner)

C1['full_text'].map(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",str(x)).split())
# Corpus['full_text'].map(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","",str(x)).split())
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

C1['full_text'] = C1['full_text'].map(deEmojify)

C1['full_text']=C1['full_text'].dropna()
# Corpus['full_text'] = Corpus['full_text'].map(deEmojify)

# Corpus['Description']=Corpus['full_text'].dropna()

C1['text_final'] = C1['full_text'].map(text_preprocessing)
# Corpus['text_final'] = Corpus['Description'].map(text_preprocessing)
#Add temp part here
# temp=[]
# temp1=[]
# C1=C1.reset_index(drop=True)
# print(C1)
# # Test_X=Test_X.reset_index(drop=True)
# # Corpus.reset_index()
# for x in range(len(C1)):
#     found = Corpus.loc[Corpus['text_final']==C1['text_final'][x]]
#     print(found)
#     temp.append(found.iloc[0]['Likes'])


sample_text_processed_vectorized = Tfidf_vect.transform(C1['text_final'])


# temp = np.array(temp)

# temp = np.reshape(temp, (len(temp), 1))
# # temp1 = np.reshape(temp1, (len(temp1), 1))

# print(sample_text_processed_vectorized.shape)
# print(temp.shape)
# # print(Test_X_Tfidf.shape)
# # print(temp1.shape)

# X_train=sample_text_processed_vectorized.toarray()
# # X_test=Test_X_Tfidf.toarray()
# # print(X_train)
# # print(temp)
# # input('enter enter when ready')
# X_train=np.append(X_train,temp,axis=1)
# # X_test=np.append(X_test,temp1,axis=1)

# print(X_train.shape)
# # print(X_test.shape)
# # input('tt')

# x_train = pd.DataFrame(X_train)



# prediction_SVM = SVM.predict(sample_text_processed_vectorized)
# prediction_Naive = Naive.predict(sample_text_processed_vectorized)
prediction_forrest= Forrest.predict(sample_text_processed_vectorized)
np.savetxt("labels-tw-class1.csv",prediction_forrest,delimiter=",")
# prediction_Knear=KN.predict(sample_text_processed_vectorized)
# prediction_nn=NN.predict(sample_text_processed_vectorized)
# print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM)[0])
# print("Prediction from NB Model:", labelencode.inverse_transform(prediction_Naive)[0])
print("Prediction from RandomForest Model:", labelencode.inverse_transform(prediction_forrest)[0])
# print("Prediction from K-Nearest Model:", labelencode.inverse_transform(prediction_Knear)[0])
# print("Prediction from Multi Layer NN Model:", labelencode.inverse_transform(prediction_nn)[0])
