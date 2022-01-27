import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix

PATH = r'SMSSpamCollection.txt'
col_names = ['spam', 'text']
df_raw = pd.read_csv(PATH, sep='\t', names=col_names)
df = pd.read_csv(PATH, sep='\t', names=col_names)

def clean_text(x):
    x = x.lower()
    x = x.replace('.','')
    x = x.replace(',','')
    x = x.replace(':','')
    x = x.replace(';','')
    x = x.replace('!','')
    x = x.replace('?','')
    x = x.replace('<br>','')
    x = x.replace('<br />','')
    x = ' '.join([word for word in x.split() if word.isalpha()])
    return x

df['text'] = df['text'].apply(clean_text)
df['spam'].replace({"ham": 0, "spam": 1}, inplace=True)
df

        ### OUTPUT #################
        spam	text
        0	0	go until jurong point crazy available only in ...
        1	0	ok lar joking wif u oni
        2	1	free entry in a wkly comp to win fa cup final ...
        3	0	u dun say so early hor u c already then say
        4	0	nah i think he goes to usf he lives around her...
        ...	...	...
        ############################

tfidf = TfidfVectorizer(min_df=10)
inputs = tfidf.fit_transform(df['text'])
tfidf

        ### OUTPUT #################
        TfidfVectorizer(min_df=10)
        ############################

tfidf_df = pd.DataFrame(inputs.todense(), columns = tfidf.get_feature_names())
tfidf_df.head(10)

        ### OUTPUT #################
          abiola	able	about	abt	account	across	actually	address	admirer	aft	...	yes	yesterday	yet	yo	you	your	yours	yourself	yr	yup
        0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.000000	0.0	0.0	0.0	0.0
        1	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.000000	0.0	0.0	0.0	0.0
        2	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.000000	0.0	0.0	0.0	0.0
        3	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.000000	0.0	0.0	0.0	0.0
        4	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.000000	0.0	0.0	0.0	0.0
        5	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.108801	0.000000	0.0	0.0	0.0	0.0
        6	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.000000	0.0	0.0	0.0	0.0
        7	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.478202	0.0	0.0	0.0	0.0
        8	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.097003	0.000000	0.0	0.0	0.0	0.0
        9	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.000000	0.120464	0.0	0.0	0.0	0.0
        10 rows × 909 columns
        ############################

target = df['spam']

train_input,test_input,train_target,test_target = train_test_split(inputs,target, test_size=0.25)

model = LogisticRegression()
model.fit(train_input,train_target)

predictions = model.predict(test_input)


crosstab = pd.crosstab(test_target, predictions, rownames = ['Actual'], colnames =['Predicted'], margins = True)
print('Cross table:')
print(crosstab)
print('\n')

print("Classification report:")
print(classification_report(test_target, predictions))

print(f"Spam detection precision: {round(precision_score(test_target,predictions),4)}")
print(f"Spam detection recall: {round(recall_score(test_target,predictions),4)}")
print(f"No-spam detection precision:: {round(precision_score(test_target,predictions,pos_label = 0),4)}")
print(f"No-spam detection recall:: {round(recall_score(test_target,predictions,pos_label = 0),4)}")
print(f"Classification accuracy: {round(accuracy_score(test_target,predictions),4)}")

        ### OUTPUT #################
        Cross table:
        Predicted     0    1   All
        Actual                    
        0          1193    6  1199
        1            47  147   194
        All        1240  153  1393


        Classification report:
                      precision    recall  f1-score   support

                   0       0.96      0.99      0.98      1199
                   1       0.96      0.76      0.85       194

            accuracy                           0.96      1393
           macro avg       0.96      0.88      0.91      1393
        weighted avg       0.96      0.96      0.96      1393

        Spam detection precision: 0.9608
        Spam detection recall: 0.7577
        No-spam detection precision:: 0.9621
        No-spam detection recall:: 0.995
        Classification accuracy: 0.962
        ############################

weights = list(zip(tfidf.get_feature_names(), model.coef_[0]))
weights.sort(key = lambda x:x[1])

weights_dict = dict(weights)
weights_dict

        ### OUTPUT #################
        {'me': -2.4843179296166955,
         'my': -2.125869854480259,
         'later': -1.663829114778846,
         'come': -1.4984563148888868,
         'ok': -1.4824560050376951,
         'sorry': -1.304893840504233,
         ############################
         
def count_sentiment(row):
    sent_value = 0
    row_dct = row.to_dict()
    for key, value in row_dct.items():
        if value != 0:
            sent_value += weights_dict[key]
    return sent_value
  
df['sentiment_score'] = tfidf_df.apply(count_sentiment, axis=1)
df.sort_values('sentiment_score', ascending=False)
         
        ### OUTPUT #################
          spam	text	sentiment_score
        673	1	get ur ringtone free now reply to this msg wit...	36.146277
        3999	1	we tried to call you re your reply to our sms ...	32.469963
        583	1	we tried to contact you re your reply to our o...	30.885722
        4091	1	we tried to call you re your reply to our sms ...	30.344716
        5012	1	you have won a guaranteed cash or a prize to c...	30.133227
        ...	...	...	...
        ############################
         
import spacy
nlp = spacy.load("en_core_web_sm")
modeled_text = nlp(df_raw.iloc[-1][1])
modeled_text
         
         ### OUTPUT #################
         Rofl. Its true to its name
         ############################

print([token.text for token in modeled_text])
print([token.lower_ for token in modeled_text])
print([token.lower for token in modeled_text])
print([token.lemma_ for token in modeled_text])
print([token.pos_ for token in modeled_text])
         
        ### OUTPUT #################
        ['rofl', '.', 'its', 'true', 'to', 'its', 'name']
        [10885026304460510830, 12646065887601541794, 12513610393978129441, 7434368892455186804, 3791531372978436496, 12513610393978129441, 18309932012808971453]
        ['Rofl', '.', 'its', 'true', 'to', 'its', 'name']
        ['PROPN', 'PUNCT', 'PRON', 'ADJ', 'ADP', 'PRON', 'NOUN']
        ############################

from __future__ import unicode_literals
from collections import Counter
advs = []
nouns = []
verbs = []
summ = []

for i in range(len(df_raw)):
    modeled_text_df = nlp(df_raw.iloc[i][1])
    c = Counter(([token.pos_ for token in modeled_text_df]))
    advs.append(c['ADV'])
    nouns.append(c['NOUN'])
    verbs.append(c['VERB'])
    summ.append(c['ADV']+c['NOUN']+c['VERB'])
    
df_raw['SUM'] = summ
df_raw['ADVS'] = advs
df_raw['ADVS_%'] = round(100*df_raw['ADVS'] / df_raw['SUM'])
df_raw['NOUNS'] = nouns
df_raw['NOUNS_%'] = round(100*df_raw['NOUNS'] / df_raw['SUM'])
df_raw['VERBS'] = verbs
df_raw['VERBS_%'] = round(100*df_raw['VERBS'] / df_raw['SUM'])
df_raw
         ### OUTPUT #################
         spam	text	SUM	ADVS	ADVS_%	NOUNS	NOUNS_%	VERBS	VERBS_%
        0	ham	Go until jurong point, crazy.. Available only ...	8	2	25.0	4	50.0	2	25.0
        1	ham	Ok lar... Joking wif u oni...	3	0	0.0	2	67.0	1	33.0
        2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	11	0	0.0	6	55.0	5	45.0
        3	ham	U dun say so early hor... U c already then say...	7	3	43.0	1	14.0	3	43.0
         ############################
         
tfidf_df['CLASS'] = df['spam']

# KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
X = tfidf_df.drop(labels=['CLASS'], axis=1)
Y = tfidf_df['CLASS']
steps = [('scaler', StandardScaler()), ('KNN', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
knn_scaled_untuned = pipeline.fit(X_train, y_train)
knn_predict = knn_scaled_untuned.predict(X_test)

print("KNN (scaled, untuned) classification report:")
print(classification_report(y_test, knn_predict))
print(f"KNN (scaled, untuned) accuracy: {round(pipeline.score(X_test, y_test), 4)*100}%")
         
         ### OUTPUT #################
                   KNN (scaled, untuned) classification report:
                        precision    recall  f1-score   support

                     0       0.94      0.99      0.97       966
                     1       0.93      0.60      0.73       149

              accuracy                           0.94      1115
             macro avg       0.93      0.80      0.85      1115
          weighted avg       0.94      0.94      0.94      1115

          KNN (scaled, untuned) accuracy: 94.08%
         ############################

# SVC
from sklearn.svm import SVC
X = tfidf_df.drop(labels=['CLASS'], axis=1)
Y = tfidf_df['CLASS']
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
svc_scaled_untuned = pipeline.fit(X_train, y_train)
svc_pred = svc_scaled_untuned.predict(X_test)

print("SVC (scaled, untuned) classification report:")
print(classification_report(y_test, svc_pred))
print(f'SVC (scaled, untuned) accuracy: {round(svc_scaled_untuned.score(X_test, y_test), 2)*100}%')
         
         ### OUTPUT #################
                   SVC (scaled, untuned) classification report:
                        precision    recall  f1-score   support

                     0       0.98      1.00      0.99       960
                     1       1.00      0.86      0.92       155

              accuracy                           0.98      1115
             macro avg       0.99      0.93      0.96      1115
          weighted avg       0.98      0.98      0.98      1115

          SVC (scaled, untuned) accuracy: 98.0%
         ############################
         
# KMEANS
from sklearn.cluster import KMeans
X = tfidf_df.drop(labels=['CLASS'], axis=1)
Y = tfidf_df['CLASS']
steps = [('scaler', StandardScaler()), ('KMEANS', KMeans(2))]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
kmeans_scaled_untuned = pipeline.fit(X_train)
kmeans_pred = kmeans_scaled_untuned.predict(X_test)

print("KMeans (scaled, untuned) classification report:")
print(classification_report(y_test, kmeans_pred))
         
         ### OUTPUT #################
                   KMeans (scaled, untuned) classification report:
                        precision    recall  f1-score   support

                     0       0.02      0.00      0.00       966
                     1       0.04      0.23      0.06       149

              accuracy                           0.03      1115
             macro avg       0.03      0.12      0.03      1115
          weighted avg       0.02      0.03      0.01      1115
         ############################
         
  
