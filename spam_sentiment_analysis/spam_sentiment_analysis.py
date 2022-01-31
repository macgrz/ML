# Spam SMS sentiment analysis 
# SMS analysis: verbs, nouns, adjectives
# SMS spam classification with comparison of 3 models (untuned, scaled by StandardScaler): KNeighborsClassifier, SVC, KMeans
# With outputs

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

tfidf = TfidfVectorizer(min_df=10)
inputs = tfidf.fit_transform(df['text'])
tfidf

tfidf_df = pd.DataFrame(inputs.todense(), columns = tfidf.get_feature_names())
tfidf_df.head(10)

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

weights = list(zip(tfidf.get_feature_names(), model.coef_[0]))
weights.sort(key = lambda x:x[1])

weights_dict = dict(weights)
weights_dict

def count_sentiment(row):
    sent_value = 0
    row_dct = row.to_dict()
    for key, value in row_dct.items():
        if value != 0:
            sent_value += weights_dict[key]
    return sent_value
  
df['sentiment_score'] = tfidf_df.apply(count_sentiment, axis=1)
df.sort_values('sentiment_score', ascending=False)
     
import spacy
nlp = spacy.load("en_core_web_sm")
modeled_text = nlp(df_raw.iloc[-1][1])
modeled_text
  
print([token.text for token in modeled_text])
print([token.lower_ for token in modeled_text])
print([token.lower for token in modeled_text])
print([token.lemma_ for token in modeled_text])
print([token.pos_ for token in modeled_text])

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
