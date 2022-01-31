# MOVIE REVIEW SENTIMENT ANALYSIS OF 25000 REVIEWS
# WITH MODLES: KNN, GAUSSIANNB, LOGISTICREGRESSION, NEURALNETWORK
# COMPARISON OF MODELS SCORES AND ROC CURVES ON INTERACTIVE DASHBOARD ON LOCAL SERVER


### IMPORTING LIBRAIRES 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import plotly.express as px
import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import plotly.graph_objs as go
import dash_html_components as html

### DATA IMPORT
PATH = open(r"sentiment_movies.csv")
data_raw = pd.read_csv(PATH)

### DATA PREPROCESSING
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
data_raw['SentimentText'] = data_raw['SentimentText'].apply(clean_text)
data = data_raw

### CREATE TFIDF MATRIX, CHECK FOR NAN VALUES
tfidf = TfidfVectorizer(min_df=10)
inputs = tfidf.fit_transform(data['SentimentText'])
tfidf_df = pd.DataFrame(inputs.todense(), columns = tfidf.get_feature_names(), dtype='float16')
print(f"Number of NAN: {tfidf_df.isna().any().sum()}")

### FIND TOP 1000 WORDS WITH HIGHEST TFIDF VALUE
tfidf_1000 = tfidf_df.mean().nlargest(n=1000)
tfidf_1000_dict = tfidf_1000.to_dict()

### ADD COLUMNS ABOUT NUMBER OF WORDS AND AVERAGE WORD LENGHT IN REVIEW TEXT
word_counter = list(data['SentimentText'].str.split().apply(len))
tfidf_df['words'] = word_counter
word_lenght = data['SentimentText'].str.replace(' ', '').apply(len) / tfidf_df['words']
tfidf_df['word_lenght'] = word_lenght

### ADD SENTIMENT COLUMN
sentiment_list = data['Sentiment']
tfidf_df['sentiment'] = sentiment_list
tfidf_df_2 = tfidf_df.dropna()

### SPLIT DATA TO TRAIN AND TEST, SCALE THE TRAIN SET
X_unscaled = tfidf_df_2.drop(labels=['sentiment'], axis=1)
scaler = StandardScaler().fit(X_unscaled)
Y = tfidf_df_2['sentiment']
X = scaler.transform(X_unscaled).astype('float16')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state=42)

### FIND LIST OF TOP 10 WORDS WITH HIGHEST TFIDF VALUE
tfidf_10 = tfidf_df.drop(labels=['sentiment', 'words', 'word_lenght'], axis=1).mean().nlargest(n=10)
column_names = list(tfidf_10.keys())
column_names

### CREATE CORRELATION MATRIX
df_corr = tfidf_df.filter(items=column_names)
corrMatrix = df_corr.corr()
sn.set(rc={'figure.figsize':(11.7,8.27)})
sn.heatmap(corrMatrix, annot=True)
plt.show()

### GET TOP 5 CORRELATION PAIRS
df_corr = tfidf_df.filter(items=column_names)
corrMatrix = df_corr.corr()

def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df_corr.columns
    for i in range(0, df_corr.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
    
def get_top_abs_correlations(df, n=5):
    au_corr = df_corr.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations:")
print(get_top_abs_correlations(df_corr, 5))

### T-STUDENT TEST (ALPHA = 0.05)
sentiment_0 = tfidf_df[tfidf_df['sentiment'] == 0]['word_lenght'].dropna()
sentiment_1 = tfidf_df[tfidf_df['sentiment'] == 1]['word_lenght'].dropna()
sentiment_0_array = sentiment_0.to_numpy()
sentiment_1_array = sentiment_1.to_numpy()
stat, pvalue = ttest_ind(sentiment_0_array, sentiment_1_array, equal_var=True, alternative='two-sided', axis=0)
print(f"P-value of T-Student test: {pvalue}")
print(f"Mean lenght of words - positive reviews: {sentiment_0_array.mean()}")
print(f"Mean lenght of words - negative reviews: {sentiment_1_array.mean()}")

### COMPARE AVERAGE LENGHT OF WORDS IN NEGATIVE AND POSITIVE REVIEWS ON HISTOGRAM
plt.hist(sentiment_0, alpha=0.5, bins=30)
plt.hist(sentiment_1, alpha=0.5, bins=30)
plt.ylabel("Number of reviews", fontsize=12)  
plt.xlabel("Average word lenght in review", fontsize=12)
plt.legend(['Positive review', 'Negative review'])
plt.show()

### CREATE CONFUSION MATRIX FUNCTION
def cm_matrix_plot(cm_matrix): # function to plot confusion matrixes
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(cm_matrix)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_matrix[i, j], ha='center', va='center', color='red')
    plt.show()
    
### LOGISTIC REGRESSION FUNCTION
print("\nLOGISTIC REGRESSION MODEL\n")
steps = [('LOG_REG', LogisticRegression())]
pipeline = Pipeline(steps)
lr_model = pipeline.fit(X_train, y_train)
lr_model_prediction = lr_model.predict(X_test)
lr_score_ = lr_model.score(X_test, y_test)
lr_cm = confusion_matrix(y_test, lr_model_prediction)
lr_class_report = classification_report(y_test, lr_model_prediction)
print(f"Logistic Regression confusion matrix:")
cm_matrix_plot(lr_cm)
print(f"Logistic Regression model score: {lr_score_}")
print(f"Logistic Regression classification report:")
print(lr_class_report)

### LOGISTIC REGRESSION SCORES
log_reg_scores = precision_recall_fscore_support(y_test, lr_model_prediction, pos_label=None)
logreg_metric =     {
    'accuracy' : round(lr_score_, 4),
    'precision' : round(log_reg_scores[0].mean(), 4),
    'recall' : round(log_reg_scores[1].mean(), 4),
    'fscore' : round(log_reg_scores[2].mean(), 4),
    'roc' : round(roc_auc_score(y_test, lr_model.decision_function(X_test)), 4),
                    }
print(f"Precision for LogisticRegression: {logreg_metric['precision']}")
print(f"Recall for LogisticRegression: {logreg_metric['recall']}")
print(f"ROC for LogisticRegression: {logreg_metric['roc']}")
print(f"Accuracy for LogisticRegression: {logreg_metric['accuracy']}")
print(f"F1-Score for LogisticRegression: {logreg_metric['fscore']}")

### KNEIGHBORSCLASSIFIER FUNCTION
print("\nKNEIGHBORSCLASSIFIER MODEL\n")
steps = [('knn', KNeighborsClassifier(n_neighbors=3))]
pipeline = Pipeline(steps)
knn_model = pipeline.fit(X_train, y_train)
knn_model_prediction = knn_model.predict(X_test)
knn_score_ = knn_model.score(X_test, y_test)
knn_cm = confusion_matrix(y_test, knn_model_prediction)
knn_class_report = classification_report(y_test, knn_model_prediction)
print(f"KNN confusion matrix:")
cm_matrix_plot(knn_cm)
print(f"KNN model score: {knn_score_}")
print(f"KNN classification report:")
print(knn_class_report)

### KNEIGHBORSCLASSIFIER SCORES
knn_scores = precision_recall_fscore_support(y_test, knn_model_prediction, pos_label=None)
knn_proba = knn_model.predict_proba(X_test)
knn_metric =    {
    'accuracy' : round(knn_score_, 4),
    'precision' : round(knn_scores[0].mean(), 4),
    'recall' : round(knn_scores[1].mean(), 4),
    'fscore' : round(knn_scores[2].mean(), 4),
    'roc' : round(roc_auc_score(y_test, knn_proba[:, 1]), 4),
                }
print(f"Precision for KNeighborsClassifier: {knn_metric['precision']}")
print(f"Recall for KNeighborsClassifier: {knn_metric['recall']}")
print(f"ROC for KNeighborsClassifier: {knn_metric['roc']}")
print(f"Accuracy for KNeighborsClassifier: {knn_metric['accuracy']}")
print(f"F1-Score for KNeighborsClassifier: {knn_metric['fscore']}")

### NAIVE BAYES CLASSIFIER FUNCTION
print("\nNAIVE BAYES CLASSIFIER MODEL\n")
steps = [('g_nb', GaussianNB())]
pipeline = Pipeline(steps)
g_model = pipeline.fit(X_train, y_train)
g_model_prediction = g_model.predict(X_test)
g_score_ = g_model.score(X_test, y_test)
g_cm = confusion_matrix(y_test, g_model_prediction)
g_class_report = classification_report(y_test, g_model_prediction)
print(f"Naive Bayes Classifier confusion matrix:")
cm_matrix_plot(g_cm)
print(f"Naive Bayes Classifier model score: {g_score_}")
print(f"Naive Bayes Classifier classification report:")
print(g_class_report)

### NAIVE BAYES CLASSIFIER SCORES
g_scores = precision_recall_fscore_support(y_test, g_model_prediction, pos_label=None)
g_proba = g_model.predict_proba(X_test)
g_metric =    {
    'accuracy' : round(g_score_, 4),
    'precision' : round(g_scores[0].mean(), 4),
    'recall' : round(g_scores[1].mean(), 4),
    'fscore' : round(g_scores[2].mean(), 4),
    'roc' : round(roc_auc_score(y_test, g_proba[:, 1]), 4),
                }
print(f"Precision for NaiveBayesClassifier: {g_metric['precision']}")
print(f"Recall for NaiveBayesClassifier: {g_metric['recall']}")
print(f"ROC for NaiveBayesClassifier: {g_metric['roc']}")
print(f"Accuracy for NaiveBayesClassifier: {g_metric['accuracy']}")
print(f"F1-Score for NaiveBayesClassifier: {g_metric['fscore']}")

### TRAIN NETWORK
nn_metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'accuracy', tf.keras.metrics.AUC()]
model = Sequential()
model.add(Dense(units=60, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=6, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00003)
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=nn_metrics)
epoch=8
history = model.fit(X_train, y_train, epochs=epoch, validation_data=(X_test, y_test))
nn_model_prediction = model.predict(scaler.transform(X_test)).astype(int)
model.summary()

### CREATE TRAIN AND LOSS PLOTS
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("loss")
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="valid")
plt.legend()
plt.subplot(1,2,2)
plt.title("accuracy")
plt.plot(history.history['accuracy'], label="train")
plt.plot(history.history['val_accuracy'], label="valid")
plt.legend()
plt.show()

nn_metrics =    {
'precision': history.history['val_precision'][epoch-1],
'recall': history.history['val_recall'][epoch-1],
'roc':history.history['val_auc'][epoch-1],
'accuracy':history.history['val_accuracy'][epoch-1],
                }
### CREATE DATAFRAME COMPARISON OF MODELS            
all_metrics = pd.DataFrame([logreg_metric, g_metric, knn_metric, nn_metrics], index=['LogisticRegression', 'NaiveBayesClassifier', 'KNeighborsClassifier', 'NeuralNetwork'])
all_metrics.sort_values(by="accuracy", ascending=False)

### GET ROC CURVES
def get_roc_curve(y_pred, y_test, n_classes = 2):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_pred, y_test)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    return fpr, tpr, roc_auc 
log_regr_roc_curve = get_roc_curve(lr_model_prediction, y_test)
bayes_roc_curve = get_roc_curve(g_model_prediction, y_test)
knn_roc_curve = get_roc_curve(knn_model_prediction, y_test)  
nn_roc_curve = get_roc_curve(nn_model_prediction, y_test)

### PLOT ROC CURVES
roc_curve = [(nn_roc_curve, "Artificial_NN"), (knn_roc_curve, "KNeighborsClassifier"), (log_regr_roc_curve, "LogisticRegression"),(bayes_roc_curve,"NaiveBayesClassifier") ]
plt.figure(figsize=(10,5))

for cls in range(0,2):
    plt.subplot(1,2, cls+1)
    for (fpr, tpr, roc), label in roc_curve:
        plt.title(f"ROC curve for sentiment {cls} analysis")
        plt.plot(fpr[cls], tpr[cls], label = f"{label}")
    plt.plot([0, 1], [0, 1], color="red", lw=1, linestyle="dotted")
    plt.legend()
plt.show()

### PLOTS
metrics_plot = px.bar(all_metrics, barmode='group',width=700, height=500)
roc_regr_plot = px.area(x=log_regr_roc_curve[0][0], y=log_regr_roc_curve[1][0],
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
roc_bayes_plot = px.area(x=bayes_roc_curve[0][0], y=bayes_roc_curve[1][0],
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
roc_knn_plot = px.area(x=knn_roc_curve[0][0], y=knn_roc_curve[1][0],
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)
roc_nn_plot = px.area(x=nn_roc_curve[0][0], y=nn_roc_curve[1][0],
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)

### DASHBOARD
app = dash.Dash(__name__)
app.layout=html.Div([
    html.H1("Movie review sentiment analysis - comparison of models"),
    html.Div(id="no-show", style={'display':'none'}),
    dcc.RadioItems(
        id='radio',
        options = [ {'label': 'METRICS', 'value': 'met'}, {'label': 'BAYES', 'value': 'bayes'},
                    {'label': 'LOGREG', 'value': 'regr'}, {'label': 'KNN', 'value': 'knn'},
                    {'label': 'NN', 'value': 'nn'}],
                    value='value',
                    labelStyle={'display':'inline-block'}
                  ),

    dcc.Graph(id = "graph", figure=go.Figure())
                    ],
    style={'text-align':'center'}
                   )
@app.callback(
dash.dependencies.Output("graph", "figure"),
[dash.dependencies.Input("radio", "value")]
)
def update_output(value):
    plts = {'met': metrics_plot, 
    'bayes': roc_bayes_plot, 
    'regr': roc_regr_plot, 
    'knn': roc_knn_plot, 
    'nn':roc_nn_plot}
    return plts[value]

if __name__ == '__main__':
    app.run_server(debug=False, use_reloader=False)
