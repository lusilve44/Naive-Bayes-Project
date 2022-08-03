import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
import pickle
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


# Read csv

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')


 
# 1. Transform dataframe
df_transf = df_raw.copy()


# Drop package_name column

df_transf = df_transf.drop('package_name', axis=1)



df_transf['review'] = df_transf['review'].str.strip()

# elimina espacio libre al principio y al final


# column review to lower case
df_transf['review'] = df_transf['review'].str.lower()


stop = stopwords.words('english')

def clean_words(review):
    if review is not None:
        words = review.strip().split()
        new_words = []
        for word in words:
            if word not in stop:new_words.append(word)
        Result = ' '.join(new_words)    
    else:
        Result = None
    return Result



df_transf['review'] = df_transf['review'].apply(clean_words)


df = df_transf.copy()

 
# 2. Split data frame

X = df['review']
y = df['polarity']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=25)


 
# 3. Pipeline with two pre-processing steps and one modeling step


vect = CountVectorizer() # vector de conteo

text_vec = vect.fit_transform(X_train)


vect.get_feature_names_out() # name the columns


vect_tfidf = TfidfVectorizer()

text_vec_tfidf = vect_tfidf.fit_transform(X_train)


text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer())])
text_norm = text_clf.fit_transform(X_train)



clf_1 = MultinomialNB()

clf_1.fit(text_vec, y_train)


clf_2 = MultinomialNB()

clf_2.fit(text_vec_tfidf, y_train)


clf_3 = MultinomialNB()

clf_3.fit(text_norm, y_train)


pred_1 = clf_1.predict(vect.transform(X_test))
pred_2 = clf_2.predict(vect_tfidf.transform(X_test))
pred_3 = clf_3.predict(text_clf.transform(X_test))


# Acá todo junto en un pipeline

text_clf_2 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf_2.fit(X_train, y_train)

 
# 4. Check results


y_pred = text_clf_2.predict(X_test)
precision_recall_fscore_support(y_test, y_pred, average='weighted')

 
# 5. Randomized search to select hyperparameters


n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf = RandomizedSearchCV(text_clf_2, parameters, n_iter = n_iter_search)
gs_clf.fit(X_train, y_train)



text_clf_count_vect = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
text_clf_count_vect.fit(X_train, y_train)


n_iter_search = 5
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'clf__alpha': (1e-2, 1e-3)}
gs_count_vect = RandomizedSearchCV(text_clf_count_vect, parameters, n_iter = n_iter_search)
gs_count_vect.fit(X_train, y_train)


y_pred_mejor = gs_clf.predict(X_test)

best_model = gs_clf.best_estimator_



# Save best model
filename = '/workspace/Naive-Bayes-Project/models/best_model.sav'
pickle.dump(best_model, open(filename,'wb'))


df.to_csv('/workspace/Naive-Bayes-Project/data/processed/dataframe_processed.csv', index = False, encoding='utf-8')