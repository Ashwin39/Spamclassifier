import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from clean import text_clean
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
import pickle

#Importing dataset
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing

filename = 'spammodel.sav'
filename1 = 'vector.pickle'


def Train():

    ps = PorterStemmer()
    ldata = []

    clean_data = text_clean(ldata,ps,messages)
    
# Creating the Bag of Words model
    cv = CountVectorizer(max_features=2500) #Getting higher accuracy even with just 2500 words
    X = cv.fit_transform(clean_data).toarray()
    pickle.dump(cv, open(filename1, 'wb'))

    y=pd.get_dummies(messages['label'])
    y=y.iloc[:,1].values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Training model using Naive bayes classifier(works well with text data)
    spam_detect_model = MultinomialNB().fit(X_train, y_train)
    #Predictions
    y_pred=spam_detect_model.predict(X_test)

    #Calculating the roc_auc_score
    print(roc_auc_score(y_test,y_pred))

    #Saving the model
    pickle.dump(spam_detect_model,open(filename,'wb'))

if __name__ == '__main__':
    Train()


















