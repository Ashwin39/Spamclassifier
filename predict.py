from Train import filename,filename1
from clean import text_clean_test
from nltk.stem.porter import PorterStemmer
import pickle
from pywebio.input import *
from pywebio.output import *
from pywebio.platform.flask import webio_view
from pywebio import STATIC_PATH
from flask import Flask, send_from_directory

app = Flask(__name__)

def predict():

    text = input("Please enter the message",type="text")
    ps = PorterStemmer()
    ldata1 = []

    cleaned_testdata = text_clean_test(ldata1,ps,text)

    #Transforming the test data to vectors using BOW
    cv = pickle.load(open(filename1,'rb'))
    X1=cv.transform(cleaned_testdata).toarray()

    model = pickle.load(open(filename,'rb'))
    spam = model.predict(X1)
    if spam == 0:
        put_text('The message is not a SPAM')
    else:
        put_text('The message is a SPAM')

app.add_url_rule('/spam','webio_view',webio_view(predict),methods=['GET','POST','OPTIONS'])
app.run(host='localhost',port=80)