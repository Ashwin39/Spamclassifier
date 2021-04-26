import re
from nltk.corpus import stopwords

def text_clean(ldata,ps,messages):
    for i in range(0, len(messages)):
        review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
        review = review.lower()
        review = review.split()

        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        ldata.append(review)
    return ldata

def text_clean_test(ldata1,ps,text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    ldata1.append(review)
    return ldata1