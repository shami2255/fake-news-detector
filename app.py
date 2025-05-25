from flask import Flask, request, render_template
from joblib import load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string



d_stopword = set(stopwords.words('english'))
data_stm = PorterStemmer()

#clf = load('Naive_Bayes.joblib')
clf = load('Bagging.joblib')




def change_prepro (text):
    # uncapitalizie the text in the dataset
    text = text.lower()

    # Remove punctuation and digits fro 
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Tokenize the  text
    d_digit = word_tokenize(text)

    # Remove stop words from the text
    d_digit = [word for word in d_digit if word not in d_stopword]

    # Stem the words
    words = [data_stm.stem(word) for word in d_digit]
   
        # joining the words back in the string
    text = ' '.join(d_digit)

    return text


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    if text == "" or text.isspace() : return render_template('home.html')
    
    preprocessed_text = change_prepro(text)
    
    y_pred = clf.predict([preprocessed_text])
    if y_pred[0]== 1:
        result = 'This is a real news'
    else:
        result = 'This is a fake news'
    return render_template('result.html', result=result, text=text)

if __name__ == '__main__':
    app.run(debug=True)
