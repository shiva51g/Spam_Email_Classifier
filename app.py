from flask import Flask, render_template, request
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        message_vector = vectorizer.transform([message])
        result = model.predict(message_vector)[0]
        prediction = "Spam" if result == 1 else "Not Spam"
    return render_template('index.html', prediction=prediction) 

if __name__ == '__main__':
    app.run(debug=True)
