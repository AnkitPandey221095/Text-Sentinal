from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained ML model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['userInput']
    
    # Vectorize the input text
    transformed_input = vectorizer.transform([user_input])
    
    # Get prediction
    prediction = model.predict(transformed_input)
    
    return render_template('index.html', result=f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    app.run(debug=True)
