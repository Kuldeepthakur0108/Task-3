from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open(r'C:\Users\Administrator\Documents\3\Loan.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        data = request.form
        
        # Get user input from the form
        Married = 1 if data.get("Married") == 'Yes' else 0
        Graduate = 1 if data.get("Graduate") == 'Graduate' else 0
        Income = int(data.get("Income"))
        LoanAmount = int(data.get("LoanAmount"))
        LoanTerm = int(data.get("LoanTerm"))
        Credit_History = 1 if data.get("Credit_History") == '1' else 0

        # Only pass the features that your model expects
        user_input = np.array([[Married, Graduate, Income, LoanAmount, LoanTerm, Credit_History]])
        
        # Make the prediction
        model_output = model.predict(user_input)
        
        # Check (whether it's loan approval or not)
        if model_output[0] == 1:
            prediction = "Approved"
        else:
            prediction = "Not Approved"
        
        return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
