from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('suicide_prediction_model.pkl')

# Define the route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the form
        state = request.form['state']
        type_code = request.form['type_code']
        type_suicide = request.form['type']
        gender = request.form['gender']
        age_group = request.form['age_group']
        year = int(request.form['year'])
        
        # Create a DataFrame from user input
        user_data = pd.DataFrame({
            'State': [state],
            'Type_code': [type_code],
            'Type': [type_suicide],
            'Gender': [gender],
            'Age_group': [age_group],
            'Year': [year]
        })
        
        # Convert categorical columns to numeric using one-hot encoding
        user_data = pd.get_dummies(user_data, columns=['State', 'Type_code', 'Type', 'Gender', 'Age_group'], drop_first=True)
        
        # Ensure all necessary columns are present in the user_data
        missing_cols = set(X.columns) - set(user_data.columns)
        for c in missing_cols:
            user_data[c] = 0
        user_data = user_data[X.columns]
        
        # Make prediction
        prediction = model.predict(user_data)
        
        return render_template('index.html', prediction_text=f'Predicted category of suicide rate: {prediction[0]}')
    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)