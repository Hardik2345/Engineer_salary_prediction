from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_folder='static')
app = Flask(__name__, static_url_path='/static')

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the preprocessor function
def preprocess_data(df):
    # Convert object columns to numeric types
    df = df.astype(float)

    # df = df.replace(-1, np.nan)
    
    # # Fill NaN values with column mean
    # cols_with_nan = [column for column in df.columns if df.isna().sum()[column] > 0]
    # for column in cols_with_nan:
    #     df[column] = df[column].fillna(df[column].mean())
    
    # # Encode categorical variables
    # le = LabelEncoder()
    # df['Gender'] = le.fit_transform(df['Gender'])
    # # df.drop(["10percentage"],axis=1,inplace=True)
    # df.Degree = le.fit_transform(df.Degree)
    # df.Specialization = le.fit_transform(df.Specialization)
    
    return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    data = request.form.to_dict()

    # Convert the form data to a DataFrame
    df = pd.DataFrame([data])

    # Preprocess the input data
    processed_data = preprocess_data(df)

    # Make a prediction using the trained model
    prediction = model.predict(processed_data)

    # Render the template with the prediction
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)