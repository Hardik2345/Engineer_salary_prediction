import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Engineering_graduate_salary.csv')

    # Drop unnecessary columns
df.drop(['ID','DOB','10board','12board','12graduation','CollegeID','CollegeState','CollegeCityID','CollegeCityTier','GraduationYear'], axis = 1,inplace=True)
    
    # Drop duplicate rows
df = df.drop_duplicates()
    
    # Replace values in 'Specialization' column with 'other' if count is less than 10
specialization = df.Specialization.value_counts(ascending=False)
specializationlessthan10 = specialization[specialization <= 10]
def removespecializationlessthan10(value):
    if value in specializationlessthan10:
        return 'other'
    else:
        return value
df.Specialization = df.Specialization.apply(removespecializationlessthan10)
    
    # Replace -1 values with NaN
df = df.replace(-1, np.nan)
    
    # Fill NaN values with column mean
cols_with_nan = [column for column in df.columns if df.isna().sum()[column] > 0]
for column in cols_with_nan:
     df[column] = df[column].fillna(df[column].mean())
    
    # Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df.drop(["10percentage"],axis=1,inplace=True)
df.Degree = le.fit_transform(df.Degree)
df.Specialization = le.fit_transform(df.Specialization)
    

X = df.drop('Salary', axis=1)  # Replace 'target_column_name' with the name of your target variable column
y = df['Salary'] 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_X.fit(X_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_reshaped = y_train.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)


scaler_y = StandardScaler()

y_train_scaled = scaler_y.fit_transform(y_train_reshaped)
y_test_scaled = scaler_y.transform(y_test_reshaped)

XGB_regressor= xgb.XGBRegressor()
XGB_regressor.fit(X_train_scaled, y_train)


pickle.dump(XGB_regressor, open("model.pkl", "wb"))
