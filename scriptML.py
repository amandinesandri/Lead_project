
import pandas as pd
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from joblib import dump, load

# Load your dataset and perform feature engineering
data = pd.read_csv("https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv")

data.head()

print(data.columns.tolist())

data['age'] = dt.date.today().year - pd.to_datetime(data['dob']).dt.year
data['hour'] = pd.to_datetime(data['trans_date_trans_time']).dt.hour
data['day'] = pd.to_datetime(data['trans_date_trans_time']).dt.dayofweek
data['month'] = pd.to_datetime(data['trans_date_trans_time']).dt.month

fraud_data = data[['category', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long', 'age', 'hour', 'day', 'month', 'is_fraud']]

# Split your data into features (X) and target (y)
X = fraud_data.drop(columns=['is_fraud'])
y = fraud_data['is_fraud']

# Automatically detect names of numeric/categorical columns
numeric_features = []
categorical_features = []

for column in X.columns:
    if X[column].dtype == 'int64' or X[column].dtype == 'float64':
        numeric_features.append(column)
    else:
        categorical_features.append(column)


numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split your data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Preprocessings on train set
X_train = preprocessor.fit_transform(X_train)

#Label encoding
encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)


#Preprocessings on test set
X_test = preprocessor.transform(X_test)

#Label encoding
Y_test = encoder.transform(Y_test)

#label_encoder = LabelEncoder()

#Apply label encoding to the 'Category' column
#fraud_data['category'] = label_encoder.fit_transform(fraud_data['category'])

model = RandomForestClassifier()


# Train the model
model.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

X_test = pd.DataFrame(X_test)

# Choose a row from X_test for prediction (e.g., the first row)
row_to_predict = X_test.iloc[0]

# Reshape the row into a 2D array (the model expects a 2D array)
reshaped_row = row_to_predict.values.reshape(1, -1)

# Make a prediction
prediction = model.predict(reshaped_row)

# Print the prediction
print("Prediction:", prediction)

dump(model, "model.joblib")