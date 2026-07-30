import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib

# Step 1: Load the dataset
# Assuming your dataset is in a CSV file named 'credit_card_data.csv'
df = pd.read_csv('credit_card_data.csv')

# Step 2: Data Preprocessing
le = LabelEncoder()
df['trans_date_trans_time'] = df['trans_date_trans_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp())
df['trans_num'] = le.fit_transform(df['trans_num'])

# Step 3: Feature Selection
X = df[['trans_date_trans_time', 'trans_num']]
y = df['is_fraud']

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Save the trained model to a file
joblib.dump(model, 'fraud_detection_model.pkl')

# Step 7: Model Evaluation
y_pred = model.predict(X_test)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)
confusion_matrix_result = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_report_result)
print("\nConfusion Matrix:\n", confusion_matrix_result)

# Step 8: Prediction Function for User Input
def predict_fraud(user_input):
    user_input['trans_date_trans_time'] = datetime.strptime(user_input['trans_date_trans_time'], "%Y-%m-%d %H:%M:%S").timestamp()
    user_input['trans_num'] = le.transform([user_input['trans_num']])
    user_data = pd.DataFrame([user_input])
    # Step 9: Load the saved model for prediction
    loaded_model = joblib.load('fraud_detection_model.pkl')
    prediction = loaded_model.predict(user_data)
    return prediction[0]

# Step 10: Example User Input and Prediction
user_input = {
    'trans_date_trans_time': '2019-01-01 12:00:00',
    'trans_num': '0b242abb623afc578575680df30655b9'
}

fraud_prediction = predict_fraud(user_input)

print(f"The prediction for fraud is: {fraud_prediction}")
