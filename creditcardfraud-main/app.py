import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Fraud Detection App")

    # Step 1: Load the dataset
    df = pd.read_csv('creditcard.csv')

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

    st.subheader("Model Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.write("\nClassification Report:\n", classification_report_result)
    st.write("\nConfusion Matrix:\n", confusion_matrix_result)

    # Step 8: Prediction Function for User Input
    def predict_fraud(user_input):
        user_input['trans_date_trans_time'] = datetime.strptime(user_input['trans_date_trans_time'], "%Y-%m-%d %H:%M:%S").timestamp()
        user_input['trans_num'] = le.transform([user_input['trans_num']])
        user_data = pd.DataFrame([user_input])
        # Step 9: Load the saved model for prediction
        loaded_model = joblib.load('fraud_detection_model.pkl')
        prediction = loaded_model.predict(user_data)
        return prediction[0]

    # Step 10: User Input and Prediction
    st.subheader("Fraud Prediction")
    user_input = st.text_input("Transaction Date and Time (YYYY-MM-DD HH:MM:SS)", "2019-01-01 12:00:00")
    trans_num = st.text_input("Transaction Number", "0b242abb623afc578575680df30655b9")
    user_input = {
        'trans_date_trans_time': user_input,
        'trans_num': trans_num
    }

    if st.button("Predict"):
        fraud_prediction = predict_fraud(user_input)
        if fraud_prediction == 0:
            st.write("The transaction is predicted as not fraud.")
        else:
            st.write("TThe transaction is predicted as fraud.")


    

if __name__ == "__main__":
    main()
