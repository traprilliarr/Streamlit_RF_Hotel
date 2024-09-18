import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the saved model
with open("model.sav", "rb") as file:
    model = pickle.load(file)

# Create LabelEncoders
label_encoder_type_of_meal_plan = LabelEncoder()
label_encoder_room_type_reserved = LabelEncoder()
label_encoder_market_segment_type = LabelEncoder()
label_encoder_booking_status = LabelEncoder()

# Preprocess function
def preprocess_data(df):
    # Label encoding for categorical variables
    df['type_of_meal_plan'] = label_encoder_type_of_meal_plan.transform(df['type_of_meal_plan'])
    df['room_type_reserved'] = label_encoder_room_type_reserved.transform(df['room_type_reserved'])
    df['market_segment_type'] = label_encoder_market_segment_type.transform(df['market_segment_type'])
    return df

# Function to get metrics
def get_metrics(y_true, y_pred, y_pred_prob):
    # Transform y_true to numeric values (0 and 1)
    y_true_numeric = label_encoder_booking_status.transform(y_true)

    # Transform y_pred to numeric values (0 and 1) using label_encoder
    y_pred_numeric = label_encoder_booking_status.transform(y_pred)

    acc = accuracy_score(y_true_numeric, y_pred_numeric)
    prec = precision_score(y_true_numeric, y_pred_numeric)
    recall = recall_score(y_true_numeric, y_pred_numeric)
    f1 = f1_score(y_true_numeric, y_pred_numeric)
    cm = confusion_matrix(y_true_numeric, y_pred_numeric)

    # Transform the results back to string format if needed
    y_true_str = label_encoder_booking_status.inverse_transform(y_true_numeric)
    y_pred_str = label_encoder_booking_status.inverse_transform(y_pred_numeric)

    return {
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1, 2),
        'confusion_matrix': cm
    }

# Streamlit App
def main():
    # Add HTML content
    st.markdown("<h1 style='text-align: center; color: #ff5733;'>Montelo - 09021282025091</h1>", unsafe_allow_html=True)

    st.title("Hotel Booking Classification")

    # Upload CSV data
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Fit LabelEncoders
        label_encoder_type_of_meal_plan.fit(df['type_of_meal_plan'])
        label_encoder_room_type_reserved.fit(df['room_type_reserved'])
        label_encoder_market_segment_type.fit(df['market_segment_type'])
        label_encoder_booking_status.fit(df['booking_status'])

        # Preprocess the uploaded data
        df = preprocess_data(df)

        # Display the uploaded data
        st.subheader("Uploaded Data:")
        st.write(df)

        # Dropdown option untuk memilih split data
        split_options = ["80:20", "70:30", "60:40", "90:10"]
        split_option = st.selectbox("Choose Test Data Split Option:", split_options)

        # Slider untuk memilih jumlah pohon (n_estimators)
        n_estimators = st.slider("Select Number of Trees (n_estimators):", 1, 1000, 100)

        # Pilih proporsi split berdasarkan opsi yang dipilih
        if split_option == "80:20":
            test_size_percentage = 0.2
        elif split_option == "70:30":
            test_size_percentage = 0.3
        elif split_option == "60:40":
            test_size_percentage = 0.4
        elif split_option == "90:10":
            test_size_percentage = 0.1

        # Split the data
        X = df[['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'type_of_meal_plan',
                'required_car_parking_space', 'room_type_reserved', 'lead_time', 'arrival_month', 'market_segment_type',
                'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                'avg_price_per_room', 'no_of_special_requests']]
        y_true = df['booking_status']  # Y_true harus berupa string

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=test_size_percentage, random_state=0)

        # Train the model with the selected number of trees
        model = RandomForestClassifier(n_estimators=n_estimators)  # Inisialisasi model RandomForest
        model.fit(X_train, y_train)

        # Predict using the trained model
        predictions = model.predict(X_test)
        prediction_probabilities = model.predict_proba(X_test)

        # Display classification metrics in a table
        st.subheader("Classification Metrics:")
        metrics = get_metrics(y_test, predictions, prediction_probabilities)
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        st.table(metrics_df)

        # Display confusion matrix
        st.subheader("Confusion Matrix:")
        st.write("True Positives:", metrics['confusion_matrix'][1, 1])
        st.write("True Negatives:", metrics['confusion_matrix'][0, 0])
        st.write("False Positives:", metrics['confusion_matrix'][0, 1])
        st.write("False Negatives:", metrics['confusion_matrix'][1, 0])

if __name__ == "__main__":
    main()
