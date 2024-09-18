import pandas as pd
import numpy as np
import random
import math
import collections

# Fungsi untuk membaca file CSV
def read_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    header = lines[0].strip().split(',')
    data = [line.strip().split(',') for line in lines[1:]]
    df = pd.DataFrame(data, columns=header)
    return df

# Fungsi untuk menampilkan informasi dataframe
def display_info(df):
    print(df)

# Fungsi untuk menampilkan deskripsi statistik dataframe
def display_describe(df):
    print(df.describe().transpose())

# Fungsi untuk mengecek nilai missing/null
def check_missing(df):
    print(df.isnull().sum())

# Fungsi untuk menampilkan deskripsi data objek
def describe_object(df):
    print(df.describe(include=object).transpose())

# Fungsi untuk melakukan mapping pada kolom 'room_type_reserved'
def map_room_type_reserved(df):
    room_type_mapping = {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3, 'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7}
    df['room_type_reserved'] = df['room_type_reserved'].map(room_type_mapping)

# Fungsi untuk membuat bar chart dari nilai booking_status
def plot_booking_status(df):
    canceled_rate = df['booking_status'].value_counts(normalize=True) * 100

    plt.figure(figsize=(10, 6))
    bars = plt.barh(canceled_rate.index, canceled_rate.values, color=['green', 'red'])

    plt.xlabel('Percentage (%)')
    plt.ylabel('canceled_rate')
    plt.title('Canceled_rate')
    plt.gca().invert_yaxis()

    for bar in bars:
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}%', va='center')

    plt.show()

# Fungsi untuk memisahkan kolom numerik dan kategorikal
def separate_columns(df):
    num_cols = []
    cat_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        elif pd.api.types.is_object_dtype(df[col]):
            cat_cols.append(col)
    print('Numerical columns: ', num_cols)
    print('Categorical columns: ', cat_cols)

# Fungsi untuk menampilkan distribusi dan box plot dari kolom numerik
def plot_numeric_columns(df):
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col], kde=True, ax=axes[0])
        axes[0].set_title(f'Distribution of {col}')
        axes[0].grid(True)
        axes[1].set_xlabel('')
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f'Box plot of {col}')
        axes[1].grid(True)
        axes[1].set_xlabel('')
        plt.tight_layout()
        plt.show()

# Fungsi untuk melakukan uji chi-square pada variabel kategorikal
def chi_square_analysis(df):
    variables = df.columns[df.dtypes == 'object'].drop(['Booking_ID', 'booking_status'])
    results = []
    for variable in variables:
        contingency_table = pd.crosstab(df[variable], df['booking_status'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        results.append({'variable': variable, 'chi square': chi2, 'p-value': p_value})
    results_df = pd.DataFrame(results)
    print(results_df)

# Fungsi untuk menampilkan heatmap korelasi
def plot_correlation_heatmap(df):
    plt.figure(figsize=(20, 10))
    correlation = df.corr(numeric_only=True)
    ax = sns.heatmap(correlation, annot=True)
    plt.show()

# Fungsi untuk menampilkan countplot untuk beberapa variabel
def plot_countplots(df):
    plt.figure(figsize=(20, 25))

    plt.subplot(4, 2, 1)
    plt.gca().set_title('Variable no_of_adults')
    sns.countplot(x='no_of_adults', palette='Set2', data=df)

    # Tambahkan subplot lainnya sesuai kebutuhan

    plt.show()

# Fungsi untuk melakukan label encoding pada beberapa kolom
def label_encoding(df):
    label_encoder_type_of_meal_plan = LabelEncoder()
    label_encoder_room_type_reserved = LabelEncoder()
    label_encoder_market_segment_type = LabelEncoder()
    label_encoder_booking_status = LabelEncoder()

    df['type_of_meal_plan'] = label_encoder_type_of_meal_plan.fit_transform(df['type_of_meal_plan'])
    df['room_type_reserved'] = label_encoder_room_type_reserved.fit_transform(df['room_type_reserved'])
    df['market_segment_type'] = label_encoder_market_segment_type.fit_transform(df['market_segment_type'])
    df['booking_status'] = label_encoder_booking_status.fit_transform(df['booking_status'])

    le_name_mapping = dict(zip(label_encoder_booking_status.classes_, label_encoder_booking_status.transform(label_encoder_booking_status.classes_)))
    print(le_name_mapping)

# Fungsi untuk oversampling data agar kelas seimbang
def oversample_data(X, y):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    ax = sns.countplot(x=y_resampled)
    return X_resampled, y_resampled

# Fungsi untuk melakukan standar scaler pada data
def standard_scaler(X):
    scaler = StandardScaler()
    X_standard = scaler.fit_transform(X)
    return X_standard

# Fungsi untuk membagi data menjadi data latih dan data uji
def split_data(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, X_test, y_train, y_test

# Fungsi untuk melakukan grid search pada model Random Forest
def grid_search_rf(X_train, y_train):
    n_estimators = np.array([100])
    alg = ['entropy', 'gini']
    min_split = np.array([2, 3, 4, 5, 6, 7])
    max_nvl = np.array([3, 4, 5, 6, 7, 9, 11])
    values_grid = {'n_estimators': n_estimators, 'min_samples_split': min_split, 'max_depth': max_nvl, 'criterion': alg}

    model = RandomForestClassifier()
    gridRandomForest = GridSearchCV(estimator=model, param_grid=values_grid, cv=5)
    gridRandomForest.fit(X_train, y_train)

    print('Algorithm: ', gridRandomForest.best_estimator_.criterion)
    print('Score: ', gridRandomForest.best_score_)
    print('Mín Split: ', gridRandomForest.best_estimator_.min_samples_split)
    print('Max Nvl: ', gridRandomForest.best_estimator_.max_depth)

# Fungsi untuk melatih model Random Forest
def train_rf_model(X_train, y_train):
    random_forest = RandomForestClassifier(n_estimators=100, min_samples_split=2, max_depth=11, criterion='gini', random_state=0)
    random_forest.fit(X_train, y_train)
    return random_forest

# Fungsi untuk melakukan prediksi dan menampilkan metrik evaluasi
def evaluate_model(model, X_test, y_test):
    previsoes = model.predict(X_test)
    classification_random = (classification_report(y_test, previsoes))
    print(classification_random)

# Fungsi untuk mendapatkan metrik evaluasi
def get_metrics(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    entropy = log_loss(y_true, y_pred_prob)
    return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2), 'entropy': round(entropy, 2)}

# Fungsi untuk melakukan prediksi probabilitas pada data uji
def predict_prob_on_test_data(model, X_test):
    y_pred = model.predict_proba(X_test)
    return y_pred
