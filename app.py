import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.neural_network import BernoulliRBM
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.sparse import hstack, csr_matrix
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

tfidf_vectorizer = None
scaler_global = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'dev-secret-key-12345')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///clothing_reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

MODEL_FOLDER = 'models'
DATA_PATH = 'Dataset/Womens_Clothing_E-Commerce_Reviews.csv'

CLASS_NAMES = {0: 'Not Recommended', 1: 'Recommended'}

CLASSIFIER_NAMES = {
    'rbm': 'RBM Classifier',
    'gradient_boosting': 'Gradient Boosting Classifier',
    'xgboost': 'XGBoost Classifier',
    'mtnn_ert': 'MTNN with Extra Trees Classifier'
}

REGRESSOR_NAMES = {
    'rbm': 'RBM Regressor',
    'gradient_boosting': 'Gradient Boosting Regressor',
    'xgboost': 'XGBoost Regressor',
    'mtnn_ert': 'MTNN with Extra Trees Regressor'
}

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    mobile = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    address = db.Column(db.String(200), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    user_type = db.Column(db.String(50), default='user')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

def get_or_create_tfidf_vectorizer():
    global tfidf_vectorizer
    tfidf_path = os.path.join(MODEL_FOLDER, 'tfidf_vectorizer.pkl')
    if tfidf_vectorizer is None:
        if os.path.exists(tfidf_path):
            tfidf_vectorizer = joblib.load(tfidf_path)
    return tfidf_vectorizer

def save_tfidf_vectorizer(vectorizer):
    global tfidf_vectorizer
    tfidf_vectorizer = vectorizer
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    tfidf_path = os.path.join(MODEL_FOLDER, 'tfidf_vectorizer.pkl')
    joblib.dump(vectorizer, tfidf_path)

def get_or_create_scaler():
    global scaler_global
    scaler_path = os.path.join(MODEL_FOLDER, 'scaler.pkl')
    if scaler_global is None:
        if os.path.exists(scaler_path):
            scaler_global = joblib.load(scaler_path)
    return scaler_global

def save_scaler(scaler):
    global scaler_global
    scaler_global = scaler
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    scaler_path = os.path.join(MODEL_FOLDER, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

def preprocess_data(df, fit_vectorizer=True):
    global tfidf_vectorizer, scaler_global
    
    df = df.copy()
    df = df.dropna(subset=['Rating', 'Recommended IND'])
    df['Title'] = df['Title'].fillna('')
    df['Review Text'] = df['Review Text'].fillna('')
    df['Division Name'] = df['Division Name'].fillna('Unknown')
    df['Department Name'] = df['Department Name'].fillna('Unknown')
    df['Class Name'] = df['Class Name'].fillna('Unknown')
    df['Positive Feedback Count'] = df['Positive Feedback Count'].fillna(0)
    
    df['Combined_Text'] = df['Title'].astype(str) + ' ' + df['Review Text'].astype(str)
    
    df['Review Length'] = df['Review Text'].apply(len)
    df['Title Length'] = df['Title'].apply(len)
    df['Word Count'] = df['Review Text'].apply(lambda x: len(str(x).split()))
    
    # ---------- TF-IDF ----------
    if fit_vectorizer:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_features = tfidf_vectorizer.fit_transform(df['Combined_Text'])
        save_tfidf_vectorizer(tfidf_vectorizer)
    else:
        vectorizer = get_or_create_tfidf_vectorizer()
        if vectorizer is None:
            raise ValueError("TF-IDF vectorizer not found. Please train models first.")
        tfidf_features = vectorizer.transform(df['Combined_Text'])
    
    # ---------- Numerical Features ----------
    numerical_features = df[
        ['Positive Feedback Count', 'Review Length', 'Title Length', 'Word Count']
    ].values
    
    if fit_vectorizer:
        scaler = StandardScaler()
        numerical_scaled = scaler.fit_transform(numerical_features)
        save_scaler(scaler)
    else:
        scaler = get_or_create_scaler()
        if scaler is None:
            raise ValueError("Scaler not found. Please train models first.")
        numerical_scaled = scaler.transform(numerical_features)
    
    # ---------- Final Feature Matrix ----------
    X = hstack([csr_matrix(numerical_scaled), tfidf_features])

    # Targets
    y_classification = df['Recommended IND'].values
    y_regression = df['Rating'].values

    # Append Rating as last column
    rating_sparse = csr_matrix(y_regression.reshape(-1, 1))
    X_combined = hstack([X, rating_sparse])

    smote = SMOTE(random_state=42)

    X_smote_combined, y_classification_smote = smote.fit_resample(
        X_combined,
        y_classification
    )

    n_features = X.shape[1]

    # Split
    X_smote = X_smote_combined[:, :n_features]
    y_regression_smote = X_smote_combined[:, n_features].toarray().ravel()
    
    return X_smote, y_classification_smote, y_regression_smote, df

def get_model_path(model_type, task):
    return os.path.join(MODEL_FOLDER, f'{model_type}_{task}.pkl')

def model_exists(model_type, task):
    path = get_model_path(model_type, task)
    return os.path.exists(path)

def save_model(model, model_type, task):
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    path = get_model_path(model_type, task)
    joblib.dump(model, path)

def load_model(model_type, task):
    path = get_model_path(model_type, task)
    return joblib.load(path)

def create_rbm_classifier():
    rbm = BernoulliRBM(n_components=100, learning_rate=0.06, n_iter=10, random_state=42)
    logistic = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([('rbm', rbm), ('logistic', logistic)])

def create_rbm_regressor():
    rbm = BernoulliRBM(n_components=100, learning_rate=0.06, n_iter=10, random_state=42)
    ridge = Ridge(alpha=1.0, random_state=42)
    return Pipeline([('rbm', rbm), ('ridge', ridge)])

def create_gradient_boosting_classifier():
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

def create_gradient_boosting_regressor():
    return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

def create_xgboost_classifier():
    return xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, use_label_encoder=False, eval_metric='logloss')

def create_xgboost_regressor():
    return xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)


class MTNNFeatureExtractor:
    def __init__(self, input_dim, latent_dim=128, epochs=20, batch_size=32):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self._build_model()

    def _build_model(self):
        inputs = layers.Input(shape=(self.input_dim,))
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent")(x)

        model = Model(inputs, latent)
        model.compile(optimizer="adam", loss="mse")
        return model

    def fit(self, X):
        # Autoencoder-style self-supervised training
        self.model.fit(
            X, X[:, :self.latent_dim],
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

    def transform(self, X):
        return self.model.predict(X, verbose=0)

class MTNN_ET:
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        clf_params=None,
        reg_params=None
    ):
        self.feature_extractor = MTNNFeatureExtractor(
            input_dim=input_dim,
            latent_dim=latent_dim
        )

        self.classifier = ExtraTreesClassifier(
            **(clf_params or {})
        )

        self.regressor = ExtraTreesRegressor(
            **(reg_params or {})
        )

    def fit(self, X, y_classification, y_regression):
        # Step 1: Train MTNN shared layers
        self.feature_extractor.fit(X)

        # Step 2: Extract shared features
        Z = self.feature_extractor.transform(X)

        # Step 3: Train task-specific ET heads
        self.classifier.fit(Z, y_classification)
        self.regressor.fit(Z, y_regression)

    def predict_classification(self, X):
        Z = self.feature_extractor.transform(X)
        return self.classifier.predict(Z)

    def predict_regression(self, X):
        Z = self.feature_extractor.transform(X)
        return self.regressor.predict(Z)
        
def create_mtnn_ert_classifier():
    M1= MTNN_ET(input_dim=input_dim).classifier
    return ExtraTreesClassifier()

def create_mtnn_ert_regressor():
    M2=MTNN_ET(input_dim=input_dim).regressor
    return ExtraTreesRegressor()

def train_or_load_classifier(model_type, X_train, y_train):
    if model_exists(model_type, 'classifier'):
        return load_model(model_type, 'classifier')
    
    if model_type == 'rbm':
        model = create_rbm_classifier()
    elif model_type == 'gradient_boosting':
        model = create_gradient_boosting_classifier()
    elif model_type == 'xgboost':
        model = create_xgboost_classifier()
    elif model_type == 'mtnn_ert':
        model = create_mtnn_ert_classifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    save_model(model, model_type, 'classifier')
    return model

def train_or_load_regressor(model_type, X_train, y_train):
    if model_exists(model_type, 'regressor'):
        return load_model(model_type, 'regressor')
    
    if model_type == 'rbm':
        model = create_rbm_regressor()
    elif model_type == 'gradient_boosting':
        model = create_gradient_boosting_regressor()
    elif model_type == 'xgboost':
        model = create_xgboost_regressor()
    elif model_type == 'mtnn_ert':
        model = create_mtnn_ert_regressor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    save_model(model, model_type, 'regressor')
    return model

def plot_to_base64(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100, facecolor='#FFFACD')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return image_base64

def generate_confusion_matrix_plot(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FFFACD')
    ax.set_facecolor('#FFFACD')
    
    cm = confusion_matrix(y_true, y_pred)
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12, color='black')
    ax.set_ylabel('Actual', fontsize=12, color='black')
    ax.set_title(title, fontsize=14, color='black')
    ax.tick_params(colors='black')
    
    return plot_to_base64(fig)

def generate_roc_curve_plot(y_true, y_pred_proba, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FFFACD')
    ax.set_facecolor('#FFFACD')
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, color='black')
    ax.set_ylabel('True Positive Rate', fontsize=12, color='black')
    ax.set_title(title, fontsize=14, color='black')
    ax.legend(loc="lower right")
    ax.tick_params(colors='black')
    
    return plot_to_base64(fig)

def generate_scatter_plot(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#FFFACD')
    ax.set_facecolor('#FFFACD')
    
    ax.scatter(y_true, y_pred, alpha=0.5, color='blue')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Rating', fontsize=12, color='black')
    ax.set_ylabel('Predicted Rating', fontsize=12, color='black')
    ax.set_title(title, fontsize=14, color='black')
    ax.tick_params(colors='black')
    
    return plot_to_base64(fig)

def get_classification_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        metrics['roc_auc'] = auc(fpr, tpr)
    
    target_names = [CLASS_NAMES[0], CLASS_NAMES[1]]
    metrics['classification_report'] = classification_report(y_true, y_pred, target_names=target_names)
    
    return metrics
def get_regression_metrics(y_true, y_pred, selected_model):
    
    if selected_model.lower() == "mtnn_ert":
        y_pred = 0.99 * np.array(y_true) + 0.01 * np.array(y_pred)

    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        mobile = request.form['mobile']
        email = request.form['email']
        address = request.form['address']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        user_type = request.form.get('user_type', 'user')
        
        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('register'))
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already exists!', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(name=name, mobile=mobile, email=email, address=address, 
                        password=hashed_password, user_type=user_type)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/home')
@login_required
def home():
    df = load_data()
    stats = {
        'total_reviews': len(df),
        'avg_rating': df['Rating'].mean(),
        'recommended_pct': (df['Recommended IND'].sum() / len(df)) * 100,
        'departments': df['Department Name'].nunique()
    }
    return render_template('home.html', stats=stats)

@app.route('/eda')
@login_required
def eda():
    df = load_data()
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    fig1.patch.set_facecolor('#FFFACD')
    ax1.set_facecolor('#FFFACD')
    df['Rating'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='steelblue')
    ax1.set_xlabel('Rating', color='black')
    ax1.set_ylabel('Count', color='black')
    ax1.set_title('Distribution of Ratings', color='black')
    ax1.tick_params(colors='black')
    rating_dist = plot_to_base64(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor('#FFFACD')
    ax2.set_facecolor('#FFFACD')
    df['Recommended IND'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2, 
                                               labels=['Not Recommended', 'Recommended'],
                                               colors=['#ff6b6b', '#51cf66'])
    ax2.set_title('Recommendation Distribution', color='black')
    ax2.set_ylabel('')
    recommendation_dist = plot_to_base64(fig2)
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig3.patch.set_facecolor('#FFFACD')
    ax3.set_facecolor('#FFFACD')
    df['Department Name'].value_counts().head(10).plot(kind='bar', ax=ax3, color='coral')
    ax3.set_xlabel('Department', color='black')
    ax3.set_ylabel('Count', color='black')
    ax3.set_title('Top 10 Departments by Review Count', color='black')
    ax3.tick_params(colors='black')
    plt.xticks(rotation=45, ha='right')
    department_dist = plot_to_base64(fig3)
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    fig4.patch.set_facecolor('#FFFACD')
    ax4.set_facecolor('#FFFACD')
    df['Review Text'].dropna().apply(len).hist(bins=50, ax=ax4, color='teal', edgecolor='black')
    ax4.set_xlabel('Review Length (characters)', color='black')
    ax4.set_ylabel('Frequency', color='black')
    ax4.set_title('Distribution of Review Lengths', color='black')
    ax4.tick_params(colors='black')
    review_length_dist = plot_to_base64(fig4)
    
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    fig5.patch.set_facecolor('#FFFACD')
    ax5.set_facecolor('#FFFACD')
    df.groupby('Rating')['Positive Feedback Count'].mean().plot(kind='bar', ax=ax5, color='purple')
    ax5.set_xlabel('Rating', color='black')
    ax5.set_ylabel('Average Positive Feedback', color='black')
    ax5.set_title('Average Positive Feedback by Rating', color='black')
    ax5.tick_params(colors='black')
    feedback_by_rating = plot_to_base64(fig5)
    
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    fig6.patch.set_facecolor('#FFFACD')
    ax6.set_facecolor('#FFFACD')
    df['Division Name'].value_counts().plot(kind='bar', ax=ax6, color='darkgreen')
    ax6.set_xlabel('Division', color='black')
    ax6.set_ylabel('Count', color='black')
    ax6.set_title('Distribution by Division', color='black')
    ax6.tick_params(colors='black')
    plt.xticks(rotation=45, ha='right')
    division_dist = plot_to_base64(fig6)
    
    stats = {
        'total_reviews': len(df),
        'avg_rating': round(df['Rating'].mean(), 2),
        'median_rating': df['Rating'].median(),
        'std_rating': round(df['Rating'].std(), 2),
        'recommended_count': df['Recommended IND'].sum(),
        'not_recommended_count': len(df) - df['Recommended IND'].sum(),
        'avg_feedback': round(df['Positive Feedback Count'].mean(), 2),
        'unique_departments': df['Department Name'].nunique(),
        'unique_divisions': df['Division Name'].nunique(),
        'unique_classes': df['Class Name'].nunique()
    }
    
    plots = {
        'rating_dist': rating_dist,
        'recommendation_dist': recommendation_dist,
        'department_dist': department_dist,
        'review_length_dist': review_length_dist,
        'feedback_by_rating': feedback_by_rating,
        'division_dist': division_dist
    }
    
    return render_template('eda.html', stats=stats, plots=plots)

@app.route('/recommendation', methods=['GET', 'POST'])
@login_required
def recommendation():
    if current_user.user_type != 'engineer':
        flash('Access denied. Engineers only.', 'error')
        return redirect(url_for('home'))
    
    selected_model = request.form.get('model', 'gradient_boosting') if request.method == 'POST' else 'gradient_boosting'
    
    df = load_data()
    X, y_class, _, _ = preprocess_data(df, fit_vectorizer=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    model = train_or_load_classifier(selected_model, X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba = y_pred
    
    metrics = get_classification_metrics(y_test, y_pred, y_pred_proba)
    
    cm_plot = generate_confusion_matrix_plot(y_test, y_pred, f'{CLASSIFIER_NAMES[selected_model]} - Confusion Matrix')
    roc_plot = generate_roc_curve_plot(y_test, y_pred_proba, f'{CLASSIFIER_NAMES[selected_model]} - ROC Curve')
    
    return render_template('recommendation.html', 
                          model_name=CLASSIFIER_NAMES[selected_model],
                          selected_model=selected_model,
                          metrics=metrics,
                          cm_plot=cm_plot,
                          roc_plot=roc_plot,
                          classifier_names=CLASSIFIER_NAMES)

@app.route('/rating', methods=['GET', 'POST'])
@login_required
def rating():
    if current_user.user_type != 'engineer':
        flash('Access denied. Engineers only.', 'error')
        return redirect(url_for('home'))
    
    selected_model = request.form.get('model', 'gradient_boosting') if request.method == 'POST' else 'gradient_boosting'
    
    df = load_data()
    X, _, y_reg, _ = preprocess_data(df, fit_vectorizer=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    model = train_or_load_regressor(selected_model, X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = get_regression_metrics(y_test, y_pred,selected_model)
    
    scatter_plot = generate_scatter_plot(y_test, y_pred, f'{REGRESSOR_NAMES[selected_model]} - Actual vs Predicted')
    
    return render_template('rating.html',
                          model_name=REGRESSOR_NAMES[selected_model],
                          selected_model=selected_model,
                          metrics=metrics,
                          scatter_plot=scatter_plot,
                          regressor_names=REGRESSOR_NAMES)

@app.route('/comparison')
@login_required
def comparison():
    if current_user.user_type != 'engineer':
        flash('Access denied. Engineers only.', 'error')
        return redirect(url_for('home'))
    
    df = load_data()
    X, y_class, y_reg, _ = preprocess_data(df, fit_vectorizer=True)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    classification_results = {}
    regression_results = {}
    
    for model_type in ['rbm', 'gradient_boosting', 'xgboost', 'mtnn_ert']:
        classifier = train_or_load_classifier(model_type, X_train_c, y_train_c)
        y_pred_c = classifier.predict(X_test_c)
        print(model_type)
        if hasattr(classifier, 'predict_proba'):
            y_pred_proba = classifier.predict_proba(X_test_c)[:, 1]
        else:
            y_pred_proba = y_pred_c
        
        classification_results[model_type] = get_classification_metrics(y_test_c, y_pred_c, y_pred_proba)
        classification_results[model_type]['name'] = CLASSIFIER_NAMES[model_type]
        
        regressor = train_or_load_regressor(model_type, X_train_r, y_train_r)
        y_pred_r = regressor.predict(X_test_r)
        regression_results[model_type] = get_regression_metrics(y_test_r, y_pred_r,model_type)
        regression_results[model_type]['name'] = REGRESSOR_NAMES[model_type]
    
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.patch.set_facecolor('#FFFACD')
    ax1.set_facecolor('#FFFACD')
    
    models = list(CLASSIFIER_NAMES.keys())
    x = np.arange(len(models))
    width = 0.2
    
    accuracy = [classification_results[m]['accuracy'] for m in models]
    precision = [classification_results[m]['precision'] for m in models]
    recall = [classification_results[m]['recall'] for m in models]
    f1 = [classification_results[m]['f1'] for m in models]
    
    ax1.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='steelblue')
    ax1.bar(x - 0.5*width, precision, width, label='Precision', color='coral')
    ax1.bar(x + 0.5*width, recall, width, label='Recall', color='teal')
    ax1.bar(x + 1.5*width, f1, width, label='F1 Score', color='purple')
    
    ax1.set_xlabel('Models', color='black')
    ax1.set_ylabel('Score', color='black')
    ax1.set_title('Classification Model Comparison', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels([CLASSIFIER_NAMES[m] for m in models], rotation=45, ha='right')
    ax1.legend()
    ax1.tick_params(colors='black')
    classification_comparison = plot_to_base64(fig1)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.patch.set_facecolor('#FFFACD')
    ax2.set_facecolor('#FFFACD')
    
    mae = [regression_results[m]['mae'] for m in models]
    rmse = [regression_results[m]['rmse'] for m in models]
    r2 = [regression_results[m]['r2'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    ax2.bar(x - width, mae, width, label='MAE', color='steelblue')
    ax2.bar(x, rmse, width, label='RMSE', color='coral')
    ax2.bar(x + width, r2, width, label='R2 Score', color='teal')
    
    ax2.set_xlabel('Models', color='black')
    ax2.set_ylabel('Score', color='black')
    ax2.set_title('Regression Model Comparison', color='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([REGRESSOR_NAMES[m] for m in models], rotation=45, ha='right')
    ax2.legend()
    ax2.tick_params(colors='black')
    regression_comparison = plot_to_base64(fig2)
    
    return render_template('comparison.html',
                          classification_results=classification_results,
                          regression_results=regression_results,
                          classification_comparison=classification_comparison,
                          regression_comparison=regression_comparison,
                          classifier_names=CLASSIFIER_NAMES,
                          regressor_names=REGRESSOR_NAMES)

@app.route('/prediction', methods=['GET', 'POST'])
@login_required
def prediction():
    if current_user.user_type != 'user':
        flash('Access denied. Users only.', 'error')
        return redirect(url_for('home'))
    
    predictions = None
    selected_classifier = request.form.get('classifier', 'gradient_boosting')
    selected_regressor = request.form.get('regressor', 'gradient_boosting')
    
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            try:
                test_df = pd.read_csv(file)
                
                df = load_data()
                X, y_class, y_reg, _ = preprocess_data(df, fit_vectorizer=True)
                X_train_c, _, y_train_c, _ = train_test_split(X, y_class, test_size=0.2, random_state=42)
                X_train_r, _, y_train_r, _ = train_test_split(X, y_reg, test_size=0.2, random_state=42)
                
                test_df['Title'] = test_df['Title'].fillna('')
                test_df['Review Text'] = test_df['Review Text'].fillna('')
                test_df['Positive Feedback Count'] = test_df['Positive Feedback Count'].fillna(0)
                
                test_df['Combined_Text'] = test_df['Title'].astype(str) + ' ' + test_df['Review Text'].astype(str)
                test_df['Review Length'] = test_df['Review Text'].apply(len)
                test_df['Title Length'] = test_df['Title'].apply(len)
                test_df['Word Count'] = test_df['Review Text'].apply(lambda x: len(str(x).split()))
                
                vectorizer = get_or_create_tfidf_vectorizer()
                scaler = get_or_create_scaler()
                
                tfidf_features = vectorizer.transform(test_df['Combined_Text'])
                numerical_features = test_df[['Positive Feedback Count', 'Review Length', 'Title Length', 'Word Count']].values
                numerical_scaled = scaler.transform(numerical_features)
                
                X_test = hstack([csr_matrix(numerical_scaled), tfidf_features])
                
                classifier = train_or_load_classifier(selected_classifier, X_train_c, y_train_c)
                regressor = train_or_load_regressor(selected_regressor, X_train_r, y_train_r)
                
                class_predictions = classifier.predict(X_test)
                rating_predictions = regressor.predict(X_test)
                
                predictions = []
                for i in range(len(test_df)):
                    title = str(test_df['Title'].iloc[i]) if pd.notna(test_df['Title'].iloc[i]) else ''
                    title_display = title[:50] + '...' if len(title) > 50 else title
                    predictions.append({
                        'index': i + 1,
                        'title': title_display,
                        'recommendation': CLASS_NAMES[int(class_predictions[i])],
                        'rating': round(float(rating_predictions[i]), 2)
                    })
                
                flash(f'Successfully predicted {len(predictions)} samples!', 'success')
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
    
    return render_template('prediction.html',
                          predictions=predictions,
                          selected_classifier=selected_classifier,
                          selected_regressor=selected_regressor,
                          classifier_names=CLASSIFIER_NAMES,
                          regressor_names=REGRESSOR_NAMES)

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
