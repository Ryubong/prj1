from flask import Flask, render_template, request, jsonify
import pandas as pd
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Function to process the data and make predictions
def process_data_and_predict(input_data):
    # Load data
    df = pd.read_csv('./social data.csv', encoding='cp949')

    # Data preprocessing
    test = df[df.isna()['사회']]
    train = df[df.notnull()['사회']]

    # Remove class 'D'
    train = train[train['사회'] != 'D']

    # Map the labels of the 'Society' column in the desired order
    label_mapping = {'C': 0, 'B': 1, 'B+': 2, 'A': 3, 'A+': 4}
    train['사회'] = train['사회'].map(label_mapping)

    X_train = train.drop(['사회'], axis=1)
    y_train = train['사회']

    features = ["신규채용", "이직 및 퇴직", "여성 근로자 (합)", "교육 시간", "사회 공헌 및 투자", "산업재해"]
    X_train = X_train[features]

    # Create and apply ADASYN object
    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    # Create a Gradient Boosting Classifier object
    gb_clf = GradientBoostingClassifier(random_state=42)

    # Model training
    gb_clf.fit(X_train_adasyn, y_train_adasyn)

    # Predict the grade with the entered feature values
    predicted_grade = gb_clf.predict(input_data)

    return predicted_grade

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input feature values from the form
        new_recruitment = float(request.form['New Recruitment'])
        resignation_retirement = float(request.form['resignation_retirement'])
        female_workers = float(request.form['female_workers'])
        training_hours = int(request.form['training_hours'])
        social_contribution = int(request.form['social_contribution'])
        industrial_accident = float(request.form['industrial_accident'])

        # Create a 2D array of the input feature values
        input_data = [[new_recruitment, resignation_retirement, female_workers, training_hours, social_contribution, industrial_accident]]

        # Process the data and make predictions
        predicted_grade = process_data_and_predict(input_data)

        # Map the predicted grade to the corresponding social grade label
        label_mapping_reverse = {0: 'C', 1: 'B', 2: 'B+', 3: 'A', 4: 'A+'}
        predicted_grade_label = label_mapping_reverse[predicted_grade[0]]

        return render_template('index.html', predicted_grade=predicted_grade_label)
    
    return render_template('index.html', predicted_grade='')

if __name__ == '__main__':
    app.run(debug=True)
