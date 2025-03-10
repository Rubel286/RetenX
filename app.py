import os
import pandas as pd
import numpy as np
import joblib
import time
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import plotly.express as px
import plotly.io as pio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")  # Use env var with fallback
if os.getenv("RENDER", "false").lower() == "true":
    UPLOAD_FOLDER = '/opt/render/project/src/uploads'
else:
    UPLOAD_FOLDER = 'uploads'  # Local fallback
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MAX_FILES = 10
uploaded_files = glob.glob(os.path.join(UPLOAD_FOLDER, "*.csv"))
if len(uploaded_files) > MAX_FILES:
    for file in uploaded_files[:-MAX_FILES]:
        os.remove(file)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for model pipelines
model_cache = {}

# Preload IBM HR dataset globally with updated path
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'datasets', 'IBM-HR-Analytics-Employee-Attrition-and-Performance.csv')
try:
    df = pd.read_csv(DATASET_PATH)
    df['AttritionNumeric'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    logger.info(f"Loaded IBM HR dataset with {len(df)} rows. Median MonthlyIncome: {df['MonthlyIncome'].median()}")
except FileNotFoundError:
    logger.error(f"IBM HR dataset not found at {DATASET_PATH}. Using fallback median income.")
    df = pd.DataFrame()  # Empty DataFrame as fallback

# Load models on demand
def get_model(model_name):
    if model_name not in model_cache:
        MODEL_PATHS = {
            'rf': 'models/rf_pipeline.pkl',
            'knn': 'models/knn_pipeline.pkl',
            'lr': 'models/lr_pipeline.pkl',
            'svm': 'models/svm_pipeline.pkl',
            'xgb': 'models/xgb_pipeline.pkl'
        }
        model_cache[model_name] = joblib.load(MODEL_PATHS[model_name])
    return model_cache[model_name]

FEATURES = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'OverTime', 'YearsAtCompany', 
            'WorkLifeBalance', 'JobLevel', 'DistanceFromHome']

# 1. Time Window Analysis Feature
def predict_time_window(employee_data):
    try:
        monthly_income = float(employee_data.get('MonthlyIncome', 0))
        job_satisfaction = int(employee_data.get('JobSatisfaction', 0))
        years_at_company = int(employee_data.get('YearsAtCompany', 0))
        overtime = employee_data.get('OverTime', 'No')

        logger.info(f"Input data: MonthlyIncome={monthly_income}, JobSatisfaction={job_satisfaction}, "
                   f"YearsAtCompany={years_at_company}, OverTime={overtime}")

        base_risk = 0.0
        risk_factors = []

        median_income = df['MonthlyIncome'].median() if not df.empty else 6500.0
        logger.info(f"Median income: {median_income}")

        if monthly_income < median_income:
            base_risk += 0.3
            risk_factors.append(f"Below median salary (Current: ${monthly_income}, Median: ${median_income:.2f})")
            logger.info("Triggered: Below median salary")

        if job_satisfaction < 3:
            base_risk += 0.25
            risk_factors.append(f"Low job satisfaction (Score: {job_satisfaction}/4)")
            logger.info("Triggered: Low job satisfaction")

        if overtime == 'Yes':
            base_risk += 0.2
            risk_factors.append("Regular overtime work")
            logger.info("Triggered: Regular overtime")

        if years_at_company < 2:
            base_risk += 0.25
            risk_factors.append(f"Short tenure ({years_at_company} years)")
            logger.info("Triggered: Short tenure")

        logger.info(f"Base risk calculated: {base_risk}, Factors: {risk_factors}")

        predictions = {
            '30_days': {
                'risk_score': min(base_risk * 1.2, 1.0),
                'contributing_factors': risk_factors,
                'impact': 'Immediate intervention needed' if base_risk * 1.2 > 0.7 else 'Monitor situation'
            },
            '60_days': {
                'risk_score': min(base_risk * 1.1, 1.0),
                'contributing_factors': risk_factors,
                'impact': 'Plan intervention' if base_risk * 1.1 > 0.7 else 'Regular check-ins'
            },
            '90_days': {
                'risk_score': base_risk,
                'contributing_factors': risk_factors,
                'impact': 'Strategic planning needed' if base_risk > 0.7 else 'Standard monitoring'
            }
        }
        logger.info(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        logger.error(f"Error in time window prediction: {str(e)}")
        return None

# 2. Retention Strategy Feature
def generate_retention_strategy(employee_data):
    try:
        strategies = {
            'immediate_actions': [],
            'medium_term_actions': [],
            'long_term_actions': [],
            'priority_level': 'medium'
        }

        monthly_income = float(employee_data.get('MonthlyIncome', 0))
        median_income = df['MonthlyIncome'].median() if not df.empty else 6500.0
        if monthly_income < median_income:
            strategies['immediate_actions'].append({
                'action': 'Review compensation package',
                'reason': 'Below median salary for role',
                'impact': 'High',
                'timeframe': '30 days'
            })

        job_satisfaction = int(employee_data.get('JobSatisfaction', 0))
        if job_satisfaction < 3:
            strategies['immediate_actions'].append({
                'action': 'Schedule one-on-one meeting',
                'reason': 'Low job satisfaction score',
                'impact': 'High',
                'timeframe': '7 days'
            })
            strategies['medium_term_actions'].append({
                'action': 'Create development plan',
                'reason': 'Career growth opportunity',
                'impact': 'Medium',
                'timeframe': '60 days'
            })

        if employee_data.get('OverTime') == 'Yes':
            strategies['immediate_actions'].append({
                'action': 'Review workload distribution',
                'reason': 'Regular overtime indicates potential burnout risk',
                'impact': 'High',
                'timeframe': '14 days'
            })

        years_at_company = int(employee_data.get('YearsAtCompany', 0))
        if years_at_company < 2:
            strategies['medium_term_actions'].append({
                'action': 'Assign mentor',
                'reason': 'New employee retention',
                'impact': 'Medium',
                'timeframe': '30 days'
            })
        elif years_at_company > 5:
            strategies['long_term_actions'].append({
                'action': 'Consider for leadership development',
                'reason': 'Experienced employee growth',
                'impact': 'Medium',
                'timeframe': '90 days'
            })

        if len(strategies['immediate_actions']) >= 2:
            strategies['priority_level'] = 'high'
        elif len(strategies['immediate_actions']) == 0:
            strategies['priority_level'] = 'low'

        return strategies
    except Exception as e:
        logger.error(f"Error generating retention strategy: {str(e)}")
        return None

# 3. Historical Trends Analysis Feature
def analyze_historical_trends(start_date=None, end_date=None, department=None):
    try:
        if df.empty:
            raise ValueError("IBM HR dataset is not available.")

        trends = {
            'department_trends': {},
            'overall_attrition_rate': float(df['AttritionNumeric'].mean()),
            'total_employees': len(df),
            'department_statistics': {}
        }

        dept_stats = df.groupby('Department').agg({
            'AttritionNumeric': ['count', 'mean'],
            'MonthlyIncome': 'mean',
            'YearsAtCompany': 'mean'
        }).round(2)

        for dept in dept_stats.index:
            trends['department_statistics'][dept] = {
                'employee_count': int(dept_stats.loc[dept, ('AttritionNumeric', 'count')]),
                'attrition_rate': float(dept_stats.loc[dept, ('AttritionNumeric', 'mean')]),
                'avg_salary': float(dept_stats.loc[dept, ('MonthlyIncome', 'mean')]),
                'avg_tenure': float(dept_stats.loc[dept, ('YearsAtCompany', 'mean')])
            }

        avg_attrition = df['AttritionNumeric'].mean()
        trends['high_risk_departments'] = [
            dept for dept in trends['department_statistics']
            if trends['department_statistics'][dept]['attrition_rate'] > avg_attrition
        ]

        role_stats = df.groupby('JobRole').agg({
            'AttritionNumeric': 'mean'
        }).sort_values('AttritionNumeric', ascending=False)

        trends['role_trends'] = {
            role: float(rate) for role, rate in 
            role_stats['AttritionNumeric'].items()
        }

        return trends
    except Exception as e:
        logger.error(f"Error analyzing historical trends: {str(e)}")
        return None

# Main landing page
@app.route('/')
def main():
    return render_template('main.html')

# Home Page (Single Prediction with Time Window)
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            input_data = {
                'Age': int(request.form['age']),
                'MonthlyIncome': float(request.form['monthly_income']),
                'JobSatisfaction': int(request.form['job_satisfaction']),
                'OverTime': request.form['overtime'],
                'YearsAtCompany': int(request.form['years_at_company']),
                'WorkLifeBalance': int(request.form['work_life_balance']),
                'JobLevel': int(request.form['job_level']),
                'DistanceFromHome': int(request.form['distance_from_home'])
            }
            input_df = pd.DataFrame([input_data])
            input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1})
            
            pipeline = get_model('rf')
            input_processed = pd.DataFrame(pipeline['imputer'].transform(input_df), columns=FEATURES)
            input_processed = pd.DataFrame(pipeline['scaler'].transform(input_processed), columns=FEATURES)
            prediction = pipeline['model'].predict(input_processed)[0]
            prob = pipeline['model'].predict_proba(input_processed)[0][1]
            risk_percentage = f"{prob*100:.1f}%"
            
            if prediction == 1:
                result = f"⚠️ Employee is expected to leave. Risk: {risk_percentage}"
            else:
                result = f"✅ Employee is expected to stay. Risk: {risk_percentage}"
                
            time_window_result = predict_time_window(input_data)
            if time_window_result is None:
                flash("Error calculating time window analysis.")
            return render_template('home.html', prediction=result, time_window=time_window_result)
        except ValueError as e:
            flash(f"Invalid input: {str(e)}")
    return render_template('home.html', prediction=None, time_window=None)

# Retention Strategy Page
@app.route('/retention_strategy', methods=['GET', 'POST'])
def retention_strategy():
    if request.method == 'POST':
        try:
            input_data = {
                'MonthlyIncome': float(request.form['monthly_income']),
                'JobSatisfaction': int(request.form['job_satisfaction']),
                'OverTime': request.form['overtime'],
                'YearsAtCompany': int(request.form['years_at_company'])
            }
            strategy = generate_retention_strategy(input_data)
            if strategy:
                return render_template('retention_strategy.html', strategy=strategy)
            else:
                flash("Error generating retention strategy.")
        except ValueError as e:
            flash(f"Invalid input: {str(e)}")
    return render_template('retention_strategy.html', strategy=None)

# Historical Trends Page
@app.route('/historical_trends', methods=['GET', 'POST'])
def historical_trends():
    if request.method == 'POST':
        trends = analyze_historical_trends()
        if trends:
            return render_template('historical_trends.html', trends=trends)
        else:
            flash("Error analyzing historical trends.")
    return render_template('historical_trends.html', trends=None)

# Analysis Page
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    graphs = []
    if request.method == 'POST':
        if 'dataset' in request.files and request.files['dataset'].filename:
            file = request.files['dataset']
            if file and file.filename.endswith('.csv'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                df_uploaded = pd.read_csv(filepath)
                if len(df_uploaded) > 5000:
                    df_uploaded = df_uploaded.sample(n=5000, random_state=42)
                
                fig1 = px.pie(df_uploaded, names='Attrition', title='Attrition Distribution',
                             color_discrete_sequence=px.colors.sequential.Blues_r)
                graphs.append(pio.to_html(fig1, full_html=False))
                
                fig2 = px.box(df_uploaded, x='Attrition', y='MonthlyIncome', title='Monthly Income vs Attrition',
                             color='Attrition', color_discrete_sequence=px.colors.sequential.Blues_r)
                graphs.append(pio.to_html(fig2, full_html=False))
                
                fig3 = px.violin(df_uploaded, x='Attrition', y='Age', title='Age vs Attrition',
                                color='Attrition', color_discrete_sequence=px.colors.sequential.Blues_r)
                graphs.append(pio.to_html(fig3, full_html=False))
                
                fig4 = px.histogram(df_uploaded, x='JobSatisfaction', color='Attrition', barmode='group', 
                                   title='Job Satisfaction Distribution by Attrition',
                                   color_discrete_sequence=px.colors.sequential.Blues_r)
                graphs.append(pio.to_html(fig4, full_html=False))
                
                fig5 = px.scatter(df_uploaded, x='YearsAtCompany', y='MonthlyIncome', color='Attrition', 
                                 title='Years at Company vs Monthly Income by Attrition',
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                graphs.append(pio.to_html(fig5, full_html=False))
            else:
                flash('Please upload a valid CSV file.')
        return render_template('analysis.html', graphs=graphs)
    return render_template('analysis.html', graphs=graphs)

# Batch Prediction Page
@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        file = request.files['batch_file']
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                df = pd.read_csv(filepath)
                if not all(feat in df.columns for feat in FEATURES):
                    flash('CSV must contain all required features: ' + ', '.join(FEATURES))
                    return redirect(url_for('batch'))
                
                truncated = len(df) > 100
                if truncated:
                    display_df = df.head(100).copy()
                else:
                    display_df = df.copy()
                
                input_df = df[FEATURES].copy()
                input_df['OverTime'] = input_df['OverTime'].map({'No': 0, 'Yes': 1, 'no': 0, 'yes': 1})
                
                pipeline = get_model('rf')
                input_processed = pd.DataFrame(pipeline['imputer'].transform(input_df), columns=FEATURES)
                input_processed = pd.DataFrame(pipeline['scaler'].transform(input_processed), columns=FEATURES)
                predictions = pipeline['model'].predict(input_processed)
                
                probs = pipeline['model'].predict_proba(input_processed)[:, 1]
                risk_levels = []
                for prob in probs:
                    if prob < 0.3:
                        risk_levels.append('Low')
                    elif prob < 0.7:
                        risk_levels.append('Medium')
                    else:
                        risk_levels.append('High')
                
                df['Risk'] = risk_levels
                df['Risk %'] = probs * 100
                df['Prediction'] = ['Leave' if pred == 1 else 'Stay' for pred in predictions]
                
                if truncated:
                    display_df['Risk'] = risk_levels[:100]
                    display_df['Risk %'] = (probs[:100] * 100).round(1)
                    display_df['Prediction'] = ['Leave' if pred == 1 else 'Stay' for pred in predictions[:100]]
                else:
                    display_df['Risk'] = risk_levels
                    display_df['Risk %'] = probs * 100
                    display_df['Prediction'] = ['Leave' if pred == 1 else 'Stay' for pred in predictions]
                
                def style_row(row):
                    if row['Prediction'] == 'Leave':
                        if row['Risk'] == 'High':
                            return ['background-color: #ff9999' for _ in row]
                        elif row['Risk'] == 'Medium':
                            return ['background-color: #ffcc99' for _ in row]
                        else:
                            return ['background-color: #ffeecc' for _ in row]
                    else:
                        return ['background-color: #99ff99' for _ in row]
                
                styled_df = display_df[FEATURES + ['Risk', 'Risk %', 'Prediction']].style.apply(style_row, axis=1).to_html()
                message = f"Analyzed {len(df)} employees." if not truncated else f"Showing first 100 of {len(df)} employees."
                return render_template('batch.html', table=styled_df, message=message)
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
        else:
            flash('Please upload a valid CSV file.')
    return render_template('batch.html', table=None)

# Comparison Page
@app.route('/comparison')
def comparison():
    try:
        logger.info("Attempting to process comparison with global dataset...")
        if df.empty:
            raise ValueError("IBM HR dataset is not available. Please ensure the dataset is in the 'datasets' directory.")

        X = df[FEATURES]
        y = df['Attrition'].map({'Yes': 1, 'No': 0})
        X.loc[:, 'OverTime'] = X['OverTime'].map({'Yes': 1, 'No': 0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_results = {}
        model_names = ['rf', 'knn', 'lr', 'svm', 'xgb']
        
        for name in model_names:
            logger.info(f"Processing model: {name}")
            pipeline = get_model(name)
            X_test_processed = pd.DataFrame(pipeline['imputer'].transform(X_test), columns=FEATURES)
            X_test_processed = pd.DataFrame(pipeline['scaler'].transform(X_test_processed), columns=FEATURES)
            X_train_processed = pd.DataFrame(pipeline['imputer'].transform(X_train), columns=FEATURES)
            X_train_processed = pd.DataFrame(pipeline['scaler'].transform(X_train_processed), columns=FEATURES)
            
            start_time = time.time()
            pipeline['model'].fit(X_train_processed, y_train)
            training_time = time.time() - start_time
            
            y_pred = pipeline['model'].predict(X_test_processed)
            model_results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'training_time': training_time
            }
            logger.info(f"Model {name} processed: {model_results[name]}")

        best_model = max(model_results, key=lambda x: model_results[x]['accuracy'])
        logger.info(f"Best model: {best_model}")
        return render_template('comparison.html', model_results=model_results, best_model=best_model)
    except Exception as e:
        logger.error(f"Error in comparison page: {str(e)}")
        flash(f"Error in comparison page: {str(e)}. Please ensure the dataset is available in the 'datasets' directory.")
        return redirect(url_for('main'))

@app.route('/healthz')
def health_check():
    return "OK", 200

if __name__ == '__main__':
    get_model('rf')  # Preload Random Forest model
<<<<<<< HEAD
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)
=======
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)), debug=False)
>>>>>>> f87c521 (updated app.py and additional files for render)
