import os
import io
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web deployment
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import machine learning modules from custom package
from ml.config import CFG
from ml.preprocess import preprocess
from ml.selectors import traditional_selectors_report
from ml.ga import genetic_feature_selection_v2
from sklearn.ensemble import RandomForestClassifier

# Define directory paths for file uploads and plot storage
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'plots')
ALLOWED = {'.csv'}  # Set of allowed file extensions

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'dev-secret'  # Secret key for session management

def autodetect_target(df: pd.DataFrame) -> str:
    """
    Automatically detect the target column in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        str: Name of the target column
    """
    # Look for common target column names
    candidates = [
        c for c in df.columns if c.lower() in ["play", "class", "isfraud", "fraud", "target", "label", "y"]
    ]
    if len(candidates) > 0:
        return candidates[0]  # Return first matching candidate
    return df.columns[-1]  # Fallback to last column if no match found

@app.route('/')
def index():
    """Render the main upload page."""
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_pipeline():
    """
    Main pipeline endpoint that handles file upload and runs the complete feature selection process.
    
    Steps:
    1. File validation and upload
    2. Data loading and cleaning
    3. Preprocessing
    4. Traditional feature selection methods
    5. Genetic Algorithm feature selection
    6. Results compilation and visualization
    """
    # Handle file upload
    file = request.files.get('csv_file')
    if not file:
        flash('الرجاء اختيار ملف CSV.')
        return redirect(url_for('index'))

    # Validate file extension
    name = secure_filename(file.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED:
        flash('الملف يجب أن يكون CSV.')
        return redirect(url_for('index'))

    # Generate unique filename to prevent conflicts
    uid = uuid.uuid4().hex
    save_path = os.path.join(UPLOAD_DIR, f"{uid}_{name}")
    file.save(save_path)

    # Load CSV data with robust error handling
    try:
        df = pd.read_csv(save_path)
    except Exception:
        # Fallback to Python engine if standard reading fails
        df = pd.read_csv(save_path, sep=',', quotechar='"', engine='python')

    # Handle edge case: file read as single column (likely due to formatting issues)
    if df.shape[1] == 1:
        colname = df.columns[0]
        # Split single column into multiple columns
        df_fixed = df.iloc[:, 0].astype(str).str.split(',', expand=True)
        # Extract and clean column names from header
        new_cols = [c.strip().strip('"') for c in colname.split(',')]
        if len(new_cols) == df_fixed.shape[1]:
            df_fixed.columns = new_cols
        else:
            # Re-read file with different approach if column count doesn't match
            with open(save_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            df_fixed = pd.read_csv(io.StringIO(content), sep=',', quotechar='"', engine='python')
        df = df_fixed

    # Convert columns to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='ignore')

    # Auto-detect target column
    target_col = autodetect_target(df)

    # Clean column data: remove quotes, trim spaces, convert to numeric
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().str.replace('"', '', regex=False)
            tmp = pd.to_numeric(df[c], errors='ignore')
            df[c] = tmp

    # 1) Preprocessing Stage
    # Apply complete preprocessing pipeline: imputation, encoding, scaling, splitting
    X_train_df, X_test_df, y_train, y_test, feat_names = preprocess(df, target_col)
    # Calculate positive class ratio for class balance information
    pos_ratio = float((y_train == 1).mean()) if len(y_train) else 0.0

    # 2) Traditional Feature Selection Methods
    # Initialize base Random Forest classifier for evaluation
    base_rf = RandomForestClassifier(
        n_estimators=CFG.RF_N_EST,
        max_depth=CFG.RF_MAX_DEPTH,
        class_weight=CFG.RF_CLASS_WEIGHT,
        random_state=CFG.RANDOM_STATE,
        n_jobs=-1,
    )
    # Run all traditional feature selection methods and get performance report
    report_df = traditional_selectors_report(
        X_train_df, y_train, X_test_df, y_test, k_mode=CFG.TOP_K_MODE, base_estimator=base_rf
    )

    # 3) Genetic Algorithm Feature Selection
    # Initialize Random Forest for GA evaluation
    rf_ga = RandomForestClassifier(
        n_estimators=150, max_depth=None, random_state=42, n_jobs=-1, class_weight=CFG.RF_CLASS_WEIGHT
    )
    # Run Genetic Algorithm for feature selection
    best_mask, history, ga_time = genetic_feature_selection_v2(
        X_train_df.values,
        y_train,
        rf_ga,
        pop_size=CFG.GA_POP,
        generations=CFG.GA_GENS,
        crossover_prob=CFG.GA_CROSSOVER,
        mutation_prob=CFG.GA_MUTATION,
        tournament_k=CFG.GA_TOURNAMENT_K,
        elitism=CFG.GA_ELITISM,
        alpha=CFG.GA_ALPHA,
        patience=CFG.GA_PATIENCE,
        min_mut=CFG.GA_MIN_MUT,
        max_mut=CFG.GA_MAX_MUT,
        verbose=False,  # Disable verbose output for web interface
    )

    # Process GA results
    import numpy as np
    n_cols = X_train_df.shape[1]
    assert best_mask.shape[0] == n_cols  # Validate mask dimensions
    selected_idx = np.where(best_mask == 1)[0]  # Get indices of selected features
    selected_names = X_train_df.columns[selected_idx].tolist()  # Get names of selected features

    # Evaluate GA-selected features on test set
    rf_ga.fit(X_train_df[selected_names], y_train)
    p = rf_ga.predict_proba(X_test_df[selected_names])[:, 1]
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

    # Calculate performance metrics
    acc = accuracy_score(y_test, rf_ga.predict(X_test_df[selected_names]))
    roc_auc = roc_auc_score(y_test, p)
    pr_auc = average_precision_score(y_test, p)

    # 4) Visualization
    # Create and save GA progress plot
    plot_name = f"ga_hist_{uid}.png"
    plot_path = os.path.join(PLOTS_DIR, plot_name)
    plt.figure(figsize=(7, 4))
    plt.plot(history, marker='o')
    plt.title("GA v2 Fitness (AUC - penalty)")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=130)
    plt.close()

    # Prepare results context for template rendering
    context = {
        'file_name': name,
        'shape': tuple(df.shape),
        'target_col': target_col,
        'train_shape': tuple(X_train_df.shape),
        'test_shape': tuple(X_test_df.shape),
        'pos_ratio': round(pos_ratio, 4),
        'report_table': report_df.to_dict(orient='records'),  # Convert DataFrame to dictionary for template
        'report_cols': list(report_df.columns),
        'ga_selected_n': len(selected_names),
        'ga_selected_sample': selected_names[: min(25, len(selected_names))],  # Show sample of selected features
        'ga_metrics': {
            'accuracy': round(float(acc), 4),
            'roc_auc': round(float(roc_auc), 4),
            'pr_auc': round(float(pr_auc), 4),
            'time_min': round(float(ga_time) / 60.0, 2),  # Convert seconds to minutes
        },
        'ga_plot_url': url_for('static', filename=f'plots/{plot_name}')  # URL for the generated plot
    }

    # Render results page with all computed data
    return render_template('results.html', **context)

if __name__ == '__main__':
    # Run Flask application in debug mode
    app.run(debug=True)