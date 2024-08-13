import base64
import io
from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
from pmdarima import acf, pacf, plot_acf, plot_pacf
from scipy import stats
from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import json
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import boxcox
matplotlib.use('Agg')  # Using the Agg backend which is non-GUI

data_views = Blueprint('data_views', __name__)

def make_stationary(series):
    if series.nunique() <= 1:
        return series, "Column is constant and cannot be made stationary"
    result = adfuller(series.dropna())
    # Unpack the first two values from the result tuple
    adf_stat, p_value = result[0], result[1]
    if p_value < 0.05:
        return series, "Stationary"
    else:
        return series.diff().dropna(), "Differenced to make stationary"
def acf_pacf_analysis(series):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series.dropna(), lags=40, ax=ax[0])
    plot_pacf(series.dropna(), lags=40, ax=ax[1])
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf-8')
@data_views.route("/clean-data",methods=["POST"])
def clean_data():
    try:
        if 'file' not in request.files or request.files['file'].filename == '':
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if not file.filename.endswith(('.xls', '.xlsx')):
            return jsonify({"error": "File is not an Excel file"}), 400

        df = pd.read_excel(file)
        if df.empty:
            return jsonify({"error": "The DataFrame is empty. Please check the uploaded file."}), 400

        user_mappings = request.form.get('mappings', '{}')
        user_mappings = json.loads(user_mappings)
        
        if not user_mappings:
            user_mappings = {col: col for col in df.columns}

        valid_mapped_columns = {}
        missing_columns = []
        
        for friendly_name, actual_name in user_mappings.items():
            if actual_name in df.columns:
                valid_mapped_columns[friendly_name] = actual_name
            else:
                missing_columns.append(actual_name)

        if missing_columns:
            return jsonify({"error": f"Missing columns: {', '.join(missing_columns)}"}), 400

        numeric_columns = [col for col in valid_mapped_columns.values() if np.issubdtype(df[col].dtype, np.number)]
        non_numeric_columns = [col for col in valid_mapped_columns.values() if not np.issubdtype(df[col].dtype, np.number)]

        if not numeric_columns:
            return jsonify({"error": "No numeric columns found for processing."}), 400

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isna().all():
                return jsonify({"error": "Date column is invalid or improperly formatted."}), 400
            df.set_index('Date', inplace=True)

        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        initial_count = len(df)
        df.dropna(subset=numeric_columns, inplace=True)
        final_count = len(df)

        if final_count == 0:
            return jsonify({"error": "All data points have been dropped due to NaNs."}), 400

        messages = [f"Dropped rows with missing values. Total data points: {final_count}"]

        for col in numeric_columns:
            df[col] = df[col].apply(lambda x: x if x >= 0 else np.nan)

        # Recalculate z-scores and handle index alignment
        df_cleaned = df.copy()
        z_scores = np.abs(stats.zscore(df_cleaned[numeric_columns].dropna()))
        z_scores_df = pd.DataFrame(z_scores, index=df_cleaned.dropna().index, columns=numeric_columns)
        is_outlier = (z_scores_df > 3).any(axis=1)

        # Ensure boolean indexer aligns with DataFrame
        df_cleaned = df_cleaned.loc[~is_outlier.reindex(df_cleaned.index, fill_value=False)]

        messages.append(f"Detected and handled outliers. Number of data points after removing outliers: {len(df_cleaned)}")

        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned[numeric_columns]), columns=numeric_columns, index=df_cleaned.index)
        messages.append("Normalized data using Min-Max scaling.")

        stationarity_results = {}
        for col in numeric_columns:
            stationary_data, message = make_stationary(df[col])
            stationarity_results[col] = message

        df_transformed = df.copy()
        skewness_before_transformation = {}
        for col in numeric_columns:
            skewness_before = df_transformed[col].skew()
            skewness_before_transformation[col] = skewness_before

        for col in numeric_columns:
            skewness = df_transformed[col].skew()
            if skewness > 1:
                df_transformed[col] = np.sign(df_transformed[col]) * np.log1p(np.abs(df_transformed[col]))

        skewness_after_transformation = {}
        for col in numeric_columns:
            skewness_after = df_transformed[col].skew()
            skewness_after_transformation[col] = skewness_after

        clean_df = pd.DataFrame({
            'Qte_MinMax': df_normalized['Qte'],
            'Qte_Standardized': (df_cleaned['Qte'] - df_cleaned['Qte'].mean()) / df_cleaned['Qte'].std(),
            'Qte_MinMax_Diff': df_normalized['Qte'].diff().dropna(),
            'Qte_Standardized_Diff': (df_cleaned['Qte'] - df_cleaned['Qte'].mean()).diff().dropna(),
            'Qte_Cleaned': df_cleaned['Qte']
        })

        # Print the columns of clean_df
        print(clean_df.columns.tolist())

        # Perform ACF and PACF analysis
        acf_pacf_plot = acf_pacf_analysis(clean_df['Qte_Standardized_Diff'])

        # Generate a plot of the Qte column
        if 'Qte' in clean_df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(clean_df['Qte_Cleaned'], marker='o', linestyle='-', color='b')
            plt.title('Qte Column After Processing')
            plt.xlabel('Index')
            plt.ylabel('Qte')
            plt.grid(True)
            plot_img = io.BytesIO()
            plt.savefig(plot_img, format='png')
            plot_img.seek(0)
            plot_base64 = base64.b64encode(plot_img.getvalue()).decode('utf-8')
            plot_img.close()
        else:
            plot_base64 = None

        # Generate histograms for original and transformed data
        plt.figure(figsize=(18, 12))
        for i, col in enumerate(numeric_columns):
            plt.subplot(len(numeric_columns), 2, 2*i + 1)
            plt.hist(df[col], bins=30, color='blue', alpha=0.7, label='Original')
            plt.title(f'Original Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()

            plt.subplot(len(numeric_columns), 2, 2*i + 2)
            plt.hist(df_transformed[col], bins=30, color='green', alpha=0.7, label='Log Transformed')
            plt.title(f'Log Transformed Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.legend()

        plt.tight_layout()
        hist_plot_img = io.BytesIO()
        plt.savefig(hist_plot_img, format='png')
        hist_plot_img.seek(0)
        hist_plot_base64 = base64.b64encode(hist_plot_img.getvalue()).decode('utf-8')
        hist_plot_img.close()

        response = {
            "messages": messages,
            "stationarity_results": stationarity_results,
            "normalized_values": df_normalized.head().to_dict(orient='records'),
            "mapped_columns": valid_mapped_columns,
            "skewness_before_transformation": skewness_before_transformation,
            "skewness_after_transformation": skewness_after_transformation,
            "qte_plot": plot_base64,
            "hist_plot": hist_plot_base64,
            "acf_pacf_plot": acf_pacf_plot,
            "clean_df_columns": clean_df.columns.tolist()
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@data_views.route('/generate-plots', methods=['POST'])
def generate_plots():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']

        if not file.filename.endswith(('.xls', '.xlsx')):
            return jsonify({"error": "File is not an Excel file"}), 400

        df = pd.read_excel(file)
        if df.empty:
            return jsonify({"error": "The DataFrame is empty. Please check the uploaded file."}), 400

        user_mappings = request.json.get('mappings', {})
        valid_mapped_columns = {friendly_name: actual_name for friendly_name, actual_name in user_mappings.items() if actual_name in df.columns}

        if 'Marque' not in valid_mapped_columns.values():
            return jsonify({"error": "No 'Marque' column found in the dataset"}), 400

        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=numeric_columns, inplace=True)

        grouped = df.groupby('Marque')['Qte'].sum().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Marque', y='Qte', data=grouped)
        plt.title('Total Quantity Sold by Marque')
        plt.xlabel('Marque')
        plt.ylabel('Total Quantity Sold')
        plt.xticks(rotation=45)

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        response = {
            "total_quantity_sold_plot": plot_base64,
        }

        stationarity_plots = {}
        for col in numeric_columns:
            stationary_data, _ = make_stationary(df[col])

            plt.figure(figsize=(22, 10))
            plt.plot(stationary_data.index, stationary_data.values, marker='o', linestyle='-')
            plt.title(f'{col} Over Time (Stationary)')
            plt.xlabel('Date')
            plt.ylabel(col)

            img_stationary = io.BytesIO()
            plt.savefig(img_stationary, format='png')
            img_stationary.seek(0)
            img_stationary_base64 = base64.b64encode(img_stationary.getvalue()).decode('utf-8')
            plt.close()
            stationarity_plots[col] = img_stationary_base64
        
        response["stationary_plots"] = stationarity_plots

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@data_views.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        # Load and prepare the data
        file = request.files['file']
        df = pd.read_csv(file)

        # Add 'Date' column if missing
        if 'Date' not in df.columns:
            # Create a default 'Date' column
            df['Date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
        
        # Parse dates and set index
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if df['Date'].isna().all():
            return jsonify({'error': 'No valid date values found in the Date column.'}), 400
        
        df = df.set_index('Date')
        df = df.dropna()  # Drop rows with missing values

        # Define columns for forecasting and feature analysis
        forecast_column = 'Qte_Cleaned'
        features_columns = ['Qte_MinMax', 'Qte_Standardized', 'Qte_MinMax_Diff', 'Qte_Standardized_Diff']

        # Check if required columns exist
        required_columns = [forecast_column] + features_columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {", ".join(missing_columns)}'}), 400

        # Create lagged features
        df['lag_1'] = df[forecast_column].shift(1)
        df['lag_2'] = df[forecast_column].shift(2)
        df = df.dropna()

        # Prepare time series and features
        time_series = df[forecast_column]
        features = df[features_columns + ['lag_1', 'lag_2']]

        # Train-test split
        train_size = int(len(time_series) * 0.8)
        train_ts = time_series[:train_size]
        test_ts = time_series[train_size:]
        train_features = features[:train_size]
        test_features = features[train_size:]

        # Check for stationarity in the target column
        def make_stationary(ts):
            return sm.tsa.stattools.adfuller(ts.dropna())[1] < 0.05

        if not make_stationary(train_ts):
            train_ts, lambda_ = boxcox(train_ts.dropna() + 1)
            test_ts, _ = boxcox(test_ts.dropna() + 1, lmbda=lambda_)
        else:
            lambda_ = None

        # Define SARIMA parameters
        sarima_order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        # Fit the SARIMA model
        best_model = sm.tsa.statespace.SARIMAX(train_ts,
                                               order=sarima_order,
                                               seasonal_order=seasonal_order,
                                               enforce_stationarity=False,
                                               enforce_invertibility=False).fit(disp=False)

        # Out-of-Sample Forecasts
        forecast_steps = len(test_ts)
        forecast = best_model.get_forecast(steps=forecast_steps)
        forecast_mean = forecast.predicted_mean
        forecast_conf = forecast.conf_int()

        # Ensure forecast_mean is a Series with the correct index
        forecast_mean.index = test_ts.index

        # Inverse transform forecasts if Box-Cox was applied
        if lambda_ is not None:
            forecast_values = np.exp(forecast_mean) - 1
            test_ts = np.exp(test_ts) - 1
        else:
            forecast_values = forecast_mean

        # Align forecast with test_ts index
        valid_indices = test_ts.index.intersection(forecast_values.index)
        forecast_values = forecast_values.loc[valid_indices]
        test_ts = test_ts.loc[valid_indices]

        if test_ts.empty or forecast_values.empty:
            return jsonify({'error': 'Test or forecast values are empty after alignment. Check data processing steps.'}), 400

        # Calculate MSE, RMSE, and R^2 for test set
        mse_test = mean_squared_error(test_ts, forecast_values)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(test_ts, forecast_values)

        # Evaluate the model on the training set
        train_forecast = best_model.get_prediction(start=pd.to_datetime(train_ts.index[0]), end=pd.to_datetime(train_ts.index[-1]))
        train_forecast_mean = train_forecast.predicted_mean
        train_forecast_values = pd.Series(train_forecast_mean.values, index=train_ts.index)

        # Calculate MSE, RMSE, and R^2 for training set
        mse_train = mean_squared_error(train_ts, train_forecast_values)
        rmse_train = np.sqrt(mse_train)
        r2_train = r2_score(train_ts, train_forecast_values)

        # Residual Analysis
        residuals = test_ts - forecast_values

        # Plot residuals and histogram
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # Plot residuals
        axs[0].plot(residuals.index, residuals, color='blue')
        axs[0].set_title('Residuals')
        axs[0].set_xlabel('Date')
        axs[0].set_ylabel('Residual')
        axs[0].axhline(y=0, color='red', linestyle='--')

        # Plot residuals histogram
        axs[1].hist(residuals.dropna(), bins=30, color='blue', edgecolor='black')
        axs[1].set_title('Residuals Histogram')
        axs[1].set_xlabel('Residual')
        axs[1].set_ylabel('Frequency')

        plt.tight_layout()

        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()

        # Convert datetime indices to string
        forecast_values.index = forecast_values.index.strftime('%Y-%m-%d')
        test_ts.index = test_ts.index.strftime('%Y-%m-%d')

        return jsonify({
            'mse_test': mse_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test,
            'mse_train': mse_train,
            'rmse_train': rmse_train,
            'r2_train': r2_train,
            'forecast_values': forecast_values.to_dict(),
            'residuals_plot': img_str
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500