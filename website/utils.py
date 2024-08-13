import pandas as pd
import matplotlib.pyplot as plt
import os
from flask import current_app as app, url_for

def process_excel_file(file):
    df = pd.read_excel(file)

    # Convert 'Date ' column to datetime format and sort data by date
    df['Date '] = pd.to_datetime(df['Date '])

    # Extract date features
    df['Year'] = df['Date '].dt.year
    df['Month'] = df['Date '].dt.month
    df['Day'] = df['Date '].dt.day
    df['Time'] = df['Date '].dt.time  # Extract time from datetime

    # Calculate total amount
    df['Total Amount'] = df['Qte'] * df['PUHT']

    return df

def plot_outlier_removal(df):
    # Group data by year, month, day, and time to count quantity sold
    grouped_qte_by_date = df.groupby(['Year', 'Month', 'Day', 'Time'])['Qte'].sum().reset_index()

    # Set the index for plotting
    grouped_qte_by_date.set_index(['Year', 'Month', 'Day', 'Time'], inplace=True)

    # Plot the total quantity sold by date and time
    plt.figure(figsize=(16, 8))
    grouped_qte_by_date['Qte'].plot(kind='line', marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    plt.title('Quantity Sold Over Time', fontsize=16)
    plt.xlabel('Date and Time', fontsize=14)
    plt.ylabel('Quantity Sold', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plot_outliers.png')
    plt.savefig(plot_path)
    plt.close()

    return url_for('static', filename='plot_outliers.png')
