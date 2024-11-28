import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# App title
st.title("Forecasting Model Results Viewer")

# File upload for actuals
#st.header("Upload Actuals File")
#actuals_file = st.file_uploader("Upload your actuals file (CSV format)", type=["csv"], key="actuals")
actuals_file = "data/actuals.csv"

# File upload for forecasts
#st.header("Upload Forecasts File")
#forecasts_file = st.file_uploader("Upload your forecasts file (CSV format)", type=["csv"], key="forecasts")
forecasts_file = "data/forecast.csv"

if actuals_file and forecasts_file:
    # Load the data
    actuals_df = pd.read_csv(actuals_file)
    forecasts_df = pd.read_csv(forecasts_file)

    # Display previews of the uploaded data
    #st.write("### Actuals Data Preview")
    #st.dataframe(actuals_df.head())
    
    st.write("### Forecasts Data Preview")
    st.dataframe(forecasts_df.head())

    # Validate required columns
    actuals_required = {'unique_id', 'ds', 'actual'}
    forecasts_required = {'unique_id', 'ds'}
    
    # Identify model columns (all columns in forecast_df except 'unique_id' and 'ds')
    model_columns = [col for col in forecasts_df.columns if col not in {'unique_id', 'ds'}]

    # Merge the dataframes on 'unique_id' and 'ds'
    merged_df = pd.merge(actuals_df, forecasts_df, on=['unique_id', 'ds'], how='inner')

    # Sidebar for filter options
    st.sidebar.header("Filter Options")
    unique_id_filter = st.sidebar.selectbox("Select Unique ID", merged_df['unique_id'].unique())
    
    filtered_df = merged_df[merged_df['unique_id'] == unique_id_filter]

    if filtered_df.empty:
        st.warning("No data available for the selected unique ID.")
    else:
        # Line plot for actual vs forecast
        st.write(f"### Actual vs Forecast for Unique ID: {unique_id_filter}")
        plt.figure(figsize=(10, 6))
        plt.plot(filtered_df['ds'], filtered_df['y'], label='Actual', marker='o')
        
        for model in model_columns:
            plt.plot(filtered_df['ds'], filtered_df[model], label=f"{model} Forecast", marker='x')
        
        plt.legend()
        plt.title(f"Actual vs Forecast for Unique ID: {unique_id_filter}")
        plt.xlabel("Date")
        plt.ylabel("Values")
        plt.grid()
        plt.xticks(rotation=45)  # Rotate x-axis labels
        plt.tight_layout()  # Adjust layout to avoid overlap
        st.pyplot(plt)

        # Calculate error metrics for each model
        error_metrics = {}
        for model in model_columns:
            filtered_df[f'error_{model}'] = filtered_df['y'] - filtered_df[model]
            mae = filtered_df[f'error_{model}'].abs().mean()
            mse = (filtered_df[f'error_{model}'] ** 2).mean()
            error_metrics[model] = {'MAE': mae, 'MSE': mse}

        # Display error metrics
        st.write("### Error Metrics")
        for model, metrics in error_metrics.items():
            st.write(f"**{model}** - Mean Absolute Error (MAE): {metrics['MAE']:.2f}, Mean Squared Error (MSE): {metrics['MSE']:.2f}")