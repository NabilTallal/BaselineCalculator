import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Baseline Energy Calculator ⚡")

# File upload
uploaded_file = st.file_uploader("Upload your consumption data (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Check sheets
        xls = pd.ExcelFile(uploaded_file)
        st.write("Available sheets:", xls.sheet_names)
        sheet_name = st.selectbox("Select sheet to use", xls.sheet_names, index=0)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=0)
        df = df.dropna(how='all')

        st.write("### Raw data preview")
        st.dataframe(df.head(10))

        # Clean columns
        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace("\xa0", "_")
            .str.replace(" ", "_")
            .str.lower()
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        st.write("### Cleaned columns:")
        st.write(df.columns.tolist())

        # Detect consumption column
        consumption_cols = [col for col in df.columns if "consumption" in col]
        if not consumption_cols:
            st.error("No column containing 'consumption' found.")
        else:
            consumption_col = consumption_cols[0]
            df[consumption_col] = pd.to_numeric(df[consumption_col], errors='coerce')
            df = df.dropna(subset=[consumption_col])

            # Parse date_time
            if "date_time" not in df.columns:
                st.error("No column named 'date_time' found.")
            else:
                df["date_time"] = pd.to_datetime(df["date_time"], dayfirst=True, errors='coerce')
                df = df.dropna(subset=["date_time"])
                df = df.sort_values("date_time").reset_index(drop=True)

                # Event period input
                event_start = st.text_input("Enter Event Start (e.g. 01/05/2023 14:00)")
                event_end = st.text_input("Enter Event End (e.g. 01/05/2023 18:00)")

                # Rolling mean window and adjustment.
                window_size = st.slider("Rolling mean window (points)", 2, 24, 4, step=1)
                error_factor = st.slider("Baseline downward adjustment (%)", 0.0, 10.0, 7.0, step=0.5) / 100

                if st.button("Calculate Baseline"):
                    # Rolling mean baseline.
                    df["baseline"] = df[consumption_col].rolling(window=window_size, min_periods=1).mean()
                    df["baseline"] = df["baseline"] - (error_factor * df["baseline"].mean())

                    # metrics.
                    y_true = df[consumption_col].values
                    y_pred = df["baseline"].values
                    mae = mean_absolute_error(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    smape = np.mean(2 * np.abs(y_true - y_pred) /
                                    (np.abs(y_true) + np.abs(y_pred) + 1e-3)) * 100

                    st.info("### Error Metrics:")
                    st.write(f"- Mean Absolute Error (MAE): {mae:.3f} kWh")
                    st.write(f"- Mean Squared Error (MSE): {mse:.3f} kWh²")
                    st.write(f"- Smoothed Error Rate (SMAPE): {smape:.2f}%")

                    # Event filer.
                    if event_start and event_end:
                        start = pd.to_datetime(event_start, dayfirst=True)
                        end = pd.to_datetime(event_end, dayfirst=True)
                        df_event = df[(df["date_time"] >= start) & (df["date_time"] <= end)]
                        if df_event.empty:
                            st.warning("No data in selected period. Using full dataset for event metrics.")
                            df_event = df
                    else:
                        df_event = df

                    # Risk model.
                    diff = df_event[consumption_col] - df_event["baseline"]
                    pos = diff[diff > 0]

                    percent_over = (len(pos) / max(len(df_event), 1)) * 100
                    mean_over = pos.mean() if len(pos) else 0.0
                    max_over = pos.max() if len(pos) else 0.0

                    interval_secs = (df_event["date_time"].diff().median().total_seconds()
                                     if len(df_event) > 1 else 900)
                    interval_hours = interval_secs / 3600.0
                    energy_over = pos.sum() * interval_hours

                    # Duration above baseline
                    consec = (diff > 0).astype(int)
                    duration_over_minutes = (consec.sum() * interval_secs / 60.0)

                    # Normalize components
                    mean_baseline = max(df_event["baseline"].mean(), 1e-6)
                    f = percent_over / 100.0
                    m = mean_over / mean_baseline
                    e = energy_over / (mean_baseline * (len(df_event) * interval_hours) + 1e-6)
                    d = duration_over_minutes / max((len(df_event) * interval_secs / 60.0), 1e-6)

                    # Composite risk score
                    score = 0.4 * f + 0.2 * m + 0.2 * e + 0.2 * d

                    if score >= 0.25:
                        risk_flag = "🔴 High Risk"
                    elif score >= 0.10:
                        risk_flag = "🟡 Medium Risk"
                    else:
                        risk_flag = "🟢 Low Risk"

                    # --- Results ---
                    baseline_mean = df["baseline"].mean()
                    adjustment = error_factor * baseline_mean

                    st.success(f"**Calculated Baseline (Rolling Mean with adjustment):** {baseline_mean:.2f} kWh")
                    st.write(f"**Adjustment Applied:** {adjustment:.2f} kWh")
                    st.write(f"**Composite Risk Score:** {score:.3f}")
                    st.write(f"**Gaming Risk (Event-based):** {risk_flag}")
                    st.write(f"**% of readings above baseline during event:** {percent_over:.2f}%")
                    st.write(f"**Mean over baseline:** {mean_over:.3f} kWh")
                    st.write(f"**Max over baseline:** {max_over:.3f} kWh")
                    st.write(f"**Energy over baseline:** {energy_over:.3f} kWh")
                    st.write(f"**Duration above baseline:** {duration_over_minutes:.1f} minutes")

                    # --- Plot ---
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(df_event["date_time"], df_event[consumption_col], label="Consumption")
                    ax.plot(df_event["date_time"], df_event["baseline"], color='r', linestyle='--', label="Baseline (Adjusted)")
                    ax.set_xlabel("Date Time")
                    ax.set_ylabel("Consumption (kWh)")
                    ax.set_title("Consumption vs Baseline (Event Period)")
                    ax.legend()
                    fig.autofmt_xdate()
                    st.pyplot(fig)

                    st.write("### 📊 Actual vs Baseline Comparison (Event Period)")
                    result_table = df_event[["date_time", consumption_col, "baseline"]].copy()
                    result_table["difference"] = result_table[consumption_col] - result_table["baseline"]
                    result_table = result_table.rename(columns={
                        "date_time": "Date/Time",
                        consumption_col: "Actual Consumption (kWh)",
                        "baseline": "Baseline (kWh)",
                        "difference": "Difference (kWh)"
                    })
                    st.dataframe(result_table.reset_index(drop=True))

    except Exception as e:
        st.error(f"Error reading file: {e}")
