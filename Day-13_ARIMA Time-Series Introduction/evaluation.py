"""Day 13: COVID-19 Forecasting — Evaluation"""
import logging, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import config

logger = logging.getLogger(__name__)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error — intuitive for stakeholders."""
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_all(test_df, arima_fc, ets_fc, lstm_fc):
    logger.info("=" * 60)
    logger.info("EVALUATION — FORECAST ACCURACY")
    logger.info("=" * 60)

    actual = test_df["cases"].values
    results = []

    forecasts = {"ARIMA": arima_fc, "Exp Smoothing": ets_fc, "GPU LSTM": lstm_fc}
    for name, fc in forecasts.items():
        if fc is None:
            continue
        # Align lengths
        n = min(len(actual), len(fc))
        a, p = actual[:n], fc[:n]

        mae = mean_absolute_error(a, p)
        rmse = np.sqrt(mean_squared_error(a, p))
        mp = mape(a, p)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mp})
        logger.info(f"  {name:18s} | MAE={mae:>8.1f} | RMSE={rmse:>8.1f} | MAPE={mp:>5.1f}%")

    df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    logger.info(f"\n{'='*60}\nRANKINGS\n{'='*60}")
    for i, row in df.iterrows():
        m = ["🥇", "🥈", "🥉"][min(i, 2)]
        logger.info(f"  {m} {row['Model']:18s} | RMSE={row['RMSE']:.1f} | MAPE={row['MAPE (%)']:.1f}%")

    return df


def plot_forecasts(train_df, test_df, arima_fc, ets_fc, lstm_fc):
    """The BIG plot — actual vs all forecasts on the same timeline."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [3, 1.5]})
    actual = test_df["cases"].values

    # (0) Full forecast comparison
    ax = axes[0]
    ax.plot(train_df.index, train_df["cases"], color='#BDBDBD', lw=1, alpha=0.6, label='Training data')
    ax.plot(test_df.index, actual, color='#212121', lw=2.5, label='Actual (ground truth)')

    forecast_data = [
        ("ARIMA", arima_fc, '#EF5350', '-', 2),
        ("Exp Smoothing", ets_fc, '#AB47BC', '--', 2),
        ("GPU LSTM", lstm_fc, '#4FC3F7', '-', 2.5),
    ]
    for name, fc, color, ls, lw in forecast_data:
        if fc is None:
            continue
        n = min(len(test_df), len(fc))
        ax.plot(test_df.index[:n], fc[:n], color=color, linestyle=ls, lw=lw, label=name, alpha=0.85)

    # Split line
    ax.axvline(test_df.index[0], color='#FFB74D', lw=2, linestyle='-.', alpha=0.7)
    ax.text(test_df.index[0], actual.max() * 1.02, ' ← Forecast starts', fontsize=10, color='#FFB74D')

    ax.set_title("🦠 COVID-19 Forecast Comparison — ARIMA vs Exp Smoothing vs GPU LSTM",
                  fontweight='bold', fontsize=14)
    ax.set_ylabel("Daily New Cases")
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    # (1) Forecast errors over time
    ax = axes[1]
    for name, fc, color, _, _ in forecast_data:
        if fc is None:
            continue
        n = min(len(actual), len(fc))
        error = actual[:n] - fc[:n]
        ax.plot(test_df.index[:n], error, color=color, lw=1.5, alpha=0.7, label=f'{name} error')

    ax.axhline(0, color='k', lw=1, linestyle='--')
    ax.set_title("📊 Forecast Errors Over Time (actual - predicted)", fontweight='bold')
    ax.set_ylabel("Error (cases)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/03_forecast_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 03_forecast_comparison.png")


def plot_model_comparison(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [("MAE", "#EF5350"), ("RMSE", "#4FC3F7"), ("MAPE (%)", "#66BB6A")]

    for ax, (metric, color) in zip(axes, metrics):
        bars = ax.barh(results_df["Model"], results_df[metric], color=color, edgecolor='white', alpha=0.85)
        for bar, val in zip(bars, results_df[metric]):
            ax.text(val + results_df[metric].max() * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}', va='center', fontsize=10, fontweight='bold')
        ax.set_xlabel(metric)
        ax.set_title(f"📈 {metric}", fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle("🦠 Model Comparison — Time-Series Forecasting", fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/04_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 04_model_comparison.png")


def plot_zoom_forecast(test_df, arima_fc, ets_fc, lstm_fc):
    """Zoom into first 30 days of forecast for detail."""
    fig, ax = plt.subplots(figsize=(14, 6))
    n_zoom = min(config.FORECAST_HORIZON, len(test_df))
    actual = test_df["cases"].values[:n_zoom]
    dates = test_df.index[:n_zoom]

    ax.plot(dates, actual, 'ko-', lw=2.5, markersize=5, label='Actual', zorder=5)

    for name, fc, color, marker in [("ARIMA", arima_fc, '#EF5350', 's'),
                                      ("Exp Smoothing", ets_fc, '#AB47BC', 'D'),
                                      ("GPU LSTM", lstm_fc, '#4FC3F7', '^')]:
        if fc is None:
            continue
        n = min(n_zoom, len(fc))
        ax.plot(dates[:n], fc[:n], f'{marker}--', color=color, lw=1.5, markersize=4, label=name, alpha=0.8)

    ax.set_title(f"🔍 Zoomed: First {n_zoom} Days of Forecast", fontweight='bold', fontsize=13)
    ax.set_ylabel("Daily Cases"); ax.legend(fontsize=10)
    ax.grid(alpha=0.3); ax.spines[['top', 'right']].set_visible(False)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{config.PLOT_DIR}/05_zoom_forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: 05_zoom_forecast.png")


def save_results(results_df):
    results_df.to_csv(f"{config.OUTPUT_DIR}/day13_results.csv", index=False, float_format='%.2f')
    with open(f"{config.OUTPUT_DIR}/day13_report.txt", 'w') as f:
        f.write("=" * 70 + "\n  DAY 13: COVID-19 CASE FORECASTING — FINAL REPORT\n" + "=" * 70 + "\n\n")
        f.write("FIRST TIME-SERIES PROJECT IN THE 60-DAY CHALLENGE!\n\n")
        f.write("OBJECTIVE: Forecast daily COVID-19 cases using ARIMA, ETS, and GPU LSTM\n")
        f.write(f"DATA: {config.N_DAYS} days of simulated pandemic data with 3 waves\n\n")
        f.write("-" * 70 + "\nRESULTS\n" + "-" * 70 + "\n\n")
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        f.write("\n\n" + "-" * 70 + "\nKEY TAKEAWAYS\n" + "-" * 70 + "\n\n")
        f.write("1. Time-series data MUST be split temporally — never random shuffle\n")
        f.write("2. ARIMA(p,d,q): p=AR lags, d=differencing, q=MA errors\n")
        f.write("3. Stationarity is required for ARIMA — differencing (d) fixes trends\n")
        f.write("4. Weekly seasonality in COVID data (fewer weekend reports)\n")
        f.write("5. LSTM captures complex nonlinear temporal patterns\n")
        f.write("6. Exponential Smoothing handles trend + seasonality explicitly\n")
        f.write("7. MAPE is the most intuitive metric for stakeholders\n")
        f.write("8. Forecast error grows with horizon — near-term is always more accurate\n")
    logger.info("Results saved")
