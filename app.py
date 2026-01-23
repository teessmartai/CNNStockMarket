"""
CNN Stock Market Prediction Dashboard

A Streamlit web application for generating stock movement predictions
using a trained 1D CNN model.

Run with: streamlit run app.py

Phase 6 Features:
- Custom CSV data upload
- Configurable timeframes and presets
- Sample mode selection

Phase 7 Features:
- Backtesting framework
- Performance metrics and visualization
- Report generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import io

# Set page config
st.set_page_config(
    page_title="Stock Movement Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .buy-signal {
        color: #00c853;
        font-size: 32px;
        font-weight: bold;
    }
    .sell-signal {
        color: #ff1744;
        font-size: 32px;
        font-weight: bold;
    }
    .confidence-high {
        color: #00c853;
    }
    .confidence-medium {
        color: #ff9800;
    }
    .confidence-low {
        color: #ff1744;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)


def load_predictor():
    """Load the prediction model."""
    from src.prediction.predictor import Predictor
    from src.utils.config import MODELS_DIR

    model_path = MODELS_DIR / 'best_model.pt'
    if not model_path.exists():
        return None
    return Predictor(model_path)


def get_sp500_tickers():
    """Get list of S&P 500 tickers."""
    from src.data.fetcher import fetch_sp500_tickers
    try:
        return fetch_sp500_tickers()
    except Exception:
        # Fallback list
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']


def fetch_price_data(ticker: str, days: int = 365):
    """Fetch historical price data."""
    from src.data.fetcher import fetch_stock_data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    return fetch_stock_data(ticker, start_date, end_date)


def create_price_chart(df: pd.DataFrame, ticker: str, prediction=None):
    """Create an interactive price chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot price
    ax.plot(df.index, df['Close'], 'b-', linewidth=1.5, label='Close Price')
    ax.fill_between(df.index, df['Low'], df['High'], alpha=0.2, color='blue')

    # Highlight prediction window (last 256 days)
    if len(df) > 256:
        window_start_idx = len(df) - 256
        ax.axvspan(df.index[window_start_idx], df.index[-1], alpha=0.1, color='yellow')

    # Add prediction annotation
    if prediction:
        color = 'green' if prediction.signal == 'BUY' else 'red'
        arrow = '↑' if prediction.signal == 'BUY' else '↓'
        ax.annotate(
            f'{prediction.signal} {arrow}\n{prediction.confidence:.1%}',
            xy=(df.index[-1], df['Close'].iloc[-1]),
            xytext=(20, 20),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color=color),
        )

    ax.set_ylabel('Price ($)')
    ax.set_title(f'{ticker} Price History')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()

    return fig


def main():
    """Main application."""

    # Sidebar
    st.sidebar.title("📈 Stock Predictor")
    st.sidebar.markdown("---")

    # Model status
    predictor = load_predictor()
    if predictor is None:
        st.sidebar.error("⚠️ No trained model found")
        st.sidebar.info("Train a model using the training notebook first.")
    else:
        st.sidebar.success("✓ Model loaded")

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Single Prediction", "Batch Analysis", "Top Signals", "Backtesting", "Custom Data", "About"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Settings")

    # Horizon selection
    horizon = st.sidebar.selectbox(
        "Prediction Horizon",
        [5, 30],
        format_func=lambda x: f"T+{x} days",
    )

    # Main content
    if page == "Single Prediction":
        single_prediction_page(predictor, horizon)
    elif page == "Batch Analysis":
        batch_analysis_page(predictor, horizon)
    elif page == "Top Signals":
        top_signals_page(predictor, horizon)
    elif page == "Backtesting":
        backtesting_page(predictor, horizon)
    elif page == "Custom Data":
        custom_data_page(predictor, horizon)
    else:
        about_page()


def single_prediction_page(predictor, horizon):
    """Single stock prediction page."""
    st.title("🔮 Single Stock Prediction")
    st.markdown("Enter a ticker symbol to get a prediction.")

    # Get tickers
    tickers = get_sp500_tickers()

    col1, col2 = st.columns([2, 1])

    with col1:
        ticker = st.selectbox(
            "Select Stock",
            options=tickers,
            index=0,
        )

    with col2:
        custom_ticker = st.text_input("Or enter ticker:", "")
        if custom_ticker:
            ticker = custom_ticker.upper()

    if st.button("🎯 Get Prediction", type="primary"):
        if predictor is None:
            st.error("No model available. Please train a model first.")
            return

        with st.spinner(f"Analyzing {ticker}..."):
            try:
                prediction = predictor.predict(ticker, horizon=horizon)

                # Display results
                st.markdown("---")

                col1, col2, col3 = st.columns(3)

                with col1:
                    signal_class = "buy-signal" if prediction.signal == "BUY" else "sell-signal"
                    st.markdown(f'<p class="{signal_class}">{prediction.signal}</p>', unsafe_allow_html=True)
                    st.caption("Predicted Signal")

                with col2:
                    conf_pct = prediction.confidence * 100
                    if conf_pct >= 70:
                        conf_class = "confidence-high"
                    elif conf_pct >= 50:
                        conf_class = "confidence-medium"
                    else:
                        conf_class = "confidence-low"
                    st.markdown(f'<p class="{conf_class}" style="font-size:32px;font-weight:bold;">{conf_pct:.1f}%</p>', unsafe_allow_html=True)
                    st.caption("Confidence")

                with col3:
                    st.markdown(f'<p style="font-size:32px;font-weight:bold;">T+{horizon}</p>', unsafe_allow_html=True)
                    st.caption("Horizon (days)")

                # Probability breakdown
                st.markdown("### Probability Breakdown")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Bullish Probability", f"{prediction.bullish_prob:.1%}")
                    st.progress(prediction.bullish_prob)

                with col2:
                    st.metric("Bearish Probability", f"{prediction.bearish_prob:.1%}")
                    st.progress(prediction.bearish_prob)

                # Price chart
                st.markdown("### Price History")
                df = fetch_price_data(ticker)
                if not df.empty:
                    fig = create_price_chart(df, ticker, prediction)
                    st.pyplot(fig)
                    plt.close()

            except Exception as e:
                st.error(f"Error: {str(e)}")


def batch_analysis_page(predictor, horizon):
    """Batch analysis page."""
    st.title("📊 Batch Analysis")
    st.markdown("Generate predictions for multiple stocks at once.")

    if predictor is None:
        st.error("No model available. Please train a model first.")
        return

    tickers = get_sp500_tickers()

    # Selection options
    option = st.radio(
        "Select stocks to analyze:",
        ["Top 10", "Top 50", "All S&P 500", "Custom Selection"],
        horizontal=True,
    )

    if option == "Top 10":
        selected_tickers = tickers[:10]
    elif option == "Top 50":
        selected_tickers = tickers[:50]
    elif option == "All S&P 500":
        selected_tickers = tickers
    else:
        selected_tickers = st.multiselect(
            "Select tickers:",
            options=tickers,
            default=tickers[:5],
        )

    st.info(f"Selected {len(selected_tickers)} stocks")

    if st.button("🚀 Run Batch Analysis", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        predictions = []
        for i, ticker in enumerate(selected_tickers):
            status_text.text(f"Processing {ticker}... ({i+1}/{len(selected_tickers)})")
            progress_bar.progress((i + 1) / len(selected_tickers))

            try:
                pred = predictor.predict(ticker, horizon=horizon)
                predictions.append({
                    'Ticker': pred.ticker,
                    'Signal': pred.signal,
                    'Confidence': pred.confidence,
                    'Bullish %': pred.bullish_prob,
                    'Bearish %': pred.bearish_prob,
                })
            except Exception as e:
                predictions.append({
                    'Ticker': ticker,
                    'Signal': 'ERROR',
                    'Confidence': 0,
                    'Bullish %': 0,
                    'Bearish %': 0,
                })

        status_text.text("Analysis complete!")

        # Display results
        df = pd.DataFrame(predictions)
        df_valid = df[df['Signal'] != 'ERROR']

        # Summary metrics
        st.markdown("### Summary")
        col1, col2, col3, col4 = st.columns(4)

        buy_count = len(df_valid[df_valid['Signal'] == 'BUY'])
        sell_count = len(df_valid[df_valid['Signal'] == 'SELL'])

        with col1:
            st.metric("Total Analyzed", len(df_valid))
        with col2:
            st.metric("BUY Signals", buy_count)
        with col3:
            st.metric("SELL Signals", sell_count)
        with col4:
            avg_conf = df_valid['Confidence'].mean() if len(df_valid) > 0 else 0
            st.metric("Avg Confidence", f"{avg_conf:.1%}")

        # Results table
        st.markdown("### All Predictions")
        st.dataframe(
            df.style.format({
                'Confidence': '{:.1%}',
                'Bullish %': '{:.1%}',
                'Bearish %': '{:.1%}',
            }).apply(
                lambda x: ['background-color: #c8e6c9' if v == 'BUY' else
                          'background-color: #ffcdd2' if v == 'SELL' else ''
                          for v in x],
                subset=['Signal']
            ),
            use_container_width=True,
        )

        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


def top_signals_page(predictor, horizon):
    """Top signals page."""
    st.title("🏆 Top Signals")
    st.markdown("View the strongest BUY and SELL signals.")

    if predictor is None:
        st.error("No model available. Please train a model first.")
        return

    n_stocks = st.slider("Number of stocks to analyze:", 10, 100, 50)

    if st.button("🔍 Find Top Signals", type="primary"):
        tickers = get_sp500_tickers()[:n_stocks]

        progress_bar = st.progress(0)
        status_text = st.empty()

        predictions = []
        for i, ticker in enumerate(tickers):
            status_text.text(f"Processing {ticker}... ({i+1}/{len(tickers)})")
            progress_bar.progress((i + 1) / len(tickers))

            try:
                pred = predictor.predict(ticker, horizon=horizon)
                predictions.append(pred)
            except Exception:
                pass

        status_text.text("Analysis complete!")

        # Separate BUY and SELL signals
        buy_signals = sorted(
            [p for p in predictions if p.signal == 'BUY'],
            key=lambda x: x.confidence,
            reverse=True
        )
        sell_signals = sorted(
            [p for p in predictions if p.signal == 'SELL'],
            key=lambda x: x.confidence,
            reverse=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🟢 Top BUY Signals")
            if buy_signals:
                buy_df = pd.DataFrame([
                    {
                        'Ticker': p.ticker,
                        'Confidence': f"{p.confidence:.1%}",
                        'Bullish %': f"{p.bullish_prob:.1%}",
                    }
                    for p in buy_signals[:10]
                ])
                st.dataframe(buy_df, use_container_width=True)
            else:
                st.info("No BUY signals found")

        with col2:
            st.markdown("### 🔴 Top SELL Signals")
            if sell_signals:
                sell_df = pd.DataFrame([
                    {
                        'Ticker': p.ticker,
                        'Confidence': f"{p.confidence:.1%}",
                        'Bearish %': f"{p.bearish_prob:.1%}",
                    }
                    for p in sell_signals[:10]
                ])
                st.dataframe(sell_df, use_container_width=True)
            else:
                st.info("No SELL signals found")


def custom_data_page(predictor, horizon):
    """Custom data analysis page (Phase 6)."""
    st.title("📁 Custom Data Analysis")
    st.markdown("Upload your own OHLCV data for analysis and prediction.")

    # Import Phase 6 modules
    try:
        from src.data.csv_loader import load_csv, get_data_summary, validate_ohlcv_data, infer_timeframe
        from src.data.preprocessor import estimate_sample_count, compare_sample_modes
        from src.utils.presets import list_presets, get_preset, PRESETS
    except ImportError as e:
        st.error(f"Phase 6 modules not available: {e}")
        return

    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Presets", "Data Analysis"])

    with tab1:
        st.markdown("### Upload CSV File")
        st.markdown("""
        Upload a CSV file with OHLCV (Open, High, Low, Close, Volume) data.

        **Supported formats:**
        - Standard column names: Open, High, Low, Close, Volume
        - Common variations: open, close, vol, etc.
        - Auto-detection of date columns
        """)

        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with OHLCV data"
        )

        if uploaded_file is not None:
            try:
                # Read CSV to show preview
                df_preview = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)  # Reset file pointer

                st.markdown("#### File Preview")
                st.dataframe(df_preview.head(10), use_container_width=True)

                st.markdown(f"**Rows:** {len(df_preview)} | **Columns:** {list(df_preview.columns)}")

                # Column mapping options
                st.markdown("#### Column Mapping")
                col1, col2 = st.columns(2)

                with col1:
                    auto_detect = st.checkbox("Auto-detect columns", value=True)

                if not auto_detect:
                    with col2:
                        date_col = st.selectbox("Date column", [''] + list(df_preview.columns))

                    col_mapping = {}
                    cols = st.columns(5)
                    for i, col_name in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
                        with cols[i]:
                            selected = st.selectbox(
                                col_name,
                                [''] + list(df_preview.columns),
                                key=f"map_{col_name}"
                            )
                            if selected:
                                col_mapping[col_name] = selected
                else:
                    date_col = None
                    col_mapping = None

                # Load and validate data
                if st.button("Load and Validate Data", type="primary"):
                    with st.spinner("Loading data..."):
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        try:
                            df = load_csv(
                                tmp_path,
                                column_mapping=col_mapping,
                                date_column=date_col if date_col else None,
                                validate=True
                            )

                            # Store in session state
                            st.session_state['custom_data'] = df
                            st.session_state['custom_data_name'] = uploaded_file.name

                            st.success(f"Data loaded successfully! {len(df)} rows")

                            # Show summary
                            summary = get_data_summary(df)
                            st.markdown("#### Data Summary")

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Rows", summary['rows'])
                            with col2:
                                st.metric("Timeframe", summary.get('timeframe', 'Unknown'))
                            with col3:
                                if 'price_latest' in summary:
                                    st.metric("Latest Price", f"${summary['price_latest']:.2f}")

                            if 'start_date' in summary:
                                st.markdown(f"**Date Range:** {summary['start_date']} to {summary['end_date']}")

                            # Validation results
                            is_valid, issues = validate_ohlcv_data(df, strict=False)
                            if issues:
                                st.warning("Data validation warnings:")
                                for issue in issues:
                                    st.write(f"- {issue}")
                            else:
                                st.success("Data validation passed!")

                        except Exception as e:
                            st.error(f"Error loading data: {e}")

                        finally:
                            Path(tmp_path).unlink(missing_ok=True)

    with tab2:
        st.markdown("### Configuration Presets")
        st.markdown("Select a preset configuration for your asset class and timeframe.")

        # Preset selection
        preset_name = st.selectbox(
            "Select Preset",
            list_presets(),
            format_func=lambda x: f"{x} - {PRESETS[x].description[:50]}..."
        )

        if preset_name:
            preset = get_preset(preset_name)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Preset Configuration")
                st.write(f"**Asset Class:** {preset.asset_class.value}")
                st.write(f"**Timeframe:** {preset.timeframe}")
                st.write(f"**Window Size:** {preset.window_size}")
                st.write(f"**Horizons:** {preset.horizons}")

            with col2:
                st.markdown("#### Training Settings")
                st.write(f"**Sample Mode:** {preset.sample_mode}")
                st.write(f"**Sample Stride:** {preset.sample_stride}")
                st.write(f"**Trading Days Ratio:** {preset.trading_days_ratio:.0%}")
                st.write(f"**Min Data Rows:** {preset.min_data_rows}")

            # Sample estimation if data is loaded
            if 'custom_data' in st.session_state:
                df = st.session_state['custom_data']
                st.markdown("---")
                st.markdown("#### Sample Estimation for Your Data")

                comparison = compare_sample_modes(
                    df,
                    window_size=preset.window_size,
                    horizon=preset.horizons[0],
                    custom_stride=preset.sample_stride
                )

                results = []
                for mode_name, stats in comparison.items():
                    results.append({
                        'Mode': mode_name,
                        'Samples': stats['samples'],
                        'Train': stats['train_samples'],
                        'Val': stats['val_samples'],
                        'Sufficient': 'Yes' if stats['sufficient'] else 'No'
                    })

                st.dataframe(pd.DataFrame(results), use_container_width=True)

    with tab3:
        st.markdown("### Data Analysis")

        if 'custom_data' not in st.session_state:
            st.info("Upload a CSV file in the 'Upload CSV' tab first.")
            return

        df = st.session_state['custom_data']
        data_name = st.session_state.get('custom_data_name', 'Custom Data')

        st.markdown(f"**Analyzing:** {data_name}")

        # Price chart
        st.markdown("#### Price Chart")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['Close'], 'b-', linewidth=0.5, label='Close')
        ax.fill_between(df.index, df['Low'], df['High'], alpha=0.2, color='blue')
        ax.set_title(f'{data_name} Price History')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close()

        # Volume chart
        st.markdown("#### Volume")
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(df.index, df['Volume'], width=0.8, alpha=0.7)
        ax.set_ylabel('Volume')
        fig.autofmt_xdate()
        st.pyplot(fig)
        plt.close()

        # Statistics
        st.markdown("#### Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Min Price", f"${df['Close'].min():.2f}")
        with col2:
            st.metric("Max Price", f"${df['Close'].max():.2f}")
        with col3:
            st.metric("Mean Price", f"${df['Close'].mean():.2f}")
        with col4:
            returns = df['Close'].pct_change().dropna()
            st.metric("Volatility", f"{returns.std() * 100:.2f}%")

        # Timeframe info
        detected_tf = infer_timeframe(df)
        if detected_tf:
            st.info(f"Detected timeframe: {detected_tf}")

        # Download processed data
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        st.download_button(
            label="Download Processed Data",
            data=csv_buffer.getvalue(),
            file_name=f"processed_{data_name}",
            mime="text/csv"
        )


def backtesting_page(predictor, horizon):
    """Backtesting page (Phase 7)."""
    st.title("📊 Backtesting")
    st.markdown("Evaluate model performance on historical data.")

    if predictor is None:
        st.error("No model available. Please train a model first.")
        return

    # Import backtesting modules
    try:
        from src.backtesting.engine import BacktestEngine, BacktestConfig
        from src.backtesting.plots import (
            plot_equity_curve,
            plot_drawdown,
            plot_confusion_matrix,
            plot_returns_distribution,
            create_backtest_dashboard,
        )
        from src.backtesting.report import generate_report, export_trades_csv, export_metrics_json
        from src.utils.config import PROJECT_ROOT
    except ImportError as e:
        st.error(f"Backtesting modules not available: {e}")
        return

    # Tabs for different functions
    tab1, tab2, tab3 = st.tabs(["Run Backtest", "Results", "Reports"])

    with tab1:
        st.markdown("### Configure Backtest")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Stock Selection")
            tickers = get_sp500_tickers()
            ticker = st.selectbox("Select Stock", options=tickers, index=0, key="bt_ticker")

            custom_ticker = st.text_input("Or enter ticker:", "", key="bt_custom_ticker")
            if custom_ticker:
                ticker = custom_ticker.upper()

            st.markdown("#### Date Range")
            col_start, col_end = st.columns(2)
            with col_start:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                    key="bt_start"
                )
            with col_end:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now(),
                    key="bt_end"
                )

        with col2:
            st.markdown("#### Strategy Settings")

            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000,
                key="bt_capital"
            )

            commission = st.slider(
                "Commission (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                key="bt_commission"
            ) / 100

            confidence_threshold = st.slider(
                "Confidence Threshold (%)",
                min_value=50,
                max_value=90,
                value=55,
                step=5,
                key="bt_confidence"
            ) / 100

            position_pct = st.slider(
                "Position Size (% of equity)",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                key="bt_position"
            ) / 100

            prediction_freq = st.selectbox(
                "Prediction Frequency",
                options=[1, 5, 10, 20],
                index=1,
                format_func=lambda x: f"Every {x} day(s)",
                key="bt_freq"
            )

        # Run backtest button
        if st.button("🚀 Run Backtest", type="primary", key="bt_run"):
            config = BacktestConfig(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=initial_capital,
                commission=commission,
                slippage=0.0005,
                confidence_threshold=confidence_threshold,
                position_sizing='percentage',
                position_pct=position_pct,
                allow_short=False,
                horizon=horizon,
                prediction_frequency=prediction_freq,
                verbose=False,
            )

            with st.spinner(f"Running backtest for {ticker}..."):
                try:
                    engine = BacktestEngine(predictor=predictor)
                    result = engine.run_backtest(ticker, config)

                    # Store result in session state
                    st.session_state['backtest_result'] = result
                    st.session_state['backtest_ticker'] = ticker

                    st.success("Backtest completed!")

                    # Show summary metrics
                    st.markdown("### Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        acc = result.metrics.get('accuracy', 0)
                        st.metric("Accuracy", f"{acc:.1%}")
                    with col2:
                        ret = result.metrics.get('total_return', 0)
                        st.metric("Total Return", f"{ret:.1%}")
                    with col3:
                        sharpe = result.metrics.get('sharpe_ratio', 0)
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    with col4:
                        max_dd = result.metrics.get('max_drawdown', 0)
                        st.metric("Max Drawdown", f"{max_dd:.1%}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        win_rate = result.metrics.get('win_rate', 0)
                        st.metric("Win Rate", f"{win_rate:.1%}")
                    with col2:
                        n_trades = len(result.trades)
                        st.metric("Total Trades", n_trades)
                    with col3:
                        profit_factor = result.metrics.get('profit_factor', 0)
                        st.metric("Profit Factor", f"{profit_factor:.2f}")
                    with col4:
                        n_preds = result.metrics.get('n_predictions', len(result.predictions))
                        st.metric("Predictions", n_preds)

                except Exception as e:
                    st.error(f"Backtest failed: {str(e)}")

    with tab2:
        st.markdown("### Backtest Results")

        if 'backtest_result' not in st.session_state:
            st.info("Run a backtest first to see results.")
            return

        result = st.session_state['backtest_result']
        ticker = st.session_state.get('backtest_ticker', 'Unknown')

        st.markdown(f"**Results for: {ticker}** ({result.start_date} to {result.end_date})")

        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization",
            ["Dashboard", "Equity Curve", "Drawdown", "Confusion Matrix", "Returns Distribution"],
            key="bt_viz"
        )

        if viz_option == "Dashboard":
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                fig = create_backtest_dashboard(result)
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Insufficient data for dashboard visualization")

        elif viz_option == "Equity Curve":
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                fig = plot_equity_curve(result.equity_curve, title=f"{ticker} - Equity Curve")
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No equity curve data available")

        elif viz_option == "Drawdown":
            if result.equity_curve is not None and len(result.equity_curve) > 0:
                fig = plot_drawdown(result.equity_curve, title=f"{ticker} - Drawdown")
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No equity curve data available")

        elif viz_option == "Confusion Matrix":
            if result.predictions:
                preds = [p['predicted_direction'] for p in result.predictions if p.get('actual_direction', -1) >= 0]
                acts = [p['actual_direction'] for p in result.predictions if p.get('actual_direction', -1) >= 0]
                if preds and acts:
                    fig = plot_confusion_matrix(preds, acts, title=f"{ticker} - Prediction Confusion Matrix")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("No prediction data available")
            else:
                st.warning("No predictions available")

        elif viz_option == "Returns Distribution":
            if result.returns is not None and len(result.returns) > 0:
                fig = plot_returns_distribution(result.returns, title=f"{ticker} - Returns Distribution")
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("No returns data available")

        # Detailed metrics
        st.markdown("### Detailed Metrics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Prediction Metrics")
            pred_metrics = {
                'Accuracy': ('accuracy', '{:.2%}'),
                'Precision': ('precision', '{:.2%}'),
                'Recall': ('recall', '{:.2%}'),
                'F1 Score': ('f1_score', '{:.3f}'),
                'True Positives': ('true_positives', '{:d}'),
                'True Negatives': ('true_negatives', '{:d}'),
                'False Positives': ('false_positives', '{:d}'),
                'False Negatives': ('false_negatives', '{:d}'),
            }
            for label, (key, fmt) in pred_metrics.items():
                if key in result.metrics:
                    val = result.metrics[key]
                    if isinstance(val, float):
                        st.write(f"**{label}:** {fmt.format(val)}")
                    else:
                        st.write(f"**{label}:** {val}")

        with col2:
            st.markdown("#### Trading Metrics")
            trade_metrics = {
                'Total Return': ('total_return', '{:.2%}'),
                'Annualized Return': ('annualized_return', '{:.2%}'),
                'Sharpe Ratio': ('sharpe_ratio', '{:.2f}'),
                'Sortino Ratio': ('sortino_ratio', '{:.2f}'),
                'Max Drawdown': ('max_drawdown', '{:.2%}'),
                'Win Rate': ('win_rate', '{:.2%}'),
                'Profit Factor': ('profit_factor', '{:.2f}'),
                'Avg Win': ('avg_win', '{:.2%}'),
                'Avg Loss': ('avg_loss', '{:.2%}'),
            }
            for label, (key, fmt) in trade_metrics.items():
                if key in result.metrics:
                    st.write(f"**{label}:** {fmt.format(result.metrics[key])}")

        # Trade log
        if result.trades:
            st.markdown("### Trade Log")
            trades_data = []
            for i, trade in enumerate(result.trades[:20], 1):  # Show first 20
                if isinstance(trade, dict):
                    trades_data.append({
                        '#': i,
                        'Entry': str(trade.get('entry_date', 'N/A'))[:10],
                        'Exit': str(trade.get('exit_date', 'N/A'))[:10],
                        'Signal': trade.get('entry_signal', 'N/A'),
                        'Return': f"{trade.get('return_pct', 0):.2%}",
                        'Correct': '✓' if trade.get('prediction_correct') else '✗',
                    })
                else:
                    trades_data.append({
                        '#': i,
                        'Entry': str(trade.entry_date)[:10],
                        'Exit': str(trade.exit_date)[:10],
                        'Signal': trade.entry_signal,
                        'Return': f"{trade.return_pct:.2%}",
                        'Correct': '✓' if trade.prediction_correct else '✗',
                    })

            st.dataframe(pd.DataFrame(trades_data), use_container_width=True)

            if len(result.trades) > 20:
                st.caption(f"Showing 20 of {len(result.trades)} trades")

    with tab3:
        st.markdown("### Generate Reports")

        if 'backtest_result' not in st.session_state:
            st.info("Run a backtest first to generate reports.")
            return

        result = st.session_state['backtest_result']
        ticker = st.session_state.get('backtest_ticker', 'Unknown')

        reports_dir = PROJECT_ROOT / 'reports'
        reports_dir.mkdir(exist_ok=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### HTML Report")
            st.write("Generate a comprehensive HTML report with all metrics and charts.")

            if st.button("Generate HTML Report", key="bt_html"):
                with st.spinner("Generating report..."):
                    report_path = reports_dir / f'backtest_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
                    generate_report(result, report_path, format='html')
                    st.success(f"Report saved to: {report_path}")

                    # Read and offer download
                    with open(report_path, 'r') as f:
                        html_content = f.read()

                    st.download_button(
                        label="Download HTML Report",
                        data=html_content,
                        file_name=f"backtest_{ticker}.html",
                        mime="text/html",
                        key="bt_download_html"
                    )

        with col2:
            st.markdown("#### Export Data")

            if st.button("Export Trades CSV", key="bt_csv"):
                if result.trades:
                    csv_path = reports_dir / f'trades_{ticker}_{datetime.now().strftime("%Y%m%d")}.csv'
                    export_trades_csv(result.trades, csv_path)

                    trades_df = pd.DataFrame([
                        t if isinstance(t, dict) else t.to_dict()
                        for t in result.trades
                    ])
                    csv_data = trades_df.to_csv(index=False)

                    st.download_button(
                        label="Download Trades CSV",
                        data=csv_data,
                        file_name=f"trades_{ticker}.csv",
                        mime="text/csv",
                        key="bt_download_csv"
                    )
                else:
                    st.warning("No trades to export")

            if st.button("Export Metrics JSON", key="bt_json"):
                import json
                metrics_json = json.dumps(result.metrics, indent=2, default=str)
                st.download_button(
                    label="Download Metrics JSON",
                    data=metrics_json,
                    file_name=f"metrics_{ticker}.json",
                    mime="application/json",
                    key="bt_download_json"
                )


def about_page():
    """About page."""
    st.title("ℹ️ About")

    st.markdown("""
    ## CNN Stock Market Predictor

    This application uses a 1D Convolutional Neural Network (CNN) to predict
    stock price movements based on historical OHLCV (Open, High, Low, Close, Volume) data.

    ### How It Works

    1. **Data Input**: The model uses the last 256 trading periods of price data
    2. **Feature Extraction**: 5 channels (OHLCV) are processed through 8 convolutional layers
    3. **Prediction**: The model outputs probabilities for bearish (price decrease) and bullish (price increase) outcomes
    4. **Signal Generation**: BUY if bullish probability > bearish, SELL otherwise

    ### Model Architecture

    - 8 Conv1D layers with increasing channels (128 → 1024)
    - Batch normalization and LeakyReLU activation
    - 2 fully connected layers with dropout
    - Softmax output for binary classification

    ### Prediction Horizons

    - **T+5**: Predicts price movement over the next 5 trading periods
    - **T+30**: Predicts price movement over the next 30 trading periods

    ### Phase 6 Features (Custom Data Support)

    - **CSV Upload**: Load your own OHLCV data from CSV files
    - **Multiple Timeframes**: Support for 1m, 5m, 15m, 1h, 4h, 1d, 1w data
    - **Asset Class Presets**: Pre-configured settings for stocks, crypto, forex
    - **Sample Modes**: Choose between overlapping, strided, or non-overlapping samples
    - **Flexible Column Mapping**: Auto-detection of common column name variations

    ### Phase 7 Features (Backtesting Framework)

    - **Chronological Backtest**: Run predictions on historical data without look-ahead bias
    - **Trade Simulation**: Simulate trades with realistic commission and slippage costs
    - **Performance Metrics**: Accuracy, Sharpe ratio, max drawdown, win rate, profit factor
    - **Visualization**: Equity curves, drawdown charts, confusion matrices, returns distributions
    - **Report Generation**: Export HTML reports, trade logs (CSV), and metrics (JSON)

    ### Disclaimer

    ⚠️ **This tool is for educational purposes only.**

    Predictions are based on historical patterns and should NOT be used as financial advice.
    Stock markets are inherently unpredictable, and past performance does not guarantee future results.
    Always do your own research and consult with a qualified financial advisor before making
    investment decisions.

    ### Technical Details

    - Based on: "S&P 500 Stock's Movement Prediction using CNN" (arXiv:2512.21804)
    - Framework: PyTorch
    - Data Source: Yahoo Finance (via yfinance) + Custom CSV (Phase 6)
    """)


if __name__ == "__main__":
    main()
