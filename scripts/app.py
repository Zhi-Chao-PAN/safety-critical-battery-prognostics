"""
PINN Battery RUL Prediction Dashboard.

An interactive Streamlit web application for industrial-grade battery remaining
useful life (RUL) prediction using Physics-Informed Neural Networks (PINN).

Features:
    - Interactive sidebar for parameter adjustment
    - Real-time prediction with uncertainty quantification
    - Plotly-based interactive visualizations
    - Safety status indicators (Green/Yellow/Red)
    - Support for NASA/CALCE data format

Author: AI Engineer
Date: 2026-04-04
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.subplots import make_subplots

# Import project-specific modules
from src.infrastructure.config_schema import PINNConfig, load_config
from src.models.pinn_model import PINNModel
from src.data.unified_loader import UnifiedDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PINN Battery RUL Prediction Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .status-green {
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .status-yellow {
        background: linear-gradient(135deg, #fdcb6e, #e67e22);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .status-red {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    .metric-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class DashboardSessionState:
    """Manages session state for the dashboard."""

    @staticmethod
    def init_session_state():
        """Initialize session state variables."""
        if "config" not in st.session_state:
            st.session_state.config = None
        if "model" not in st.session_state:
            st.session_state.model = None
        if "data_loaded" not in st.session_state:
            st.session_state.data_loaded = False
        if "predictions" not in st.session_state:
            st.session_state.predictions = None
        if "safety_status" not in st.session_state:
            st.session_state.safety_status = "green"


class DataLoader:
    """Wrapper for loading battery data from various sources."""

    def __init__(self, config: PINNConfig):
        self.config = config
        self.loader = UnifiedDataLoader(
            data_dir=config.data.data_dir,
            dataset_names=config.data.datasets,
            batch_size=config.data.batch_size,
            val_split=config.data.val_fraction,
            test_split=config.data.test_fraction,
        )

    def load_from_upload(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file (CSV format).

        Args:
            uploaded_file: Streamlit uploaded file object.

        Returns:
            DataFrame with battery data or None if loading fails.
        """
        try:
            df = pd.read_csv(uploaded_file)
            # Validate required columns
            required_cols = ["cycle", "capacity"]
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None

    def generate_synthetic_data(
        self,
        n_samples: int = 100,
        rated_capacity: float = 2.0,
        eol_fraction: float = 0.7,
        noise_level: float = 0.02,
    ) -> pd.DataFrame:
        """
        Generate synthetic battery degradation data for demonstration.

        Args:
            n_samples: Number of data points to generate.
            rated_capacity: Rated capacity of the battery (Ah).
            eol_fraction: End-of-life capacity fraction.
            noise_level: Measurement noise level.

        Returns:
            DataFrame with synthetic battery data.
        """
        cycles = np.linspace(0, 1000, n_samples)

        # Exponential degradation model
        eol_cycle = 1000
        alpha = -np.log(eol_fraction) / eol_cycle
        capacity = rated_capacity * np.exp(-alpha * cycles)

        # Add noise
        noise = np.random.normal(0, noise_level * rated_capacity, n_samples)
        capacity_noisy = capacity + noise

        df = pd.DataFrame({
            "cycle": cycles.astype(int),
            "capacity": capacity_noisy,
            "voltage": 3.7 + 0.1 * np.sin(cycles / 100),  # Synthetic voltage
            "current": np.ones(n_samples) * 2.0,  # Constant current
        })

        return df


class Predictor:
    """Handles model prediction with uncertainty quantification."""

    def __init__(self, model: PINNModel):
        self.model = model

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        return_full: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions with uncertainty quantification.

        Args:
            X: Input features [n_samples, n_features]
            return_full: Whether to return full MC samples

        Returns:
            Dictionary with mean, std, confidence intervals, and optionally full samples
        """
        result = self.model.predict_single(X)

        if not return_full:
            # Remove full samples to save memory
            result.pop("samples", None)

        return result


class Visualizer:
    """Creates interactive Plotly visualizations."""

    @staticmethod
    def create_prediction_plot(
        df: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        rated_capacity: float,
        eol_fraction: float,
    ) -> go.Figure:
        """
        Create main prediction plot with uncertainty bands.

        Args:
            df: DataFrame with actual data
            predictions: Dictionary with prediction results
            rated_capacity: Rated capacity for EOL line
            eol_fraction: End-of-life fraction

        Returns:
            Plotly Figure object
        """
        cycles = df["cycle"].values
        actual = df["capacity"].values
        mean_pred = predictions["mean"]
        lower_95 = predictions["lower_95"]
        upper_95 = predictions["upper_95"]

        fig = go.Figure()

        # Actual data points
        fig.add_trace(go.Scatter(
            x=cycles,
            y=actual,
            mode="markers",
            name="Actual Data",
            marker=dict(color="black", size=6, opacity=0.7),
            hovertemplate="Cycle: %{x}<br>Capacity: %{y:.3f} Ah<extra></extra>",
        ))

        # Mean prediction
        fig.add_trace(go.Scatter(
            x=cycles,
            y=mean_pred,
            mode="lines",
            name="PINN Prediction",
            line=dict(color="#1f77b4", width=3),
            hovertemplate="Cycle: %{x}<br>Predicted: %{y:.3f} Ah<extra></extra>",
        ))

        # 95% Confidence interval (upper bound)
        fig.add_trace(go.Scatter(
            x=cycles,
            y=upper_95,
            mode="lines",
            name="95% CI Upper",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))

        # 95% Confidence interval (fill)
        fig.add_trace(go.Scatter(
            x=cycles,
            y=lower_95,
            mode="lines",
            name="95% Confidence Interval",
            line=dict(width=0),
            fillcolor="rgba(31, 119, 180, 0.3)",
            fill="tonexty",
            hovertemplate="Cycle: %{x}<br>Lower: %{y:.3f} Ah<extra></extra>",
        ))

        # End-of-Life threshold line
        eol_capacity = rated_capacity * eol_fraction
        fig.add_hline(
            y=eol_capacity,
            line_dash="dash",
            line_color="red",
            annotation_text=f"EOL Threshold ({eol_fraction*100:.0f}%)",
            annotation_position="right",
        )

        # Layout
        fig.update_layout(
            title=dict(
                text="Battery Capacity Degradation Prediction with Uncertainty Quantification",
                font=dict(size=18),
                x=0.5,
            ),
            xaxis=dict(
                title="Cycle Number",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
            ),
            yaxis=dict(
                title="Capacity (Ah)",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128,128,128,0.2)",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
            template="plotly_white",
            height=600,
        )

        return fig

    @staticmethod
    def create_rul_gauge(rul: float, threshold_warning: float, threshold_critical: float) -> go.Figure:
        """
        Create RUL gauge chart.

        Args:
            rul: Predicted RUL value
            threshold_warning: Warning threshold
            threshold_critical: Critical threshold

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rul,
            domain=dict(x=[0, 1], y=[0, 1]),
            title=dict(text="RUL (Cycles)", font=dict(size=24)),
            delta=dict(reference=threshold_warning, increasing=dict(color="red")),
            gauge=dict(
                axis=dict(range=[0, max(rul * 1.5, threshold_warning * 2)], tickwidth=1),
                bar=dict(color="darkblue"),
                bgcolor="white",
                borderwidth=2,
                bordercolor="gray",
                steps=[
                    dict(range=[0, threshold_critical], color="rgba(231, 76, 60, 0.3)"),
                    dict(range=[threshold_critical, threshold_warning], color="rgba(253, 203, 110, 0.3)"),
                    dict(range=[threshold_warning, max(rul * 1.5, threshold_warning * 2)], color="rgba(0, 184, 148, 0.3)"),
                ],
                threshold=dict(
                    line=dict(color="red", width=4),
                    thickness=0.75,
                    value=threshold_critical,
                ),
            ),
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        return fig


class SafetyMonitor:
    """Monitors safety status based on RUL predictions."""

    @staticmethod
    def determine_safety_status(
        rul: float,
        threshold_warning: float,
        threshold_critical: float,
    ) -> Tuple[str, str]:
        """
        Determine safety status based on RUL.

        Args:
            rul: Predicted RUL value
            threshold_warning: Warning threshold
            threshold_critical: Critical threshold

        Returns:
            Tuple of (status_class, status_text)
        """
        if rul <= threshold_critical:
            return "status-red", "🔴 CRITICAL - Immediate Replacement Required"
        elif rul <= threshold_warning:
            return "status-yellow", "🟡 WARNING - Schedule Maintenance Soon"
        else:
            return "status-green", "🟢 NORMAL - Battery Healthy"

    @staticmethod
    def calculate_rul(
        current_cycle: int,
        predictions: np.ndarray,
        eol_threshold: float,
    ) -> float:
        """
        Calculate Remaining Useful Life (RUL).

        Args:
            current_cycle: Current cycle number
            predictions: Predicted capacity values
            eol_threshold: End-of-life capacity threshold

        Returns:
            RUL in cycles
        """
        # Find when predicted capacity falls below EOL threshold
        cycles = np.arange(current_cycle, current_cycle + len(predictions))
        below_threshold = predictions < eol_threshold

        if np.any(below_threshold):
            eol_cycle = cycles[below_threshold][0]
            return float(eol_cycle - current_cycle)
        else:
            # If no EOL in prediction horizon, extrapolate
            return float(len(predictions))


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "config" not in st.session_state:
        st.session_state.config = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "predictions" not in st.session_state:
        st.session_state.predictions = None
    if "safety_status" not in st.session_state:
        st.session_state.safety_status = "green"


def render_sidebar(config: PINNConfig) -> Dict[str, Any]:
    """
    Render sidebar with interactive controls.

    Args:
        config: PINN configuration object

    Returns:
        Dictionary of user inputs
    """
    with st.sidebar:
        st.markdown("## 🔧 Control Panel")
        st.markdown("---")

        # Physics Parameters Section
        st.markdown("### ⚛️ Physics Parameters")

        rated_capacity = st.slider(
            "Rated Capacity (Ah)",
            min_value=0.5,
            max_value=5.0,
            value=config.physics.rated_capacity,
            step=0.1,
            help="Battery rated capacity in Ampere-hours",
        )

        eol_fraction = st.slider(
            "EOL Fraction",
            min_value=0.5,
            max_value=0.9,
            value=config.physics.eol_fraction,
            step=0.05,
            help="End-of-life capacity as fraction of rated capacity",
        )

        st.markdown("---")

        # Prediction Horizon Section
        st.markdown("### 🔮 Prediction Horizon")

        prediction_cycles = st.slider(
            "Future Cycles to Predict",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
            help="Number of future cycles to predict",
        )

        st.markdown("---")

        # Safety Thresholds Section
        st.markdown("### 🛡️ Safety Thresholds")

        warning_threshold = st.slider(
            "Warning Threshold (cycles)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="RUL threshold for warning status",
        )

        critical_threshold = st.slider(
            "Critical Threshold (cycles)",
            min_value=5,
            max_value=50,
            value=10,
            step=5,
            help="RUL threshold for critical status",
        )

        st.markdown("---")

        # Data Upload Section
        st.markdown("### 📁 Data Upload")

        uploaded_file = st.file_uploader(
            "Upload Battery Data (CSV)",
            type=["csv"],
            help="Upload CSV with columns: cycle, capacity, voltage, current",
        )

        use_synthetic = st.checkbox(
            "Use Synthetic Data",
            value=True,
            help="Generate synthetic data if no file uploaded",
        )

        st.markdown("---")

        # Action Buttons
        st.markdown("### 🚀 Actions")

        col1, col2 = st.columns(2)
        with col1:
            run_prediction = st.button("▶️ Run Prediction", use_container_width=True)
        with col2:
            reset_app = st.button("🔄 Reset", use_container_width=True)

        return {
            "rated_capacity": rated_capacity,
            "eol_fraction": eol_fraction,
            "prediction_cycles": prediction_cycles,
            "warning_threshold": warning_threshold,
            "critical_threshold": critical_threshold,
            "uploaded_file": uploaded_file,
            "use_synthetic": use_synthetic,
            "run_prediction": run_prediction,
            "reset_app": reset_app,
        }


def load_data(
    inputs: Dict[str, Any],
    config: PINNConfig,
) -> Optional[pd.DataFrame]:
    """
    Load battery data from upload or generate synthetic data.

    Args:
        inputs: User inputs from sidebar
        config: PINN configuration

    Returns:
        DataFrame with battery data or None
    """
    data_loader = DataLoader(config)

    if inputs["uploaded_file"] is not None:
        df = data_loader.load_from_upload(inputs["uploaded_file"])
        if df is not None:
            st.success(f"✅ Loaded {len(df)} data points from uploaded file")
            return df

    if inputs["use_synthetic"]:
        df = data_loader.generate_synthetic_data(
            n_samples=200,
            rated_capacity=inputs["rated_capacity"],
            eol_fraction=inputs["eol_fraction"],
        )
        st.info(f"ℹ️ Generated {len(df)} synthetic data points for demonstration")
        return df

    return None


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Prepare feature matrix from DataFrame.

    Args:
        df: DataFrame with battery data

    Returns:
        Feature matrix X
    """
    # Basic features: cycle number and capacity
    features = ["cycle", "capacity"]

    # Add optional features if available
    for col in ["voltage", "current"]:
        if col in df.columns:
            features.append(col)

    X = df[features].values.astype(np.float32)

    # Pad to 2 features if needed (minimum for PINN)
    if X.shape[1] < 2:
        padding = np.zeros((X.shape[0], 2 - X.shape[1]), dtype=np.float32)
        X = np.concatenate([X, padding], axis=1)

    return X


def run_prediction_pipeline(
    df: pd.DataFrame,
    inputs: Dict[str, Any],
    config: PINNConfig,
) -> Optional[Dict[str, Any]]:
    """
    Run the complete prediction pipeline.

    Args:
        df: DataFrame with battery data
        inputs: User inputs
        config: PINN configuration

    Returns:
        Dictionary with prediction results or None
    """
    try:
        with st.spinner("🔄 Initializing PINN model..."):
            # Initialize model
            model = PINNModel(**config.get_pinn_model_kwargs())

        with st.spinner("📊 Preparing features..."):
            X = prepare_features(df)
            y = df["capacity"].values.astype(np.float32)

        with st.spinner("🎯 Training model..."):
            # Train model
            model.fit(X, y)

        with st.spinner("🔮 Generating predictions with uncertainty..."):
            # Generate predictions
            mean, lower, upper = model.predict(X)

            # Get detailed uncertainty
            detailed = model.predict_single(X)

        # Calculate RUL
        current_cycle = int(df["cycle"].max())
        eol_threshold = inputs["rated_capacity"] * inputs["eol_fraction"]
        rul = SafetyMonitor.calculate_rul(current_cycle, mean, eol_threshold)

        # Determine safety status
        status_class, status_text = SafetyMonitor.determine_safety_status(
            rul,
            inputs["warning_threshold"],
            inputs["critical_threshold"],
        )

        return {
            "model": model,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "detailed": detailed,
            "rul": rul,
            "status_class": status_class,
            "status_text": status_text,
            "current_cycle": current_cycle,
        }

    except Exception as e:
        st.error(f"❌ Prediction pipeline failed: {e}")
        logger.exception("Prediction pipeline error")
        return None


def render_header():
    """Render the application header."""
    st.markdown('<div class="main-header">🔋 PINN Battery RUL Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Physics-Informed Neural Network for Safety-Critical Battery Prognostics</div>',
        unsafe_allow_html=True
    )


def render_results(results: Dict[str, Any], df: pd.DataFrame, inputs: Dict[str, Any]):
    """
    Render prediction results with visualizations.

    Args:
        results: Prediction results dictionary
        df: Original data DataFrame
        inputs: User inputs
    """
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    # Safety Status Card
    st.markdown(f'<div class="{results["status_class"]}">{results["status_text"]}</div>', unsafe_allow_html=True)

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Predicted RUL",
            value=f"{results['rul']:.0f} cycles",
            delta=f"Current: {results['current_cycle']}",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        uncertainty = results["detailed"]["std"].mean()
        st.metric(
            label="Mean Uncertainty",
            value=f"±{uncertainty:.3f} Ah",
            delta="95% CI",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        current_capacity = df["capacity"].iloc[-1]
        st.metric(
            label="Current Capacity",
            value=f"{current_capacity:.3f} Ah",
            delta=f"{current_capacity/inputs['rated_capacity']*100:.1f}%",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        capacity_retention = current_capacity / inputs["rated_capacity"]
        st.metric(
            label="Capacity Retention",
            value=f"{capacity_retention*100:.1f}%",
            delta=None,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Main Prediction Plot
    st.markdown("---")
    st.markdown("### 📈 Capacity Degradation with Uncertainty")

    fig = Visualizer.create_prediction_plot(
        df,
        results["detailed"],
        inputs["rated_capacity"],
        inputs["eol_fraction"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # RUL Gauge
    st.markdown("---")
    st.markdown("### 🎯 RUL Gauge")

    fig_gauge = Visualizer.create_rul_gauge(
        results["rul"],
        inputs["warning_threshold"],
        inputs["critical_threshold"],
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Confidence Interval Details
    st.markdown("---")
    st.markdown("### 📋 Detailed Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Uncertainty Quantification (95% CI)**")
        stats_df = pd.DataFrame({
            "Metric": ["Mean Prediction", "Lower Bound (95%)", "Upper Bound (95%)", "Uncertainty (Std)"],
            "Value": [
                f"{results['detailed']['mean'].mean():.4f} Ah",
                f"{results['detailed']['lower_95'].mean():.4f} Ah",
                f"{results['detailed']['upper_95'].mean():.4f} Ah",
                f"{results['detailed']['std'].mean():.4f} Ah",
            ],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Safety Assessment**")
        safety_df = pd.DataFrame({
            "Parameter": ["Predicted RUL", "Current Cycle", "Warning Threshold", "Critical Threshold", "Status"],
            "Value": [
                f"{results['rul']:.0f} cycles",
                f"{results['current_cycle']}",
                f"{inputs['warning_threshold']:.0f} cycles",
                f"{inputs['critical_threshold']:.0f} cycles",
                results["status_text"].split(" - ")[0],
            ],
        })
        st.dataframe(safety_df, use_container_width=True, hide_index=True)


def main():
    """Main application entry point."""
    # Initialize session state
    DashboardSessionState.init_session_state()

    # Render header
    render_header()

    # Load configuration
    try:
        if st.session_state.config is None:
            config = load_config("configs/pinn_config.yaml")
            st.session_state.config = config
            logger.info("Configuration loaded successfully")
        else:
            config = st.session_state.config
    except Exception as e:
        st.error(f"❌ Failed to load configuration: {e}")
        logger.exception("Configuration loading error")
        return

    # Render sidebar and get user inputs
    inputs = render_sidebar(config)

    # Handle reset
    if inputs["reset_app"]:
        st.session_state.clear()
        st.rerun()

    # Info box
    st.markdown("""
    <div class="info-box">
        <strong>📖 How to Use:</strong><br>
        1. Adjust physics parameters and safety thresholds in the sidebar<br>
        2. Upload your battery data or use synthetic data<br>
        3. Click "Run Prediction" to generate RUL predictions with uncertainty<br>
        4. Monitor the safety status and confidence intervals
    </div>
    """, unsafe_allow_html=True)

    # Load data
    if inputs["run_prediction"] or st.session_state.data_loaded:
        data_loader = DataLoader(config)

        if not st.session_state.data_loaded:
            df = load_data(inputs, config)
            if df is not None:
                st.session_state.data = df
                st.session_state.data_loaded = True
            else:
                st.error("❌ Failed to load data. Please check your input.")
                return
        else:
            df = st.session_state.data

        # Run prediction pipeline
        if inputs["run_prediction"] or st.session_state.predictions is None:
            with st.spinner("🔄 Running PINN prediction pipeline..."):
                results = run_prediction_pipeline(df, inputs, config)
                if results is not None:
                    st.session_state.predictions = results
                    st.success("✅ Prediction completed successfully!")
                else:
                    st.error("❌ Prediction failed. Please check the logs.")
                    return

        # Render results
        if st.session_state.predictions is not None:
            render_results(st.session_state.predictions, df, inputs)

    else:
        # Show placeholder
        st.info("👈 Configure parameters in the sidebar and click 'Run Prediction' to start.")

        # Sample visualization placeholder
        st.markdown("### 📊 Sample Visualization")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 200, 400, 600, 800, 1000],
            y=[2.0, 1.9, 1.7, 1.5, 1.3, 1.1],
            mode="lines+markers",
            name="Capacity Fade Curve",
            line=dict(color="#1f77b4", width=3),
        ))
        fig.add_hline(y=1.4, line_dash="dash", line_color="red", annotation_text="EOL Threshold")
        fig.update_layout(
            title="Sample Battery Degradation Curve",
            xaxis_title="Cycle Number",
            yaxis_title="Capacity (Ah)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
