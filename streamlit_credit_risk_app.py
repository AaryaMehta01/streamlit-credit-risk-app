import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Upload Your Data",
    page_icon="📁",
    layout="wide"
)

# ---------------- Header ----------------
col1, col2 = st.columns([1,3])
with col1:
    st.title("Upload Your Data")
with col2:
    st.markdown("<div style='text-align:right;'>Built with SQL + Python + Power BI + Matplotlib</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("🔧 Controls")
threshold = st.sidebar.slider("Probability Threshold (accept if PD ≤ threshold)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
lgd = st.sidebar.slider("Loss Given Default (LGD)", 0.0, 1.0, 1.0, 0.05)
ead_multiplier = st.sidebar.number_input("EAD Multiplier (scale loan amounts)", value=1.0, step=0.1)

# ---------------- Data Upload ----------------
st.subheader("📥 Upload Your Scored Loans (CSV)")
st.caption("Required columns: `prob_default` (0-1), `loan_amnt` (numeric). Additional columns are welcome.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info("Please upload a CSV file to begin the analysis.")
else:
    try:
        df = pd.read_csv(uploaded)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Ensure required columns exist
    required_cols = ['prob_default', 'loan_amnt']
    if not all(col in df.columns for col in required_cols):
        st.error(f"The uploaded CSV must contain the following columns: {required_cols}. It currently has {list(df.columns)}.")
        st.stop()

    # ---------------- Main Dashboard Logic ----------------
    df["loan_amnt"] = df["loan_amnt"] * ead_multiplier
    df["expected_loss"] = df["prob_default"] * lgd * df["loan_amnt"]
    df["accept"] = (df["prob_default"] <= threshold).astype(int)

    # ---------------- KPIs ----------------
    st.markdown("### 📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_loans = len(df)
        st.metric(label="Total Loans", value=f"{total_loans:,.0f}")
    with col2:
        accepted_loans = df["accept"].sum()
        acceptance_rate = accepted_loans / total_loans * 100
        st.metric(label="Acceptance Rate", value=f"{acceptance_rate:,.1f}%")
    with col3:
        total_expected_loss = df["expected_loss"].sum()
        st.metric(label="Total Expected Loss", value=f"${total_expected_loss:,.0f}")
    with col4:
        total_accepted_loss = df[df['accept'] == 1]['expected_loss'].sum()
        st.metric(label="Expected Loss (Accepted Loans)", value=f"${total_accepted_loss:,.0f}")

    st.markdown("---")

    # ---------------- Visualizations (Plotly) ----------------
    st.markdown("### 📈 Interactive Data Visualizations")
    col_chart1, col_chart2 = st.columns(2)

    # Chart 1: Distribution of Probability of Default
    with col_chart1:
        fig1 = px.histogram(df, x="prob_default", nbins=50, title="Distribution of Probability of Default (PD)")
        fig1.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red", annotation_text=f'Threshold: {threshold:.2f}')
        fig1.update_layout(xaxis_title="Probability of Default", yaxis_title="Frequency")
        st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Expected Loss by Acceptance
    with col_chart2:
        loss_by_bucket = df.groupby(df['accept'].map({1: 'Accepted', 0: 'Rejected'}))['expected_loss'].sum().reset_index()
        fig2 = px.bar(loss_by_bucket, x='index', y='expected_loss', title='Expected Loss by Acceptance Status')
        fig2.update_layout(xaxis_title="Acceptance Status", yaxis_title="Total Expected Loss ($)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ---------------- Strategy Curve ----------------
    st.markdown("### 📉 Strategy Curve")
    st.write("Analyze the trade-off between Acceptance Rate and Expected Loss at different PD thresholds.")
    sweep = np.linspace(0.01, 1.0, 100)
    strategy_data = []
    for t in sweep:
        accepted_df = df[df['prob_default'] <= t]
        acceptance_rate = len(accepted_df) / len(df)
        expected_loss = accepted_df['expected_loss'].sum()
        strategy_data.append({'threshold': t, 'acceptance_rate': acceptance_rate, 'expected_loss': expected_loss})
    strategy_df = pd.DataFrame(strategy_data)

    fig_strategy = go.Figure()
    fig_strategy.add_trace(go.Scatter(x=strategy_df['acceptance_rate'], y=strategy_df['expected_loss'], mode='lines', name='Strategy Curve'))
    fig_strategy.update_layout(
        title='Strategy Curve: Acceptance Rate vs. Total Expected Loss',
        xaxis_title='Acceptance Rate',
        yaxis_title='Total Expected Loss ($)',
    )
    fig_strategy.add_annotation(
        x=strategy_df[strategy_df['threshold'] == threshold]['acceptance_rate'].iloc[0],
        y=strategy_df[strategy_df['threshold'] == threshold]['expected_loss'].iloc[0],
        text=f'Current Threshold: {threshold}',
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )
    st.plotly_chart(fig_strategy, use_container_width=True)

    st.markdown("---")

    # ---------------- Strategy Table ----------------
    st.markdown("#### 📋 Strategy Table (Sweep Thresholds)")
    st.write("This table shows key metrics at various probability thresholds.")
    sweep_table = np.linspace(0.1, 0.95, 18)
    rows = []
    for t in sweep_table:
        acc_rate = (df["prob_default"] <= t).mean()
        acc_df = df[df["prob_default"] <= t]
        # Calculate bad rate only for accepted loans
        bad_rate_acc = (acc_df['prob_default'] > t).mean() if not acc_df.empty else 0.0
        el_total = acc_df["expected_loss"].sum()
        rows.append({"Threshold": t, "Acceptance Rate": acc_rate, "Bad Rate (Accepted)": bad_rate_acc, "Expected Loss": el_total})
    table = pd.DataFrame(rows)
    st.dataframe(table.style.format({
        "Threshold": "{:.2f}",
        "Acceptance Rate": "{:.0%}",
        "Bad Rate (Accepted)": "{:.1%}",
        "Expected Loss": "${:,.0f}"
    }))

    st.markdown("---")

    # ---------------- Segmented Analysis ----------------
    st.markdown("### 🔍 Segmented Analysis")
    segment_by = st.selectbox("Select a variable to segment by:", ['None'] + [col for col in df.columns if df[col].dtype == 'object'])

    if segment_by != 'None':
        fig_seg = px.box(df, x=segment_by, y="prob_default", title=f"Probability of Default Distribution by {segment_by}")
        st.plotly_chart(fig_seg, use_container_width=True)

        st.markdown("#### Key Metrics by Segment")
        agg_df = df.groupby(segment_by).agg(
            total_loans=('loan_amnt', 'count'),
            total_loan_amount=('loan_amnt', 'sum'),
            total_expected_loss=('expected_loss', 'sum'),
            avg_expected_loss=('expected_loss', 'mean')
        ).reset_index()
        st.dataframe(agg_df.style.format({
            "total_loan_amount": "${:,.0f}",
            "total_expected_loss": "${:,.0f}",
            "avg_expected_loss": "${:,.2f}"
        }))
