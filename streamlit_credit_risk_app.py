import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise Credit Risk Dashboard",
    page_icon="🏢",
    layout="wide"
)

# --- Helper Functions ---
def format_currency(value):
    """Formats a number into a readable currency string."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:,.1f}M"
    else:
        return f"${value:,.0f}"

def render_enterprise_dashboard(df, page_title):
    """Renders a fully-featured enterprise-grade dashboard with advanced analytics."""

    # ---------------- Header & Introduction ----------------
    st.title(page_title)
    st.markdown("""
        <style>
            .title-divider {
                border-bottom: 2px solid #E0E0E0;
                margin-top: 0.5rem;
                margin-bottom: 1.5rem;
            }
        </style>
        <div class="title-divider"></div>
        <p style="font-size:1.1rem;">
            Welcome to the Enterprise Credit Risk Dashboard. This tool provides a comprehensive, 
            360-degree view of your loan portfolio, enabling you to manage risk and 
            make data-driven decisions.
        </p>
    """, unsafe_allow_html=True)
    
    df_original = df.copy()

    # ---------------- Sidebar Controls ----------------
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        
        with st.expander("Model Parameters", expanded=True):
            threshold = st.slider("Probability Threshold (Accept if PD ≤ threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 1.0, 0.05)
            ead_multiplier = st.number_input("EAD Multiplier (scale loan amounts)", value=1.0, step=0.1)

        with st.expander("Portfolio Filters", expanded=True):
            categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col not in ['acceptance_status']]
            numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['prob_default', 'loan_amnt', 'expected_loss', 'accept']]

            filters = {}
            for col in categorical_cols:
                unique_values = df[col].unique()
                selected_values = st.multiselect(f"Filter by {col}", unique_values, unique_values)
                filters[col] = selected_values
            
            for col in numerical_cols:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                selected_range = st.slider(f"Filter by {col}", min_val, max_val, (min_val, max_val))
                filters[col] = selected_range

    # Apply filters
    for col, values in filters.items():
        if col in categorical_cols:
            df = df[df[col].isin(values)]
        elif col in numerical_cols:
            df = df[df[col].between(values[0], values[1])]

    # ---------------- Pre-calculate metrics based on controls ----------------
    df["loan_amnt"] = df["loan_amnt"] * ead_multiplier
    df["expected_loss"] = df["prob_default"] * lgd * df["loan_amnt"]
    df["accept"] = (df["prob_default"] <= threshold).astype(int)
    df["acceptance_status"] = df["accept"].map({1: 'Accepted', 0: 'Rejected'})
    df['true_label'] = df['loan_status'] # Assuming loan_status is the ground truth
    df['prediction'] = df['accept']

    # --- Main Dashboard Content ---
    
    st.markdown("### Portfolio Health & Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        total_loans = len(df)
        st.metric(label="Total Loans", value=f"{total_loans:,.0f}")
    with col2:
        accepted_loans = df["accept"].sum()
        acceptance_rate = accepted_loans / total_loans * 100 if total_loans > 0 else 0
        st.metric(label="Acceptance Rate", value=f"{acceptance_rate:,.1f}%")
    with col3:
        total_loan_amount = df["loan_amnt"].sum()
        st.metric(label="Total Loan Amount", value=format_currency(total_loan_amount))
    with col4:
        total_expected_loss = df["expected_loss"].sum()
        st.metric(label="Total Expected Loss", value=format_currency(total_expected_loss))
    with col5:
        el_accepted = df[df['accept'] == 1]['expected_loss'].sum()
        st.metric(label="EL (Accepted)", value=format_currency(el_accepted))

    st.markdown("---")

    col_vis1, col_vis2 = st.columns(2)
    with col_vis1:
        st.subheader("Distribution of Probability of Default (PD)")
        fig1 = px.histogram(df, x="prob_default", nbins=50, color='acceptance_status',
                            title="PD Distribution by Acceptance Status")
        fig1.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red", annotation_text=f'Threshold: {threshold:.2f}')
        fig1.update_layout(xaxis_title="Probability of Default", yaxis_title="Number of Loans")
        st.plotly_chart(fig1, use_container_width=True)

    with col_vis2:
        st.subheader("Expected Loss by Acceptance Status")
        loss_by_bucket = df.groupby('acceptance_status')['expected_loss'].sum().reset_index()
        fig2 = px.bar(loss_by_bucket, x='acceptance_status', y='expected_loss', title='Total Expected Loss by Acceptance Status')
        fig2.update_layout(xaxis_title="Acceptance Status", yaxis_title="Total Expected Loss ($)")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Portfolio Segmentation")
    
    col_seg1, col_seg2 = st.columns(2)
    with col_seg1:
        segment_by_1 = st.selectbox("Segment Portfolio by:", ['None'] + categorical_cols, key='seg_1')
        if segment_by_1 != 'None':
            fig_seg = px.box(df, x=segment_by_1, y="prob_default", 
                             title=f"Probability of Default Distribution by {segment_by_1}",
                             color=segment_by_1)
            st.plotly_chart(fig_seg, use_container_width=True)
    
    with col_seg2:
        segment_by_2 = st.selectbox("Segment Portfolio by:", ['None'] + categorical_cols, key='seg_2')
        if segment_by_2 != 'None':
            fig_seg = px.box(df, x=segment_by_2, y="loan_amnt", 
                             title=f"Loan Amount Distribution by {segment_by_2}",
                             color=segment_by_2)
            st.plotly_chart(fig_seg, use_container_width=True)

    # ---------------- Strategic Analysis & Performance ----------------
    st.markdown("---")
    st.header("Strategic Analysis & Model Performance")

    # Strategy Table
    st.subheader("Strategy Table: Impact of PD Threshold")
    sweep = np.linspace(0.1, 0.95, 18)
    rows = []
    for t in sweep:
        acc_count = (df_original["prob_default"] <= t).sum()
        total_count = len(df_original)
        acc_rate = acc_count / total_count if total_count > 0 else 0
        
        accepted_df = df_original[df_original["prob_default"] <= t]
        rejected_df = df_original[df_original["prob_default"] > t]
        
        # Calculate expected loss for accepted loans only
        el_accepted = (accepted_df["prob_default"] * lgd * accepted_df["loan_amnt"]).sum()
        
        # Calculate bad rate for accepted loans
        bad_rate = (accepted_df['loan_status'] == 1).sum() / acc_count if acc_count > 0 else 0
        
        rows.append({
            "PD Threshold": t, 
            "Acceptance Rate": acc_rate, 
            "Bad Rate (Accepted)": bad_rate, 
            "Total Expected Loss (Accepted)": el_accepted
        })
    
    strategy_table_df = pd.DataFrame(rows)
    st.dataframe(strategy_table_df.style.format({
        "PD Threshold": "{:.2f}",
        "Acceptance Rate": "{:.1%}",
        "Bad Rate (Accepted)": "{:.1%}",
        "Total Expected Loss (Accepted)": "${:,.0f}"
    }), use_container_width=True)

    # Confusion Matrix
    if not df.empty:
        try:
            st.subheader("Confusion Matrix")
            y_true = df['true_label']
            y_pred = df['prediction']
            
            cm = confusion_matrix(y_true, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted Rejected', 'Predicted Accepted'],
                y=['Actual Good Loan', 'Actual Bad Loan'],
                colorscale='Viridis',
                hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted Outcome',
                yaxis_title='Actual Outcome',
                xaxis={'side': 'bottom'},
                yaxis={'side': 'left'}
            )
            
            for i in range(len(cm)):
                for j in range(len(cm[0])):
                    fig_cm.add_annotation(
                        x=fig_cm.data[0].x[j],
                        y=fig_cm.data[0].y[i],
                        text=str(cm[i][j]),
                        showarrow=False,
                        font=dict(color="white", size=16)
                    )
            
            st.plotly_chart(fig_cm, use_container_width=True)

            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            col_metrics1.metric("Precision", f"{precision:.2f}")
            col_metrics2.metric("Recall", f"{recall:.2f}")
            col_metrics3.metric("Accuracy", f"{accuracy:.2f}")

        except ValueError:
            st.warning("Cannot generate a confusion matrix. The 'loan_status' column might contain values other than 0 or 1.")
        
    st.markdown("### Your Insights")
    st.write("Use this space to document your key insights from the dashboard and analysis.")
    st.text_area("Write your notes here...", height=200, key="insights_text_area")

# --- Main Page Content ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Main Dashboard", "Upload Your Data", "About"])

if page == "Main Dashboard":
    try:
        # Load sample data
        df = pd.read_csv("cr_loan_clean.csv")
        # Ensure correct data types for calculation
        df['prob_default'] = df['loan_status']
        df['loan_amnt'] = df['loan_amnt']
        render_enterprise_dashboard(df.copy(), "Main Dashboard")
    except FileNotFoundError:
        st.error("Sample data file 'cr_loan_clean.csv' not found. Please make sure it's in the same directory as this script.")

elif page == "Upload Your Data":
    st.title("Upload Your Data")
    st.markdown("---")
    st.subheader("📥 Upload Your CSV File")
    st.caption("The dashboard requires a **'loan_amnt'** column and a **'prob_default'** (or binary **'loan_status'**) column. Additional columns will be used for filtering.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.info("Please upload a CSV file to begin the analysis.")
    else:
        try:
            df = pd.read_csv(uploaded)
            st.success("File uploaded successfully! The dashboard is now updated with your data.")
            
            if 'prob_default' not in df.columns and 'loan_status' not in df.columns:
                st.warning("Could not find a 'prob_default' or 'loan_status' column. Please select it below.")
                prob_col = st.selectbox("Select the Probability of Default/Loan Status column", options=[None] + list(df.columns))
                if prob_col:
                    df['prob_default'] = df[prob_col]
                else:
                    st.stop()
            elif 'loan_status' in df.columns and 'prob_default' not in df.columns:
                df['prob_default'] = df['loan_status']
            
            if 'loan_amnt' not in df.columns:
                st.warning("Could not find a 'loan_amnt' column. Please select it below.")
                loan_col = st.selectbox("Select the Loan Amount column", options=[None] + list(df.columns))
                if loan_col:
                    df['loan_amnt'] = df[loan_col]
                else:
                    st.stop()
                    
            if not pd.api.types.is_numeric_dtype(df['prob_default']):
                st.error("The 'prob_default' column must be numeric. Please check your data.")
                st.stop()
            if not pd.api.types.is_numeric_dtype(df['loan_amnt']):
                st.error("The 'loan_amnt' column must be numeric. Please check your data.")
                st.stop()
            
            render_enterprise_dashboard(df.copy(), "Uploaded Data Dashboard")
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "About":
    st.title("About this Project")
    st.markdown("---")
    st.write(
        """
        This Streamlit application was developed as an enterprise-grade tool for performing quick and insightful credit risk analysis. 
        It is designed to be a professional, user-friendly, and interactive platform for financial analysts, data scientists, and students 
        to explore key concepts in credit risk management.
        """
    )
    st.header("Behind the Scenes")
    st.markdown(
        """
        The application is built using **Python** and the **Streamlit** library. The visualizations are created with **Plotly**,
        which enables the interactive charts you see on the dashboards. The underlying data manipulation is handled by **Pandas** and **NumPy**.
        """
    )
    st.markdown(
        """
        This project demonstrates the power of these tools in creating powerful, accessible, and enterprise-grade data applications.
        """
    )
    st.markdown("---")
    st.markdown("Thank you for using this application!")
