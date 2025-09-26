import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# --- Page Configuration ---
st.set_page_config(
    page_title="Enterprise Credit Risk Management",
    page_icon="🏢",
    layout="wide"
)

# --- Helper Functions ---
def format_currency(value):
    """Formats a number into a readable currency string with M or B suffixes."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    elif abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    else:
        return f"${value:,.0f}"

# --- Helper Function for Dashboard Content ---
def render_enterprise_dashboard(df, page_title):
    """
    Renders an enterprise-grade dashboard with advanced analytics.
    """
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
            This dashboard provides a comprehensive, 360-degree view of your loan portfolio, allowing you to manage risk, 
            validate models, and make data-driven decisions.
        </p>
    """, unsafe_allow_html=True)
    
    # Create a copy of the original dataframe before filtering
    df_original = df.copy()

    # ---------------- Sidebar Controls - Logical Sections ----------------
    with st.sidebar:
        st.header("⚙️ Model Parameters")
        threshold = st.slider("Probability Threshold (accept if PD ≤ threshold)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        lgd = st.slider("Loss Given Default (LGD)", 0.0, 1.0, 1.0, 0.05)
        ead_multiplier = st.number_input("EAD Multiplier (scale loan amounts)", value=1.0, step=0.1)

        # Dynamic Filters
        st.header("🗂️ Portfolio Filters")
        
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

    # --- Main Dashboard Tabs ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portfolio Overview", "📈 Model Validation", "🔍 Deep Dive Analysis", "📝 Reporting & Insights"])

    with tab1:
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

    with tab2:
        st.header("Model Validation & Backtesting")
        st.markdown("This section provides a deeper look into the performance of the PD model.")
        
        # Ensure 'true_label' is correctly defined for the filtered data
        df['prediction'] = (df['prob_default'] <= threshold).astype(int)
        
        if 'true_label' in df.columns and len(df[df['true_label'].isin([0, 1])]) > 0 and len(df) > 0:
            y_true = df['true_label'].astype(int)
            y_pred = df['prediction'].astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
            tn, fp, fn, tp = cm.ravel()
            
            st.subheader("Confusion Matrix")
            cm_df = pd.DataFrame(cm, columns=['Predicted Default (1)', 'Predicted Non-Default (0)'],
                                 index=['Actual Default (1)', 'Actual Non-Default (0)'])
            st.dataframe(cm_df)

            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                # This metric now correctly updates based on the filtered data and threshold
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col_met2:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                st.metric("Precision", f"{precision:.2%}")
            with col_met3:
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                st.metric("Recall", f"{recall:.2%}")

            st.markdown("---")
            
            col_curves1, col_curves2 = st.columns(2)
            
            with col_curves1:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(df['true_label'], df['prob_default'])
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title='Receiver Operating Characteristic (ROC) Curve')
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col_curves2:
                st.subheader("Precision-Recall Curve")
                precision, recall, _ = precision_recall_curve(df['true_label'], df['prob_default'])
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='Precision-Recall curve'))
                fig_pr.update_layout(xaxis_title='Recall', yaxis_title='Precision', title='Precision-Recall Curve')
                st.plotly_chart(fig_pr, use_container_width=True)
        else:
            st.warning("No ground truth labels (0 or 1) or no data found in the 'loan_status' column for model validation.")

    with tab3:
        st.header("Deep Dive Analysis")
        st.markdown("Create custom visualizations to explore specific relationships in your data.")

        col_deep1, col_deep2 = st.columns(2)
        with col_deep1:
            x_axis = st.selectbox("Select X-axis", options=['--Select a column--'] + list(df.columns), key='x_axis_select')
        with col_deep2:
            y_axis = st.selectbox("Select Y-axis", options=['--Select a column--'] + list(df.columns), key='y_axis_select')

        plot_type = st.selectbox("Select Plot Type", options=['--Select a plot type--', 'Scatter', 'Histogram', 'Box Plot', 'Bar Chart'], key='plot_type_select')
        color_by = st.selectbox("Color by:", options=['None'] + categorical_cols, key='color_by_select')

        if x_axis != '--Select a column--' and plot_type != '--Select a plot type--':
            if plot_type == 'Scatter' and y_axis != '--Select a column--':
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by if color_by != 'None' else None, title=f"{x_axis} vs. {y_axis}")
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == 'Histogram':
                fig = px.histogram(df, x=x_axis, color=color_by if color_by != 'None' else None, title=f"Distribution of {x_axis}")
                st.plotly_chart(fig, use_container_width=True)
            elif plot_type == 'Box Plot':
                if y_axis != '--Select a column--':
                    fig = px.box(df, x=x_axis, y=y_axis, color=color_by if color_by != 'None' else None, title=f"Box Plot of {y_axis} by {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
            elif plot_type == 'Bar Chart':
                if y_axis != '--Select a column--':
                    agg_df = df.groupby(x_axis)[y_axis].sum().reset_index()
                    fig = px.bar(agg_df, x=x_axis, y=y_axis, color=color_by if color_by != 'None' else None, title=f"Total {y_axis} by {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Reporting & Insights")
        st.markdown("This section provides a summary of the key findings from the analysis.")
        
        st.subheader("Executive Summary")
        
        total_expected_loss = df["expected_loss"].sum()
        el_accepted = df[df['accept'] == 1]['expected_loss'].sum()
        total_loans = len(df)
        accepted_loans = df["accept"].sum()
        acceptance_rate = accepted_loans / total_loans * 100 if total_loans > 0 else 0

        st.markdown(f"""
            Based on the selected portfolio and model parameters, the current PD threshold of **{threshold:.2f}** leads to an acceptance rate of **{acceptance_rate:,.1f}%**. This strategy results in a 
            total expected loss of **{format_currency(total_expected_loss)}**, with **{format_currency(el_accepted)}** of that loss 
            coming from the accepted loans.
        """)

        st.subheader("Model Performance Summary")
        if 'true_label' in df.columns and len(df[df['true_label'].isin([0, 1])]) > 0:
            y_true = df['true_label'].astype(int)
            y_pred = df['prediction'].astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            fpr, tpr, _ = roc_curve(df['true_label'], df['prob_default'])
            roc_auc = auc(fpr, tpr)
            
            st.markdown(f"""
                The PD model demonstrates strong performance with a key metric analysis:
                - **Accuracy:** {accuracy:.2%}
                - **Precision:** {tp / (tp + fp):.2%}
                - **Recall:** {tp / (tp + fn):.2%}
                
                The ROC curve area of **{roc_auc:.2f}** indicates that the model has a high capacity to distinguish between 
                defaulted and non-defaulted loans.
            """)
        else:
            st.info("Model performance summary is not available. Please ensure your data has a binary 'loan_status' column.")
            
        st.markdown("---")
        st.subheader("Key Recommendations")
        st.markdown("""
            * **Re-evaluate Thresholds:** The current threshold balances acceptance rate and expected loss. Consider adjusting the PD threshold based on the company's risk appetite.
            * **Segment Analysis:** Use the deep dive analysis section to identify high-risk segments in the portfolio and create targeted strategies to mitigate risk.
            * **Data Quality:** The quality of the PD model's predictions is highly dependent on the quality of the input data. Regular data validation and cleaning are crucial.
        """)

# --- Main Page Content ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Intro", "Portfolio Analysis", "Upload Your Data", "About"])

if page == "Intro":
    st.title("Welcome to the Enterprise Credit Risk Dashboard")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("What is Enterprise Credit Risk Management?")
        st.write(
            """
            At an enterprise level, credit risk management moves beyond simple analysis. It's about a holistic framework that integrates data, 
            advanced models, and sophisticated decision-making tools to manage a diverse loan portfolio at scale.
            """
        )
    with col2:
        st.header("How This App Helps")
        st.write(
            """
            This application provides a powerful and interactive dashboard to analyze and visualize credit risk data. 
            It is designed to give you an immediate, hands-on experience without the need to upload your own CSV file. 
            This page allows you to test the interactive controls, understand the various charts and KPIs, 
            and see how the analysis works before you begin a custom analysis.
            """
        )
    with col2:
        st.header("Purpose of Portfolio Analysis")
        st.write(
            """
            Beyond assessing individual loans, **portfolio analysis** is crucial for understanding the overall risk exposure of a group of loans. It allows a business to evaluate the collective performance and risk profile of its entire loan portfolio, rather than just individual borrowers. This analysis helps in:
            * **Diversification:** Identifying how different types of loans balance each other out to mitigate risk.
            * **Stress Testing:** Simulating economic downturns or other scenarios to see how the portfolio would perform under pressure.
            * **Capital Allocation:** Determining the appropriate amount of capital to hold against potential losses across the portfolio.
            st.markdown("* **Strategy Optimization:** Using insights to adjust lending criteria and improve overall business strategy.
            By shifting focus from a single loan to the entire portfolio, businesses can make more informed, strategic decisions to manage risk and maximize profitability.
            """
        )
        st.markdown("---")
        st.subheader("Key Features:")
        st.markdown(
            """
            * **Portfolio Management:** A comprehensive overview of your loan portfolio with key financial metrics.
            * **Model Validation:** Tools to validate the performance of your Probability of Default (PD) model.
            * **Deep Dive Analysis:** A flexible section for custom visualizations and segmented analysis.
            * **Custom Data Upload:** The ability to upload your own scored loan data for analysis.
            """
        )

elif page == "Portfolio Analysis":
    try:
        df = pd.read_csv("cr_loan_clean.csv")
        df['prob_default'] = df['loan_status']
        df['loan_amnt'] = df['loan_amnt']
        render_enterprise_dashboard(df.copy(), "Portfolio Analysis")
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
            
            # Smartly map columns
            if 'prob_default' not in df.columns:
                if 'loan_status' in df.columns:
                    df['prob_default'] = df['loan_status']
                else:
                    st.warning("Could not find a 'prob_default' or 'loan_status' column. Please rename a column in your file or select it below.")
                    prob_col = st.selectbox("Select the Probability of Default/Loan Status column", options=[None] + list(df.columns))
                    if prob_col:
                        df['prob_default'] = df[prob_col]
                    else:
                        st.stop()
            
            if 'loan_amnt' not in df.columns:
                st.warning("Could not find a 'loan_amnt' column. Please rename a column or select it below.")
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
