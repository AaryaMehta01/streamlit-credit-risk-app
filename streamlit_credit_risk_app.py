
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk Analytics", page_icon="📊", layout="wide")

# ---------------- Header ----------------
col1, col2 = st.columns([1,3])
with col1:
    st.markdown("### 📊 Credit Risk Analytics")
with col2:
    st.markdown("<div style='text-align:right;'>Built with SQL + Python + Power BI + Matplotlib</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Sidebar ----------------
st.sidebar.header("🔧 Controls")
threshold = st.sidebar.slider("Probability Threshold (accept if PD ≤ threshold)", min_value=0.0, max_value=1.0, value=0.85, step=0.01)
lgd = st.sidebar.slider("Loss Given Default (LGD)", 0.0, 1.0, 1.0, 0.05)
ead_multiplier = st.sidebar.number_input("EAD Multiplier (scale loan amounts)", value=1.0, step=0.1)

st.sidebar.caption("Tip: Use the threshold to balance acceptance rate and bad rate.")

# ---------------- Data Upload / Sample ----------------
st.subheader("📥 Upload Scored Loans (CSV) or use Sample")
st.caption("Required columns: `prob_default` (0-1), `loan_amnt` (numeric). Additional columns are welcome.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    np.random.seed(42)
    n = 8000
    # Simulate a realistic PD distribution and loan amounts
    pd_def = np.clip(np.random.beta(a=1.8, b=8.5, size=n), 0, 1)
    loan_amnt = np.random.lognormal(mean=10.2, sigma=0.6, size=n)  # around $27k median
    df = pd.DataFrame({"prob_default": pd_def, "loan_amnt": loan_amnt})

# Basic validation
if not {"prob_default", "loan_amnt"}.issubset(df.columns):
    st.error("CSV must contain 'prob_default' and 'loan_amnt' columns.")
    st.stop()

df["loan_amnt"] = df["loan_amnt"] * ead_multiplier

# ---------------- Acceptance Strategy ----------------
df["accept"] = (df["prob_default"] <= threshold).astype(int)
accept_rate = df["accept"].mean()

# Bad rate among accepted (defaults are PD > threshold as proxy)
accepted_df = df[df["accept"] == 1]
if len(accepted_df) > 0:
    bad_rate = (accepted_df["prob_default"] > threshold).mean()  # proxy if you don't have y_true
else:
    bad_rate = 0.0

# Expected loss
df["expected_loss"] = df["prob_default"] * lgd * df["loan_amnt"]
total_expected_loss = df["expected_loss"].sum()

accepted_expected_loss = accepted_df["expected_loss"].sum() if len(accepted_df) else 0.0
total_accepted_volume = accepted_df["loan_amnt"].sum() if len(accepted_df) else 0.0

# ---------------- Key Metrics ----------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Acceptance Rate", f"{accept_rate:,.0%}")
m2.metric("Bad Rate (Accepted)", f"{bad_rate:,.1%}")
m3.metric("Expected Loss (All)", f"${total_expected_loss:,.0f}")
m4.metric("Accepted Volume", f"${total_accepted_volume:,.0f}")

st.markdown("---")

# ---------------- Visuals ----------------
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### 📈 PD Distribution with Threshold")
    fig1, ax1 = plt.subplots()
    ax1.hist(df["prob_default"], bins=50)
    ax1.axvline(threshold, linestyle="--")
    ax1.set_xlabel("Probability of Default")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

with c2:
    st.markdown("#### 💵 Expected Loss by Acceptance")
    tmp = df.copy()
    tmp["bucket"] = np.where(tmp["accept"]==1, "Accepted", "Rejected")
    loss_by_bucket = tmp.groupby("bucket")["expected_loss"].sum().reindex(["Accepted","Rejected"]).fillna(0)
    fig2, ax2 = plt.subplots()
    ax2.bar(loss_by_bucket.index, loss_by_bucket.values)
    ax2.set_ylabel("Expected Loss ($)")
    st.pyplot(fig2)

st.markdown("---")

# ---------------- Strategy Table ----------------
st.markdown("#### 📋 Strategy Table (Sweep Thresholds)")
sweep = np.linspace(0.1, 0.95, 18)
rows = []
for t in sweep:
    acc = (df["prob_default"] <= t).mean()
    acc_df = df[df["prob_default"] <= t]
    bad = (acc_df["prob_default"] > t).mean() if len(acc_df) else 0.0
    el = (df["prob_default"] * lgd * df["loan_amnt"]).sum()
    rows.append({"threshold": t, "accept_rate": acc, "bad_rate_acc": bad, "expected_loss": el})
table = pd.DataFrame(rows)
st.dataframe(table.style.format({"threshold":"{:.2f}","accept_rate":"{:.0%}","bad_rate_acc":"{:.1%}","expected_loss":"${:,.0f}"}))

st.caption("Note: If you have true labels, you can replace the bad-rate proxy with actual default outcomes.")

# ---------------- Download ----------------
st.download_button(
    label="Download Strategy Table (CSV)",
    data=table.to_csv(index=False),
    file_name="strategy_table.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("Built for analytics storytelling — highlight SQL preprocessing, Python modeling, and Power BI dashboards for decision-makers.")
