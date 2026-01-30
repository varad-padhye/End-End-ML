import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Federated Learning Dashboard")

st.title("📊 Federated Learning Training Monitor")

# Load metrics
df = pd.read_csv("logs/metrics.csv")

st.subheader("Global Accuracy Over Federated Rounds")

fig, ax = plt.subplots()
ax.plot(df["round"], df["global_accuracy"], marker="o")
ax.set_xlabel("Federated Round")
ax.set_ylabel("Global Accuracy")
ax.grid(True)

st.pyplot(fig)

# Summary
final_acc = df["global_accuracy"].iloc[-1]
total_rounds = df["round"].nunique()

st.subheader("📌 Final Summary")
st.write(f"**Total Federated Rounds:** {total_rounds}")
st.write(f"**Final Global Accuracy:** {final_acc:.4f}")
