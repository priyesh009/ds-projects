import streamlit as st

st.set_page_config(page_title="Distribution Explorer", layout="centered")

st.title("📊 Statistical Distribution Explorer")
st.markdown("""
Welcome to the **Distribution Explorer**! This interactive tool lets you explore key probability distributions used in statistical modeling.

Choose a distribution below to learn more and visualize it in action:
""")

# Poisson
st.subheader("🔵 Poisson Distribution")
st.markdown("""
Used for modeling **count data**, the Poisson distribution is essential when events occur **independently** and at a **constant average rate**.

👉 [Launch Poisson App](https://poisson.streamlit.app/)
""")

# Gamma (placeholder)
st.subheader("🟣 Gamma Distribution")
st.markdown("""
The Gamma distribution models **positive continuous data**, often used in **insurance claims**, **waiting times**, and **survival analysis**.

👉 *(Coming Soon)*  
""")

# Tweedie (placeholder)
st.subheader("🟠 Tweedie Distribution")
st.markdown("""
Tweedie is a versatile distribution that covers **compound Poisson-Gamma models**, great for insurance severity/frequency models.

👉 *(Coming Soon)*  
""")

st.info("More distributions and interactive apps coming soon. Stay tuned!")
