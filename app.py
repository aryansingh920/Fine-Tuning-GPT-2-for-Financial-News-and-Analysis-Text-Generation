# app.py
import streamlit as st
from main import FinancialGPT2Generator


def create_streamlit_app():
    st.title("Fine-Tuned Financial News Generator")

    # Load fine-tuned model
    @st.cache_resource
    def load_generator():
        return FinancialGPT2Generator(model_dir="./financial-gpt2")

    generator = load_generator()

    # Streamlit Controls
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    max_length = st.sidebar.slider("Max Length", 50, 500, 150)

    ticker_option = st.radio("News Type:", ["Random News", "Specific Ticker"])
    selected_ticker = st.text_input(
        "Enter Ticker:") if ticker_option == "Specific Ticker" else None

    if st.button("Generate News"):
        with st.spinner("Generating..."):
            result = generator.generate_news(
                ticker=selected_ticker, max_length=max_length, temperature=temperature)
            st.write("### Generated Financial News")
            st.write(result)


if __name__ == "__main__":
    create_streamlit_app()
