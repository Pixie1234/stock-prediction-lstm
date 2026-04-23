import streamlit as st
import pandas as pd
import requests

st.title("📈 S&P 500 Stock Selector")

@st.cache_data
def load_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Use a real browser user-agent to avoid 403
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            "AppleWebKit/537.36 (KHTML, like Gecko)"
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()  # raises if 403 or any other error

    # Parse all HTML tables
    tables = pd.read_html(response.text)

    # Find the table containing "Symbol"
    for table in tables:
        if "Symbol" in table.columns:
            return table

    raise ValueError("Could not find S&P 500 table with a 'Symbol' column.")


# ---------------- MAIN APP ----------------
try:
    df_sp500 = load_sp500()
    st.success("Successfully loaded S&P 500 data from Wikipedia!")

    st.dataframe(df_sp500)

    tickers = df_sp500["Symbol"].tolist()
    selected_ticker = st.selectbox("Choose a stock ticker:", tickers)

    st.write(f"You selected: **{selected_ticker}**")

except Exception as e:
    st.error("Failed to fetch S&P 500 data.")
    st.exception(e)


