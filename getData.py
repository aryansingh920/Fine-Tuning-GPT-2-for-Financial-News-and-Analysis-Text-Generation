import numpy as np
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf


def fetch_yahoo_finance_news(tickers, output_file="combined_news.csv"):
    # Initialize an empty DataFrame to store all articles
    all_articles = []

    # Loop through each ticker
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        try:
            # Get news using yfinance
            stock = yf.Ticker(ticker)
            news = stock.news

            # Extract news data
            for article in news:
                content = article.get('content', {})
                all_articles.append({
                    'ticker': ticker,
                    'title': content.get('title', 'No Title'),
                    'summary': content.get('summary', 'No Summary'),
                    'description': content.get('description', 'No Description'),
                    'date': content.get('pubDate', 'No Date'),
                    'provider': content.get('provider', {}).get('displayName', 'Unknown')
                })

        except Exception as e:
            print(f"Failed to fetch news for {ticker}: {e}")

    # Convert list of all articles to DataFrame
    news_df = pd.DataFrame(all_articles)

    # Save to CSV
    news_df.to_csv(output_file, index=False)
    print(f"\nNews for all tickers saved to {output_file}")

    return news_df


# Define the file path
file_path = "data/constituents.csv"  # Replace with your file path

# Define the column name to extract
column_name = "Symbol"  # Replace with your desired column name

# Read the CSV file
df = pd.read_csv(file_path)

# Extract the particular column as a DataFrame
column_df = df[[column_name]]


# List of tickers
tickers = column_df[column_name].to_numpy()  # Add more tickers as needed

# Fetch and save news for all tickers
output_file = "data/finance_news.csv"
news_df = fetch_yahoo_finance_news(tickers, output_file)

# Display first few rows
print("\nFirst few rows of the dataset:")
print(news_df[['ticker', 'title', 'summary', 'description']].head())
