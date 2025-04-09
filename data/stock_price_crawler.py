"""
Stock Price Scraper

A streamlined script to fetch historical stock price data from Yahoo Finance and save it to CSV.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data for a given ticker and date range.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
            
    Returns:
        DataFrame: Stock price data
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    try:
        # Use yfinance to download the data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Process the DataFrame
        if not df.empty:
            # Reset the index to make Date a column
            df = df.reset_index()
            
            # Add ticker column
            df['Ticker'] = ticker
            
            # Format the Date column
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            return df
        else:
            print(f"No data retrieved for {ticker}")
            return None
            
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def save_to_csv(df, ticker, output_dir=None):
    """
    Save stock data to a CSV file in the specified directory.
    
    Args:
        df (DataFrame): Stock data to save
        ticker (str): Stock ticker symbol
        output_dir (str, optional): Directory to save the file
            
    Returns:
        str: Path to the saved file
    """
    if df is None or df.empty:
        print(f"No data to save for {ticker}")
        return None
    
    # Set default output directory if not specified
    if output_dir is None:
        # Use the data directory in the project
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, 'data', 'stock_prices')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename with the ticker and current date
    current_date = datetime.now().strftime('%Y%m%d')
    filename = f"{ticker}_data_{current_date}.csv"
    file_path = os.path.join(output_dir, filename)
    
    # Save the DataFrame to CSV
    df.to_csv(file_path, index=False)
    print(f"Saved data for {ticker} to {file_path}")
    
    return file_path

def get_stock_data(ticker, start_date, end_date, save=True, output_dir=None):
    """
    Fetch and optionally save stock data for a given ticker and date range.
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        save (bool, optional): Whether to save the data to CSV
        output_dir (str, optional): Directory to save the file
            
    Returns:
        tuple: (DataFrame with stock data, Path to saved file if save=True)
    """
    # Fetch the data
    df = fetch_stock_data(ticker, start_date, end_date)
    
    # Save the data if requested
    file_path = None
    if save and df is not None:
        file_path = save_to_csv(df, ticker, output_dir)
    
    return df, file_path

def main():
    """Stock data retrieval"""
    # Example parameters
    ticker = "AAPL"
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2021-01-01"
    
    # Fetch and save data
    df, file_path = get_stock_data(ticker, start_date, end_date, output_dir='data/stock_prices')
    
    # Print summary
    if df is not None:
        print(f"\nRetrieved {len(df)} days of data for {ticker}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Data saved to: {file_path}")
    
    return df

if __name__ == "__main__":
    main()
