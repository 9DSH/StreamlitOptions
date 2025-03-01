import requests
import logging
from datetime import datetime, date, timezone
import pandas as pd
import numpy as np
import os
import re
from Start_fetching_data import get_btcusd_price


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Fetching_data:
    def __init__(self, 
                 options_data_csv: str = "options_data.csv", 
                 options_screener_csv: str = "options_screener.csv",
                 analytic_data_csv: str = "analytic_data.csv",
                 public_profits_csv: str = "public_trades_profits.csv"):
        
        self.options_data_csv = options_data_csv  # Path for CSV storage for options data
        self.options_screener_csv = options_screener_csv  # Path for CSV storage for options screener
        self.analytic_data_csv = analytic_data_csv
        self.public_profits_csv = public_profits_csv
        # Initialize empty DataFrames for options data and screener
        self.analytic_data = pd.DataFrame()
        self.options_data = pd.DataFrame()
        self.options_screener = pd.DataFrame()
        self.public_profits = pd.DataFrame()


    def get_available_currencies(self):
        """Fetch available currencies for options."""
        return ['BTC', 'ETH']
    
        
    def fetch_available_dates(self, currency='BTC'):
        """Fetch all available expiration dates from the local options data DataFrame."""
        if self.options_data.empty:
            self.load_from_csv(data_type="options_data")
            if self.options_data.empty:
                logging.warning("Options data CSV is empty. Please refresh data first.")
                return []

        # Extract unique expiration dates
        available_dates = self.options_data['expiration_date'].unique()
        
        # Convert to date objects and sort
        return sorted(pd.to_datetime(available_dates).date.tolist())


    def load_from_csv(self, data_type):
        """Load options data or options screener data from CSV files."""

        if data_type == "options_data":  # Load options data
            if os.path.exists(self.options_data_csv):
                self.options_data = pd.read_csv(self.options_data_csv)
                logging.info("Loaded options data from CSV.")
            else:
                self.options_data = pd.DataFrame()
                logging.warning("Options data CSV does not exist.")

        elif data_type == "options_screener":  # Load options screener data
            if os.path.exists(self.options_screener_csv):
                self.options_screener = pd.read_csv(self.options_screener_csv)
                logging.info("Loaded options screener data from CSV.")
            else:
                self.options_screener = pd.DataFrame()
                logging.warning("Options screener CSV does not exist.")

        elif data_type == "analytics_data":  # Load options screener data
            if os.path.exists(self.analytic_data_csv):
                self.analytic_data = pd.read_csv(self.analytic_data_csv)
                logging.info("Loaded analytic data from CSV.")
            else:
                self.analytic_data = pd.DataFrame()
                logging.warning("Analytic data CSV does not exist.")

        elif data_type == "public_profits":  # Load options screener data
            if os.path.exists(self.public_profits_csv):
                self.public_profits = pd.read_csv(self.public_profits_csv)
                logging.info("Loaded public profit data from CSV.")
            else:
                self.public_profits = pd.DataFrame()
                logging.warning("Public profit data CSV does not exist.")


    def get_options_for_date(self, currency='BTC', expiration_date=None):
        """Get available options for a specific expiration date."""
        # Try loading from CSV first
        if self.options_data.empty:
            self.load_from_csv(data_type="options_data")

        if expiration_date and not self.options_data.empty:
            options_df = self.options_data[self.options_data['expiration_date'] == str(expiration_date)]
            return options_df['symbol'].tolist() if not options_df.empty else []

        return []

    def fetch_option_data(self, option_symbol=None):
        """Fetch detailed option data for either a specific symbol or a list of symbols using the loaded options DataFrame."""
        if self.options_data.empty:
            self.load_from_csv(data_type="options_data")

        if option_symbol is not None and not self.options_data.empty:
            if isinstance(option_symbol, list):
                # If a list of symbols is provided
                detailed_data = self.options_data[self.options_data['symbol'].isin(option_symbol)]
                return detailed_data, detailed_data['symbol'].unique() if not detailed_data.empty else (None, None)
            elif isinstance(option_symbol, str):
                # If a single symbol is provided
                detailed_data = self.options_data[self.options_data['symbol'] == option_symbol]
                return detailed_data, detailed_data['symbol'].values[0] if not detailed_data.empty else (None, None)

        # If option_symbol is not provided, return the DataFrame
        return self.options_data, None

    def get_all_options(self, currency='BTC', filter=None, type='data'):
        """Fetch all available options and return them as a DataFrame, using the loaded options data.

        Args:
            currency (str): The currency type, default is 'BTC'.
            filter (str or None): The column name to filter by, default is None.
            type (str): The type of result to return ('data', 'sum', 'average').

        Returns:
            pd.DataFrame or numeric results based on the input parameters.
        """
        # Load options data if it's not already loaded
        if self.options_data.empty:
            self.load_from_csv(data_type="options_data")
        
        # Check if options data is not empty
        if not self.options_data.empty:
            if filter is None:
                return self.options_data  # Return full DataFrame
            
            elif filter in self.options_data.columns:
                filtered_df = self.options_data[[filter]]
                if pd.api.types.is_numeric_dtype(filtered_df[filter]):
                    if type == 'sum':
                        return filtered_df[filter].sum()  # Return sum only
                    elif type == 'average':
                        return filtered_df[filter].mean()  # Return average only
                    else:  # Default case is to return filtered data
                        return filtered_df[filter]  # Return the filtered Series
                else:
                    if type == 'data':
                        return filtered_df[filter]  # Return the filtered Series
                    else:
                        raise ValueError(f"Cannot compute {type} on non-numeric column '{filter}'.")
                    
            else:
                raise ValueError(f"'{filter}' is not a valid column name.")
        
        return pd.DataFrame()
    
    def get_all_strike_options(self, currency, option_strike=None, option_type=None):
        """Get all options available for a specific strike price from the cached options_data DataFrame."""
        
        # Load options data if it's not already loaded
        if self.options_data.empty:
            self.load_from_csv(data_type="options_data")
        
        if self.options_data.empty:
            logging.error("No options available. Please fetch all options first.")
            return pd.DataFrame()

        # Filter options based on the specified strike price
        if option_strike is not None:
            filtered_options = self.options_data[self.options_data['strike_price'] == option_strike]
        else:
            filtered_options = self.options_data  # If no option_strike provided, return all options

        # Further filter by option type if provided
        if option_type:
            filtered_options = filtered_options[filtered_options['option_type'] == option_type]

        return filtered_options
    
    def get_itm_options_for_date(self, currency='BTC', expiration_date=None):
        """Get all  options for a specific expiration date.
           - catch all the call options that has strike price less than current price
           - catch all the put options that has strike price more that current price
           - filters them by positive open interest,
           - then sorts them by monatary volume  
           
           returns dataframes of ITM options with their options details"""
        
        if not isinstance(expiration_date, date):
            logging.error("Invalid date parameter. It must be a datetime.date object.")
            return pd.DataFrame()

        # Fetch current market price
        current_price = get_btcusd_price()

        # Load all options from the CSV
        if self.options_data.empty:
            self.load_from_csv(data_type="options_data")
        
        if self.options_data.empty:
            logging.warning("No options data available. Please refresh data first.")
            return pd.DataFrame()

        # Filter options for the specified expiration date
        options_df = self.options_data[self.options_data['expiration_date'] == str(expiration_date)]

        # If no options are found, return an empty DataFrame
        if options_df.empty:
            logging.info(f"No options found for expiration date: {expiration_date}")
            return pd.DataFrame()

        itm_options = []

        # Filter for ITM options
        for index, row in options_df.iterrows():
            strike_price = row['strike_price']
            option_type = row['option_type']
            instrument_symbol = row['symbol']

            if option_type == 'call' and current_price > strike_price:  # Call Options ITM
                itm_options.append(instrument_symbol)

            elif option_type == 'put' and current_price < strike_price:  # Put Options ITM
                itm_options.append(instrument_symbol)

        # Return ITM options as a DataFrame
        # Fetch detailed information for each ITM option
        detailed_itm_options = []
        for symbol in itm_options:
            detailed_data, _ = self.fetch_option_data(option_symbol=symbol)  # Fetch detailed data for each option
            if not detailed_data.empty and (detailed_data['open_interest'] > 0).any():
                detailed_itm_options.append(detailed_data)

        # Concatenate the detailed DataFrames into one
        if detailed_itm_options:
            combined_df = pd.concat(detailed_itm_options, ignore_index=True)
            
            # Sort by 'open_interest' before returning
            return combined_df.sort_values(by='monetary volume', ascending=False)  # Change ascending=True to False based on your requirement

        logging.info(f"No detailed ITM options found for expiration date: {expiration_date}")
        return pd.DataFrame()
    
    def load_market_trades(self, filter=None , drop = True):
        """Load options screener data and return it after filtering.

        Parameters:
            filter (str, optional): The instrument name to filter by. If None, return the full DataFrame.

        Returns:
            pd.DataFrame: The filtered or unfiltered options screener DataFrame.
        """
        # Load the complete options screener data from the CSV
        self.load_from_csv(data_type="options_screener")

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        options_screener_copy = self.options_screener.copy()

        # Check if 'entry date' exists before sorting
        if not options_screener_copy.empty:
            if 'Entry Date' in options_screener_copy.columns:
                options_screener_copy['Entry Date'] = pd.to_datetime(options_screener_copy['Entry Date'], errors='coerce')
                # Sort the DataFrame by 'entry date'
                options_screener_copy.sort_values(by='Entry Date', ascending=False, inplace=True)

            # Drop unnecessary columns
            if drop: 
                options_screener_copy = options_screener_copy.drop(columns=['block_trade_id', 'combo_id', 'block_trade_leg_count', 'combo_trade_id'], errors='ignore')

            # If filter is provided and not None, filter the DataFrame
            if filter:
                options_screener_copy = options_screener_copy[options_screener_copy['Instrument'] == filter]

        return options_screener_copy
    
    
    def extract_instrument_info(self, column_name):
        """Extracts instrument name and option side from the column name."""
        parts = column_name.split('-')
        instrument_name = '-'.join(parts[:-1])  # All parts except the last part (option side)
        option_side = parts[-1]                  # Last part is the option side
        return instrument_name, option_side

    def filter_best_options_combo(self, loss_threshold, premium_threshold, quantity, show_buy, show_sell):
        """
        Filters the combined results to identify options that meet the loss threshold and premium criteria.

        Parameters:
            combined_results (pd.DataFrame): DataFrame containing profit results.
            loss_threshold (float): Maximum loss to consider an option as acceptable.
            premium_threshold (float): Minimum premium to consider an option as profitable.
            quantity (int): Quantity of options for calculating premium.

        Returns:
            pd.DataFrame: A DataFrame where the first row contains premiums,
                        followed by the filtered profit results.
        """

        self.load_from_csv(data_type="analytics_data")
         
        combined_results = self.analytic_data.copy()
        # Ensure combined_results has the expected structure
        if combined_results.empty or 'Underlying Price' not in combined_results.columns:
            logger.error("No valid data available in the combined results.")
            return pd.DataFrame()  # Return an empty DataFrame on error

        # Create a copy of combined_results to modify
        filtered_results = combined_results.copy()

        # This will hold the premiums for valid options
        premium_row = [np.nan] * (len(filtered_results.columns) - 1)  # Initialize with NaNs for all valid columns

        # Loop through each column (ignoring the first column which is 'Underlying Price')
        for i, column in enumerate(combined_results.columns[1:]):
            try:
                # Check if any value in the column is below the loss threshold
                if (combined_results[column] < loss_threshold).any():
                    filtered_results = filtered_results.drop(columns=[column], errors='ignore')  # Drop if loss threshold not met
                    continue  # Skip further processing for this column

                # Extract instrument name and option side
                instrument_name, option_side = self.extract_instrument_info(column)

                # Fetch option data from your data source
                instrument_detail, _ = self.fetch_option_data(instrument_name)

                if instrument_detail is None or instrument_detail.empty:
                    logger.warning(f"No details found for instrument: {instrument_name}.")
                    continue  # Skip to the next column if no details are found

                # Validate that the fetched instrument details correspond to the instrument name
                if 'symbol' in instrument_detail.columns and instrument_detail['symbol'].values[0] != instrument_name:
                    logger.error(f"Incompatible instrument data for {instrument_name}. Expected name not found.")
                    continue  # Skip processing for this column

                # Calculate premium based on the option side
                if option_side == 'BUY':
                    if 'ask_price_usd' not in instrument_detail.columns:
                        logger.error(f"No ask price data available for instrument: {instrument_name}.")
                        continue
                    instrument_premium = instrument_detail['ask_price_usd'].values[0] * quantity
                else:
                    if 'bid_price_usd' not in instrument_detail.columns:
                        logger.error(f"No bid price data available for instrument: {instrument_name}.")
                        continue
                    instrument_premium = instrument_detail['bid_price_usd'].values[0] * quantity

                # Check if the premium meets the threshold
                if instrument_premium <= premium_threshold and instrument_premium > 0:
                    # Valid premium, update the premium row at the correct index
                    premium_row[i] = instrument_premium
                else:
                    # Drop the column in filtered_results since the premium is below the threshold
                    filtered_results = filtered_results.drop(columns=[column], errors='ignore')

            except Exception as e:
                logger.error(f"Error processing column {column}: {e}")
                continue  # Continue processing other columns

        # If there are valid premiums, insert them as the first row
        valid_premiums = [p for p in premium_row if pd.notna(p)]
        if valid_premiums:
            premium_row_df = pd.DataFrame([valid_premiums], columns=filtered_results.columns[1:len(valid_premiums) + 1])
            
            # Insert the new row into the top of the filtered results
            filtered_results = pd.concat([premium_row_df, filtered_results], ignore_index=True)

            # Add a new row index name for the premium row
            filtered_results.index = ['Premium'] + list(range(len(filtered_results) - 1))

        filtered_results_copy = filtered_results[['Underlying Price']].copy()
        # Applying filters based on flags
        if show_buy and show_sell:
            # If both flags are true, do nothing since we want all relevant columns
            filtered_results_copy = filtered_results 
        else:
            if show_buy:
                buy_columns = filtered_results.filter(like="BUY").columns
                filtered_results_copy = filtered_results [['Underlying Price']].join(filtered_results[buy_columns])
            if show_sell:
                sell_columns = filtered_results.filter(like="SELL").columns
                filtered_results_copy = filtered_results [['Underlying Price']].join(filtered_results[sell_columns])

        return filtered_results_copy
    

    def load_public_trades_profit(self, filter=None):
        """Load options screener data and return it after filtering.

        Parameters:
            filter (str, optional): The instrument name to filter by. If None, return the full DataFrame.

        Returns:
            pd.DataFrame: The filtered or unfiltered options screener DataFrame.
        """
        # Load the complete options screener data from the CSV
        self.load_from_csv(data_type="public_profits")

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        options_screener_copy = self.public_profits.copy()

        # Check if 'entry date' exists before sorting
        if 'Entry Date' in options_screener_copy.columns:
            options_screener_copy['Entry Date'] = pd.to_datetime(options_screener_copy['Entry Date'], errors='coerce')
            # Sort the DataFrame by 'entry date'
            options_screener_copy.sort_values(by='Entry Date', ascending=False, inplace=True)

        # Drop unnecessary columns
        options_screener_copy = options_screener_copy.drop(columns=['block_trade_id', 'combo_id', 'block_trade_leg_count', 'combo_trade_id'], errors='ignore')

        # If filter is provided and not None, filter the DataFrame
        if filter:
            options_screener_copy = options_screener_copy[options_screener_copy['Instrument'] == filter]

        return options_screener_copy
        


        

        