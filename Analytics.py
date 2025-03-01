
import logging
import pandas as pd
from Calculations import calculate_option_profit, black_scholes
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from datetime import datetime, timezone, timedelta
import time
from Fetch_data import Fetching_data
from Start_fetching_data import get_btcusd_price


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

fetch_data = Fetching_data()

class Analytic_processing:
    def __init__(self, analytic_data_csv: str = "analytic_data.csv", public_profits_csv: str = "public_trades_profits.csv"):
        self.analytic_data_csv_path = analytic_data_csv  # Store the path
        self.public_profits_csv_path = public_profits_csv
        self.analytic_data_df = pd.DataFrame()  # DataFrame to hold the analytic data
        self.public_profit_df = pd.DataFrame()

    def save_to_csv(self, df , mode):
        """Save  data to a CSV file based on data_type."""
        if not df.empty:
            if mode == "analytics" : 
                df.to_csv(self.analytic_data_csv_path, index=False)  # Use the path variable
                logging.info(f"Saved analytic data to {self.analytic_data_csv_path}")
            if mode == "public_profits" :
                df.to_csv(self.public_profits_csv_path , index=False)  # Use the path variable
                logging.info(f"Saved public trades data to {self.analytic_data_csv_path}")


    def compare_combined_profits(self, results_df, available_combo_options, combo_options_with_details_df, days_ahead_slider, quantity, risk_free_rate, selected_option_filter):
        combined_results = {'Underlying Price': results_df['Underlying Price']}
        
        # Get profits for the option symbol
        option_symbol_buy_profit = results_df[f'Day {days_ahead_slider} Profit (BUY)'] if f'Day {days_ahead_slider} Profit (BUY)' in selected_option_filter else None
        option_symbol_sell_profit = results_df[f'Day {days_ahead_slider} Profit (SELL)'] if f'Day {days_ahead_slider} Profit (SELL)' in selected_option_filter else None

        if option_symbol_buy_profit is None and option_symbol_sell_profit is None:
            logging.warning("No profits found for the selected option.")
            return combined_results  # Exit early if no profits for the main option

        index_price_range = np.arange(40000, 141000, 1000)
        results_dict = {'Underlying Price': index_price_range}

        with ProcessPoolExecutor() as executor:
            # Iterate through each combo option to calculate profits
            for combo_symbol in available_combo_options:
                filtered_df = combo_options_with_details_df.loc[combo_options_with_details_df['symbol'] == combo_symbol]

                if filtered_df.empty:
                    logging.warning(f"No data found for combo symbol: {combo_symbol}")
                    continue

                position_details = filtered_df.iloc[0]

                strike_price = position_details['strike_price']
                position_side_buy = "BUY"
                position_side_sell = "SELL"
                position_value_buy = position_details['ask_price_usd']
                position_value_sell = position_details['bid_price_usd']
                position_size = quantity
                future_iv_buy = position_details['ask_iv'] / 100 + 0.0
                future_iv_sell = position_details['bid_iv'] / 100 + 0.0

                expiration_date = pd.to_datetime(position_details['expiration_date']).date()
                now_utc = datetime.now(timezone.utc).date()
                remaining_days = max((expiration_date - now_utc).days - days_ahead_slider, 1)
                time_to_expiration_years = max(remaining_days / 365.0, 0.0001)
                
                # Calculate buy profits
                if option_symbol_buy_profit is not None:
                    buy_profit_future = executor.submit(self.calculate_public_profits,
                                                        (index_price_range, position_side_buy, strike_price, position_value_buy, position_size,
                                                        time_to_expiration_years, risk_free_rate, future_iv_buy, position_details['option_type']))
                    sell_profit_future = executor.submit(self.calculate_public_profits,
                                                        (index_price_range, position_side_sell, strike_price, position_value_sell, position_size,
                                                        time_to_expiration_years, risk_free_rate, future_iv_sell, position_details['option_type']))
                    sell_profits = sell_profit_future.result()
                    buy_profits = buy_profit_future.result()
                    selected_buy = option_symbol_buy_profit.squeeze()
                    # Store the total buy profit combining the option symbol profit
                    total_buy_profit = selected_buy + buy_profits

                    results_dict[f"{combo_symbol}-BUY"] = total_buy_profit

                    total_sell_profit = selected_buy + sell_profits

                    results_dict[f"{combo_symbol}-SELL"] = total_sell_profit
                
                # Calculate sell profits
                if option_symbol_sell_profit is not None:
                    sell_profit_future = executor.submit(self.calculate_public_profits,
                                                        (index_price_range, position_side_sell, strike_price, position_value_sell, position_size,
                                                        time_to_expiration_years, risk_free_rate, future_iv_sell, position_details['option_type']))
                    buy_profit_future = executor.submit(self.calculate_public_profits,
                                                        (index_price_range, position_side_buy, strike_price, position_value_buy, position_size,
                                                        time_to_expiration_years, risk_free_rate, future_iv_buy, position_details['option_type']))
                    buy_profits = buy_profit_future.result()
                    sell_profits = sell_profit_future.result()
                    selected_sell = option_symbol_sell_profit.squeeze()
                    # Store the total sell profit
                    total_sell_profit = selected_sell +  sell_profits
                    results_dict[f"{combo_symbol}-SELL"] = total_sell_profit

                    total_buy_profit = selected_sell + buy_profits

                    results_dict[f"{combo_symbol}-BUY"] = total_buy_profit

        combined_results.update(results_dict)

        if combined_results:
            self.analytic_data_csv  = pd.DataFrame(combined_results)
            self.save_to_csv(self.analytic_data_csv, "analytics")
        else:
            logging.warning("No options data fetched.")

        return self.analytic_data_csv
    
    def calculate_public_profits(self,args):
        index_price_range, position_side, strike_price, position_value, position_size, time_to_expiration_years, risk_free_rate, future_iv, position_type = args
        profit_results = []

        # Calculate profits for each underlying price in the specified range
        for u_price in index_price_range:
            mtm_price = black_scholes(u_price, strike_price, time_to_expiration_years, risk_free_rate, future_iv, position_type)

            # Calculate estimated profit based on the position side
            if position_side == "BUY":
                estimated_profit = (mtm_price - position_value) * position_size  # Profit from a buy position
            elif position_side == "SELL":
                estimated_profit = (position_value - mtm_price) * position_size  # Profit from a sell position
            else:
                raise ValueError(f"Invalid position side: {position_side}. Expected 'BUY' or 'SELL'.")

            profit_results.append(estimated_profit)

        return profit_results

    def processing_public_profit(self, 
                                 df, 
                                 hours_ahead, 
                                 specific_position=None, 
                                 specific_entry_date=None,
                                 specific_strike= None,
                                 specific_side = None,
                                 specific_type= None,
                                 specific_expiration= None):
        
        specific_entry_date = pd.to_datetime(specific_entry_date) if specific_entry_date else None
        change_in_iv = 0.0
        risk_free_rate = 0.0
        
        btc_price = get_btcusd_price()
        
        lower_bound = btc_price - 20000  # 20000 down from btc_price
        upper_bound = btc_price + 30000  # 30000 up from btc_price

        # Create index price range with a step of 500
        index_price_range = np.arange(lower_bound, upper_bound + 100, 100)
        
        # Initialize a dictionary to hold results
        results_dict = {'Underlying Price': index_price_range}

        position_count = 0
        
        with ProcessPoolExecutor() as executor:
            future_results = []
            
            for _, positions in df.iterrows():
                position_instrument = positions['Instrument']
                position_side = positions['Side']
                strike_price = positions['Strike Price']
                position_type = positions['Option Type'].lower()  
                expiration_date_str = positions['Expiration Date']

                if specific_position is not None and position_instrument != specific_position:
                    continue
                if specific_entry_date is not None and positions['Entry Date'] != specific_entry_date:
                    continue
                if specific_strike is not None and strike_price != specific_strike:
                    continue
                if specific_side is not None and position_side != specific_side:
                    continue
                if specific_type is not None and position_type != specific_type:
                    continue
                if  specific_expiration is not None and expiration_date_str !=  specific_expiration:
                    continue
                
                filtered_instrument = position_instrument
                position_size = positions['Size']
                position_iv = positions['IV (%)']
                position_value = positions['Price (USD)']

                future_iv = position_iv / 100 + (change_in_iv / 100.0)

                expiration_date = pd.to_datetime(expiration_date_str, utc=True)
                now_utc = datetime.now(timezone.utc)
                time_to_expiration_days = expiration_date - now_utc
                hours_delta = timedelta(hours=hours_ahead)
                remaining_days = time_to_expiration_days - hours_delta

                seconds_in_year = 365 * 24 * 3600
                time_to_expiration_years = max(remaining_days.total_seconds() / seconds_in_year, 0.0001)

                if position_iv <= 0 or time_to_expiration_years <= 0 or strike_price <= 0 or position_size <= 0:
                    continue

                # Prepare the arguments for the calculate_profits function
                future_results.append(
                    executor.submit(self.calculate_public_profits, 
                                    (index_price_range, position_side, strike_price, position_value, position_size, time_to_expiration_years,risk_free_rate, future_iv, position_type))
                )
                
                position_count += 1

            # Collect results from futures
            for idx, future in enumerate(future_results):
                instrument_key = f"{filtered_instrument}-{idx}"
                profits = future.result()
                results_dict[instrument_key] = profits

        # Convert the results dictionary to a DataFrame
        results_df = pd.DataFrame(results_dict)

        # Save the results to a CSV file
        self.public_profits_csv = results_df
        self.save_to_csv(self.public_profits_csv, "public_profits")

        return self.public_profits_csv
    
    def start_analyzing_public_trades(self):
        test = "this is only for test"
        """while True:
            try:
                market_screener_df  = fetch_data.load_market_trades()
                public_profits_df = self.processing_public_profit(market_screener_df , 1)
                print("Simulating poblic trade profits is completed.")

                time.sleep(60)  # Wait for 5 minutes before fetching again

            except KeyboardInterrupt:
                print("Process interrupted by the user.")
                break  # Exit the loop if the user interrupts

            except Exception as e:
                print(f"An error occurred: {e}")
                # Optionally, you could implement a wait time before retrying on failure
                time.sleep(30)  # Wait for 30 seconds before trying again in case of an error"""
        
        

                
