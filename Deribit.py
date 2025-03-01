import requests
import logging
from datetime import datetime, date
import pandas as pd
import asyncio
import aiohttp
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DeribitAPI:
    def __init__(self, client_id: str, client_secret: str, options_data_csv: str = "options_data.csv", options_screener_csv: str = "options_screener.csv"):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.btc_usd_price = None
        self.session = requests.Session()
        self.options_data_csv = options_data_csv  # Path for CSV storage for options data
        self.options_screener_csv = options_screener_csv  # Path for CSV storage for options screener
        
        # Initialize empty DataFrames for options data and screener
        self.options_data = pd.DataFrame()
        self.options_screener = pd.DataFrame()

    def authenticate(self):
        if self.access_token:
            return self.access_token
        
        auth_url = 'https://www.deribit.com/api/v2/public/auth'
        auth_params = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'client_credentials'
        }
        
        try:
            response = self.session.get(auth_url, params=auth_params)
            response.raise_for_status()
            auth_data = response.json()
            self.access_token = auth_data.get('result', {}).get('access_token')
            if not self.access_token:
                raise ValueError("Authentication failed: Invalid credentials or response format")
            return self.access_token
        except requests.RequestException as e:
            logging.error(f"Authentication error: {e}")
            return None
        
    def fetch_btc_to_usd(self):
        """Fetch current BTC to USD conversion rate.""" 
        if self.btc_usd_price is not None:
            return self.btc_usd_price
        
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            self.btc_usd_price = data.get('bitcoin', {}).get('usd', 0)
            return self.btc_usd_price
        except requests.RequestException as e:
            logging.error(f"Error fetching BTC price: {e}")
            return 0
    
    def fetch_order_book(self, option_symbol):
        """Fetch the order book for a specific option."""
        order_book_url = 'https://www.deribit.com/api/v2/public/get_order_book'
        params = {'instrument_name': option_symbol}
        try:
            response = self.session.get(order_book_url, params=params)
            response.raise_for_status()
            return response.json().get('result', {})
        except requests.RequestException as e:
            logging.error(f"Failed to fetch order book for {option_symbol}: {e}")
            return {}

    def save_to_csv(self, df, data_type):
        """Save options data or options screener data to a CSV file based on data_type."""
        if data_type == "options_data":  # Save options data
            df.to_csv(self.options_data_csv, index=False)
            logging.info(f"Saved options data to {self.options_data_csv}")
        elif data_type == "options_screener":  # Save options screener data
            df.to_csv(self.options_screener_csv, index=False)
            logging.info(f"Saved options screener data to {self.options_screener_csv}")

    def refresh_options_data(self, currency='BTC'):
        """Refresh options data for the given currency."""
        logging.info(f"Refreshing options data for {currency}")

        access_token = self.authenticate()
        if not access_token:
            logging.error("Failed to authenticate.")
            return pd.DataFrame()
        
        btc_to_usd = self.fetch_btc_to_usd()

        options_data = []
        option_chains_url = 'https://www.deribit.com/api/v2/public/get_instruments'
        params = {'currency': currency, 'kind': 'option', 'expired': 'false'}
        headers = {'Authorization': f'Bearer {access_token}'}

        try:
            response = self.session.get(option_chains_url, params=params, headers=headers)
            response.raise_for_status()
            option_chains = response.json().get('result', [])
            logging.info(f"Fetched {len(option_chains)} option chains.")
        except requests.RequestException as e:
            logging.error(f"Failed to fetch option chains for {currency}: {e}")
            return pd.DataFrame()

        for instrument in option_chains:
            option_symbol = instrument['instrument_name']
            order_book = self.fetch_order_book(option_symbol)

            last_price_btc = order_book.get('last_price', 0) or 0
            bid_price_btc = order_book.get('best_bid_price', 0) or 0
            ask_price_btc = order_book.get('best_ask_price', 0) or 0
            bid_iv = order_book.get('bid_iv', 0) or 0.8  # Default IV if not available
            ask_iv = order_book.get('ask_iv', 0) or 0.8  # Default IV if not available

            delta = order_book.get('greeks', {}).get('delta')
            gamma = order_book.get('greeks', {}).get('gamma')
            theta = order_book.get('greeks', {}).get('theta')
            vega = order_book.get('greeks', {}).get('vega')

            volume = order_book.get('stats', {}).get('volume')
            volume_usd = order_book.get('stats', {}).get('volume_usd')
            open_interest = order_book.get('open_interest')

            # Check if bid or ask price is missing before adding to options_data
            if bid_price_btc <= 0 and ask_price_btc <= 0:
                continue

            # Construct the option details
            options_data.append({
                'symbol': option_symbol,
                'option_type': instrument['option_type'],
                'strike_price': instrument.get('strike', 0),
                'expiration_date': datetime.utcfromtimestamp(instrument['expiration_timestamp'] / 1000).date(),
                'last_price_usd': last_price_btc * btc_to_usd,
                'bid_price_usd': bid_price_btc * btc_to_usd,
                'ask_price_usd': ask_price_btc * btc_to_usd,
                'bid_iv': bid_iv,
                'ask_iv': ask_iv,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'open_interest': open_interest,
                'total traded volume': volume,
                'monetary volume': volume_usd
            })

        if options_data:
            self.options_data = pd.DataFrame(options_data)
            self.save_to_csv(self.options_data, data_type="options_data")
        else:
            logging.warning("No options data fetched.")

        return self.options_data

    async def fetch_public_trades_async(self, session, instrument_name, start_timestamp, end_timestamp):
        """Fetch public trade data asynchronously for a given instrument."""
        url = "https://www.deribit.com/api/v2/public/get_last_trades_by_instrument_and_time"
        params = {
            "instrument_name": instrument_name,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "count": 1000
        }

        for attempt in range(5):  # Retry up to 5 times
            async with session.get(url, params=params) as response:
                if response.status == 429:  # Too Many Requests
                    wait_time = 16  # fixed wait time for rate limit reached
                    await asyncio.sleep(wait_time)  # Wait before retrying
                    continue  # Retry the request
                response.raise_for_status()
                data = await response.json()
                return data

        logging.error(f"Failed to fetch trades for {instrument_name} after multiple attempts.")
        return None  # Return None if all attempts fail
    
    def extract_option_details(self, option_symbol: str):
        match = re.search(r'-(\d{1,2})([A-Z]{3})(\d{2})-(\d+)-(C|P)$', option_symbol)
        if not match:
            return None, None, None  # Ensure it returns a tuple with None values if no match

        day, month, year, strike_price, option_type = match.groups()
        expiration_date = f"{day}-{month.capitalize()}-{year}"
        strike_price = int(strike_price)
        option_type = "Call" if option_type == "C" else "Put"

        return expiration_date, strike_price, option_type  # Return as a tuple

    async def fetch_all_public_trades_async(self, instrument_names, start_timestamp, end_timestamp):
        """Fetch public trades for all instruments concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for instrument_name in instrument_names:
                tasks.append(self.fetch_public_trades_async(session, instrument_name, start_timestamp, end_timestamp))
            results = await asyncio.gather(*tasks)

            combined_trades = []
            for result in results:
                if result is None:  # Skip processing if result is None
                    continue
                if 'result' in result and 'trades' in result['result']:
                    combined_trades.extend(result['result']['trades'])
                else:
                    logging.error(f"Unexpected response format for instrument: {instrument_name}. Result: {result}")

            return combined_trades

    def process_screener_data(self, public_trades_df):
        """Process the public trades DataFrame by performing calculations, renaming columns, and saving it."""
        logging.info("Processing public trades data...")

        # Convert the 'timestamp' from trades to a readable date
        public_trades_df['timestamp'] = pd.to_datetime(public_trades_df['timestamp'], unit='ms')  # Convert ms to datetime
        
        public_trades_df['direction'] = public_trades_df['direction'].str.upper()

        # Format the timestamp: remove seconds and milliseconds
        public_trades_df['timestamp'] = public_trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

        # Ensure the extraction function is available and retrieve details
        for index, row in public_trades_df.iterrows():
            expiration_date, strike_price, option_type = self.extract_option_details(row['instrument_name'])
            public_trades_df.at[index, 'Expiration Date'] = expiration_date
            public_trades_df.at[index, 'Strike Price'] = strike_price
            public_trades_df.at[index, 'Option Type'] = option_type

        # Calculate additional columns
        public_trades_df['Price (USD)'] = (public_trades_df['price'] * public_trades_df['index_price']).round(2)
        public_trades_df['Entry Value'] = (public_trades_df['amount'] * public_trades_df['Price (USD)']).round(2)

        # Rename the columns as per the requirements
        columns = {
            'timestamp': 'Entry Date',
            'iv': 'IV (%)',
            'price': 'Price (BTC)',
            'direction': 'Side',
            'index_price': 'Underlying Price',
            'instrument_name': 'Instrument',
            'amount': 'Size',
            'mark_price': 'Mark Price (BTC)',
            'block_trade_id': 'BlockTrade IDs',
            'combo_id' : 'Combo ID', 
            'block_trade_leg_count' : 'BlockTrade Count', 
            'combo_trade_id': 'ComboTrade IDs',
        }
        
        public_trades_df.rename(columns=columns, inplace=True)

        # Ensure correct order of columns
        new_order = [
            'Side', 'Instrument', 'Price (BTC)', 'Price (USD)', 'Mark Price (BTC)',
            'IV (%)', 'Size', 'Entry Value', 'Underlying Price',
            'Expiration Date', 'Strike Price', 'Option Type', 'Entry Date','BlockTrade IDs',
            'BlockTrade Count', 'Combo ID', 'ComboTrade IDs'
        ]

        public_trades_df = public_trades_df[new_order]

        # Save the processed DataFrame to CSV
        self.save_to_csv(public_trades_df, data_type="options_screener")

    def execute_data_fetch(self, currency='BTC', start_date=None, end_date=None):
        """Fetch and save options and public trades data."""
        logging.info("Starting the data fetching process...")

        # Step 1: Authenticate and fetch options data
        options_data = self.refresh_options_data(currency)
        if options_data.empty:
            logging.warning("No options data fetched.")
            return

        # Step 2: Fetch public trades if start_date is provided
        if start_date and end_date:
            instrument_names = options_data['symbol'].tolist()
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000) + 86400000  # End of the day
            
            asyncio.run(asyncio.sleep(5))  # Wait 5 seconds before starting

            # Run the async fetching
            public_trades = asyncio.run(self.fetch_all_public_trades_async(instrument_names, start_timestamp, end_timestamp))

            if public_trades:
                public_trades_df = pd.DataFrame(public_trades)
                self.process_screener_data(public_trades_df)  # Process and save the public trades data
            else:
                logging.warning("No public trades fetched.")


        logging.info("Data fetching process completed.")