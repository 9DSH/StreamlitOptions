import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import subprocess
import time
import threading
from datetime import date, datetime, timedelta , timezone
import logging
import warnings
import webbrowser
from Fetch_data import Fetching_data
from Analytics import Analytic_processing
from Calculations import calculate_option_profit , calculate_totals_for_options, get_most_traded_instruments , calculate_sums_of_public_trades_profit
from Charts import plot_strike_price_vs_size , plot_stacked_calls_puts, plot_option_profit , plot_radar_chart, plot_price_vs_entry_date, plot_most_traded_instruments , plot_underlying_price_vs_entry_value , plot_identified_whale_trades
from Start_fetching_data import start_fetching_data_from_api,  get_btcusd_price

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the Streamlit page configuration
st.set_page_config(page_title='Trading Dashboard', layout='wide')

fetch_data = Fetching_data()
analytics = Analytic_processing()


# Initialize the thread reference globally
data_refresh_thread = None
public_trades_thread = None


def start_data_refresh_thread():
    global data_refresh_thread
    if data_refresh_thread is None or not data_refresh_thread.is_alive():
        data_refresh_thread = threading.Thread(target=start_fetching_data_from_api)
        data_refresh_thread.start()

#def start_processing_market_trades():
   # global public_trades_thread
   # if public_trades_thread is None or not public_trades_thread.is_alive():
       # public_trades_thread = threading.Thread(target=analytics.start_analyzing_public_trades)
       # public_trades_thread.start()
    


def app():
    combo_breakeven_sell = None
    combo_breakeven_buy = None
    title_row = st.container()
    # Fetch and display the current price
    btc_price = get_btcusd_price()
    with title_row:
        col1, col2, col3 = st.columns([1, 1, 1])  # Adjust ratio for centering
        with col2:
            
            st.header("Option Dashboard")
        with col3:
            colmm1, colmm2 = st.columns(2)
            with colmm1:
                st.write("")
            with colmm2: 
                btc_display_price = f"{btc_price:.0f}" if btc_price is not None else "Loading..."
                st.metric(label="BTC USD", value=btc_display_price, delta=None, delta_color="normal", help="Bitcoin price in USD")




    # Create columns for the layout
    colm1, colm2, colm3 = st.columns(3)

    # Populate each column with metrics using HTML for reduced size
    with colm1:
        st.write("")

    with colm2:
        st.write("")
    with colm3:
        st.write("")

    # Initialize data fetching at the start of the app
    # Button to refresh data
    

    premium_buy = 0
    premium_sell = 0
    with st.sidebar:

        currency = 'BTC'
        #current_date_initials = pd.to_datetime(datetime.now()).date()

        
        # Initialize session state for inputs if they don't exist
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = fetch_data.fetch_available_dates()[0]  # Default to first date
        if 'option_symbol' not in st.session_state:
            st.session_state.option_symbol = None  # Initialize this as None or an empty value
        if 'quantity' not in st.session_state:
            st.session_state.quantity = 0.1  # Default quantity
        if 'option_side' not in st.session_state:
            st.session_state.option_side = "BUY"  # Default side
        if 'most_profitable_df' not in st.session_state:
            st.session_state.most_profitable_df = pd.DataFrame()  # Initialize with an empty DataFrame
        


        available_dates = fetch_data.fetch_available_dates()

        selected_date = st.sidebar.selectbox("Select Expiration Date", available_dates, 
                                            index=available_dates.index(st.session_state.selected_date))
        st.session_state.selected_date = selected_date  # Update session state

        if selected_date:
            # Fetch options available for the selected date
            options_for_date = fetch_data.get_options_for_date(currency, selected_date)
            
            # Allow user to select an option
            if options_for_date:
                # Check if the currently selected option_symbol is in the available options
                if st.session_state.option_symbol not in options_for_date:
                    # Reset option_symbol to None if it's not available
                    st.session_state.option_symbol = None

                # Set the index for the selectbox based on the current selection
                option_symbol_index = 0 if st.session_state.option_symbol is None else options_for_date.index(st.session_state.option_symbol)
                option_symbol = st.sidebar.selectbox("Select Option", options_for_date, index=option_symbol_index)

                # Ensure to update the session state with the user's selection
                if st.session_state.option_symbol != option_symbol :
                    st.session_state.most_profitable_df = pd.DataFrame()
                    st.session_state.option_symbol = option_symbol  # Update session state


                col1, col2 = st.columns(2)

                with col1:
                    # Use st.session_state.quantity directly in the number_input
                    quantity = st.number_input('Quantity',
                                                min_value=0.1,
                                                step=0.1,
                                                value=st.session_state.quantity)  # Current value from session state
                    # Update session state only if the value changes
                    if quantity != st.session_state.quantity:
                        st.session_state.quantity = quantity  # Update the session state after the widget is used

                with col2:
                    option_side = st.selectbox("Select Side", options=['BUY', 'SELL'], 
                                                index=['BUY', 'SELL'].index(st.session_state.option_side))
                    st.session_state.option_side = option_side  # Store input in session state
                    show_buy = True if option_side == 'BUY' else False
                    show_sell = True if option_side == 'SELL' else False

                    
                
                if option_symbol:
                    # Get and display the details of the selected option
                    option_details, option_index_price = fetch_data.fetch_option_data(option_symbol)
                    all_options_with_details = fetch_data.get_all_options(filter=None, type='data')
                    recent_public_trades_df = fetch_data.load_market_trades(option_symbol)


                    if not option_details.empty:
                        # Extracting details safely
                        
                        expiration_date_str = option_details['expiration_date'].values[0]
                        expiration_date = pd.to_datetime(expiration_date_str).date()  # Ensure it's converted to date

                        option_type = option_details['option_type'].values[0]
                        bid_iv = option_details['bid_iv'].values[0]
                        ask_iv = option_details['ask_iv'].values[0]
                        bid_price = option_details['bid_price_usd'].values[0]
                        ask_price = option_details['ask_price_usd'].values[0]
                        strike_price = option_details['strike_price'].values[0]
                        premium_buy = ask_price * quantity
                        premium_sell = bid_price * quantity

                        breakeven_call_buy = premium_buy + strike_price
                        breakeven_call_sell = premium_sell+ strike_price
                        breakeven_put_buy = strike_price - premium_buy
                        breakeven_put_sell = strike_price - premium_sell

                        
                        if option_side == "BUY":
                                    breakeven_sell = None
                                    premium = premium_buy
                                    
                        if option_side == "SELL":
                                    breakeven_buy = None
                                    premium = premium_sell


                        
                        breakeven_buy = breakeven_call_buy if option_type == 'call' else breakeven_put_buy
                        breakeven_sell = breakeven_put_sell if option_type == 'put' else breakeven_call_sell

                        

                        now_utc = datetime.now(timezone.utc).date()

                        # Compute total days to expiration (at least 1 to avoid zero)
                        time_to_expiration_days = max((expiration_date - now_utc).days, 1)



                    else:
                        st.write("Error fetching option details.")

                

            else:
                st.warning("No options available for the selected date.")


        days_ahead_slider = st.slider(
                                            f'Days ahead for {option_symbol}',
                                            min_value=0,
                                            max_value=time_to_expiration_days,
                                            value=1,
                                            help="Simulate the option's mark-to-market value after X days (before expiration)."
                                        )
        
        
        
        

        #---------------------------------------------
        #       Analyze and Filter 
        #-----------------------------------------------------
        with st.expander("Analyze Options", expanded=False):
            analyze_row = st.container()
            filter_rows = st.container()
            with analyze_row:
                analyze_col1, analyze_col2 = st.columns(2)
                with analyze_col1:
                    combo_quantity = st.number_input(
                                                    'Combo Quantity',
                                                    min_value=0.1,
                                                    step=0.1,
                                                    value=0.1,
                                                    key="combo_quantity"  # Unique key for this input
                                                                )
                with analyze_col2:
                    number_of_total_option_chains = all_options_with_details.shape[0]
                    st.markdown(f"<p style='font-size: 14px;; margin-top: 28px;'></p>", unsafe_allow_html=True) 
                    apply_combo = st.button(label="Analyze", key="Apply_combo")
                    

                
                st.markdown(f"<p style='font-size: 14px;'>Set Filters after the analyzing process has been completed, the result apears in 'Combinations' tab</p>", unsafe_allow_html=True) 
                st.markdown("---")

            #------------- Analayze ----------------------
            risk_free_rate = 0.0
            change_in_iv = 0.0

            # ------------------------------------------------------------------------------ 
            # Calculating Profit of the selected option in sidebar
            # ------------------------------------------------------------------------------ 
            results_df = calculate_option_profit(option_details,
                                                            days_ahead_slider,
                                                            quantity,
                                                            risk_free_rate,
                                                            change_in_iv,
                                                            option_symbol)

            results_filter = ['Underlying Price']
            if show_buy:
                results_filter.append(f'Day {days_ahead_slider} Profit (BUY)') 
                results_filter.append('Expiration Profit (BUY)')  
            if show_sell:
                results_filter.append('Expiration Profit (SELL)')
                results_filter.append(f'Day {days_ahead_slider} Profit (SELL)')

            if results_filter:
                results_df = results_df[results_filter]

                
            #-----------------------------------------------------------------------
            #    Start analyzing
            # -------------------------------------------------------------------------   
            
            if apply_combo:
                    if not all_options_with_details.empty:
                        available_combo_options = all_options_with_details['symbol'].tolist()

                        if option_symbol in available_combo_options: available_combo_options.remove(option_symbol)

                        combined_results = analytics.compare_combined_profits(results_df,
                                                                        available_combo_options, 
                                                                        all_options_with_details, 
                                                                        days_ahead_slider, 
                                                                        combo_quantity , 
                                                                        risk_free_rate, 
                                                                        results_filter)
            
            #-------------- Filter ----------------------------
            
            with filter_rows:
                first_filter_row = st.container()
                second_filter_row = st.container()
                third_filter_row = st.container()
                with first_filter_row:
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        combo_loss_threshold = st.number_input(
                                                    'Maximum Loss Threshold',
                                                    min_value=0,
                                                    step=1,
                                                    value=100,
                                                    key="combo_loss_threshold"  # Unique key for this input
                                                )
                        combo_loss_threshold = - combo_loss_threshold
                    with filter_col2:
                        combo_premium_threshold = st.number_input(
                                                    'Maximum Premium Threshold',
                                                    min_value=0,
                                                    step=1,
                                                    value=300,
                                                    key="combo_premium_threshold"  # Unique key for this input
                                                )
                with second_filter_row:
                    show_combo_options = ['Buy', "Sell"]
                    multi_side_filter = st.multiselect("Select Side", options=show_combo_options, default= ['Buy', "Sell"])
                    show_combo_buy = True if "Buy" in multi_side_filter else False
                    show_combo_sell = True if "Sell" in multi_side_filter else False

                with third_filter_row:
                    filter1 , filter2, filter3 = st.columns(3)
                    with filter2:
                        apply_filter = st.button(label="Filter", key="filter_combo")

                ## apply conditions to show the best results for combinations
                if apply_filter:
                    st.session_state.most_profitable_df = fetch_data.filter_best_options_combo(
                        combo_loss_threshold, combo_premium_threshold, combo_quantity, show_combo_buy, show_combo_sell
                    )
                    



    ##------------------------------------------------------
    ##---------------------- MAIN TABS ------------------------
    #-------------------------------------------------------
    if 'expiration_date' in locals() and not option_details.empty:
        Profit_tab = f'{option_symbol}'
        main_tabs = st.tabs(["Market Watch",  Profit_tab, "Combinations"])

#---------------------------------------------------------------
#-----------------------Market Watch ---------------------------
#-------------------------------------------------------------

        with main_tabs[0]:
             # Initialize trades variable outside of any if conditions
            market_screener_df  = fetch_data.load_market_trades()
            # here we can have the function that simulates the public trades profit

            filter_row = st.container()
            with filter_row:
                col_refresh, col_date,col_range ,col_strike, col_expiration,  col_f4 = st.columns([0.01, 0.2, 0.1, 0.08, 0.08, 0.05])
                #with col_refresh:
                    #apply_market_filter = st.button(label="Apply", key="apply_market_filter")

                with col_date:
                    # date_row1 = st.container()
                    # date_row2 = st.container()

                   # with date_row1:
                    cc1,cc2,cc3 ,ca1,ca2,ca3 = st.columns([0.04, 0.02, 0.02, 0.04, 0.02, 0.02])
                    with cc1:
                        start_date = st.date_input("Start Entry Date", value=date(2025, 2, 1))
                    with cc2:
                        start_hour = st.number_input("Hour", min_value=0, max_value=23, value=0)
                    with cc3:
                        start_minute = st.number_input("Minute", min_value=0, max_value=59, value=0)

                    with ca1 :
                        current_utc_date =  datetime.now(timezone.utc).date()
                        end_date = st.date_input("End Entry Date", value=current_utc_date)

                    with ca2:
                        end_hour = st.number_input("Hour", min_value=0, max_value=23, value=23)
                    with ca3:
                        end_minute = st.number_input("Minute", min_value=0, max_value=59, value=59)

                        # Combine date and time into a single datetime object
                    start_datetime = datetime.combine(start_date, datetime.min.time().replace(hour=start_hour, minute=start_minute))
                    end_datetime = datetime.combine(end_date, datetime.min.time().replace(hour=end_hour, minute=end_minute))

                
                     # st.markdown("<div style='height: 150px; width: 1px; background-color: lightgray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line

                with col_range:
                    strike_range = st.slider(
                                            "Select strike price range",
                                            min_value=30000,
                                            max_value=400000,
                                            step=500,
                                            value=(70000, 120000)  # Default range
                                        )
                with col_strike:
                    if 'Strike Price' in market_screener_df.columns and not market_screener_df.empty:
                        # Filter the DataFrame for strikes within the selected range
                        filtered_strikes_df = market_screener_df[
                            (market_screener_df['Strike Price'] >= strike_range[0]) & 
                            (market_screener_df['Strike Price'] <= strike_range[1])
                        ]
                        unique_strikes = filtered_strikes_df['Strike Price'].unique()
                        sorted_strikes = sorted(unique_strikes, reverse=True)  # Sort in descending order

                        # Create the multiselect for the filtered strike prices
                        multi_strike_filter = st.multiselect("Select Strikes", options=sorted_strikes)
                    else:
                        # Handle case where no strikes are available
                        multi_strike_filter = st.multiselect("Select Strikes", options=[], default=[], help="No available strikes to select.")
                with col_expiration:
                    market_available_dates = market_screener_df['Expiration Date'].dropna().unique().tolist()

                    # Convert to datetime to sort
                    market_available_dates = pd.to_datetime(market_available_dates, errors='coerce')
                    # Sort the dates
                    sorted_market_available_dates = sorted(market_available_dates)

                    # Optionally convert back to desired string format for display purposes
                    sorted_market_available_dates = [date.strftime("%#d-%b-%y") for date in sorted_market_available_dates]

                    selected_expiration_filter = st.multiselect("Expiration Date", sorted_market_available_dates, key="whatch_exp_filter")
                    
                with col_f4:
                        roww1 = st.container()
                        roww2 = st.container()
                        with roww1:
                            c4_1 , c4_2 = st.columns(2)
                            with c4_1:
                                show_sides_buy = st.checkbox("BUY", value=True, key='show_buys')
                            with c4_2:
                                show_sides_sell = st.checkbox("SELL", value=True, key='show_sells')
                        with roww2:
                            c4_3 , c4_4 = st.columns(2)
                            with c4_3:
                                show_type_call = st.checkbox("Call", value=True, key='show_calss')
                            with c4_4:
                                show_type_put = st.checkbox("Put", value=True, key='show_puts')


            start_strike, end_strike = strike_range  # Unpack the tuple to get start and end values
            

            #st.markdown("---") 

            if not market_screener_df.empty:
                # Ensure 'Entry Date' is in datetime format
                market_screener_df['Entry Date'] = pd.to_datetime(market_screener_df['Entry Date'], errors='coerce')

                # Check for any NaT values
                if market_screener_df['Entry Date'].isna().any():
                    st.warning("Some entries in the 'Entry Date' column were invalid and have been set to NaT.")
                
              
                # Initial filtering by strike price and date range
                filtered_df = market_screener_df[
                        (market_screener_df['Strike Price'] >= start_strike) &
                        (market_screener_df['Strike Price'] <= end_strike) &
                        (market_screener_df['Entry Date'] >= start_datetime) &
                        (market_screener_df['Entry Date'] <= end_datetime)
                    ]
                
                if selected_expiration_filter:
                     
                     filtered_df = filtered_df[( filtered_df['Expiration Date'].isin(selected_expiration_filter))]

                # Filter by selected strikes
                if multi_strike_filter:
                    filtered_df = filtered_df[filtered_df['Strike Price'].isin(multi_strike_filter)]

                # Apply filtering for buy/sell sides
                sides_to_filter = []
                if show_sides_buy:
                    sides_to_filter.append('BUY')  # append the actual value as per your column data
                if show_sides_sell:
                    sides_to_filter.append('SELL')  # append the actual value as per your column data

                if sides_to_filter:
                    filtered_df = filtered_df[filtered_df['Side'].isin(sides_to_filter)]

                # Apply filtering for call/put types
                types_to_filter = []
                if show_type_call:
                    types_to_filter.append('Call')  # append the actual value as per your column data
                if show_type_put:
                    types_to_filter.append('Put')  # append the actual value as per your column data

                if types_to_filter:
                    filtered_df = filtered_df[filtered_df['Option Type'].isin(types_to_filter)]

                
                total_options, total_amount, total_entry_values= calculate_totals_for_options(filtered_df)
                tabs = st.tabs(["Insights",  "Top Options", "Whales" , "Data table"])

                with tabs[0]:
                    detail_column_1, detail_column_2, detail_column_3 = st.columns([0.2, 0.4, 0.4])
                    with detail_column_1:
                        st.metric(label="Positions Count", value=total_options, delta=None, delta_color="normal", help="Total Number of selected options")
                        st.metric(label="Total Size", value=total_amount, delta=None, delta_color="normal", help="Total size of the selected options")
                        st.metric(label="Total Entry Values", value= total_entry_values, delta=None, delta_color="normal", help="Total Entry Values of the selected options")
                        st.markdown("---") 

                        

                    with detail_column_2:
                        fig_2 = plot_strike_price_vs_size(filtered_df)
                        st.plotly_chart(fig_2)
                    with detail_column_3:
                        fig_3 = plot_stacked_calls_puts(filtered_df)
                        st.plotly_chart(fig_3)
                    st.markdown("---") 

                with tabs[1]:  
                    padding, cal1, cal2,cal3 = st.columns([0.02, 0.7, 0.01 ,0.6])

                    with padding: 
                        st.write("")

                    with cal1: 

                        most_traded_options , top_options_chains = get_most_traded_instruments(filtered_df)
                        fig_pie = plot_most_traded_instruments(most_traded_options)
                        st.plotly_chart(fig_pie)

                    with cal3:
                        
                        st.markdown(f"<p style='font-size: 14px; margin-top: 28px;'></p>", unsafe_allow_html=True) 
                        st.dataframe(top_options_chains, use_container_width=True, hide_index=True)
                      
                    

             #------------------------------------------
             #       public trades insights
             #-----------------------------------------------

                with tabs[2]:

                    whales_fig = plot_identified_whale_trades(filtered_df, min_marker_size=8, max_marker_size=35, min_opacity=0.2, max_opacity=0.9, showlegend=True)
                    st.plotly_chart(whales_fig)


                with tabs[3]:
                        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                        

                    

   
            else:
                st.warning("No trades available for the selected options.")

#------------------------------------------------------------------------------------
#----------------------------------------Profit -------------------------------------
#--------------------------------------------------------------------------------------

        with main_tabs[1]:
            #--------------------------------------------------------------------------------
            #-------------------------------- Poltting Results -----------------------------
            #--------------------------------------------------------------------------------
            # Fetch all options for this strike (just to display)
            df_options_for_strike = fetch_data.get_all_strike_options(currency, strike_price, option_type)
           
            
            st.subheader(f'Analytics for Option {strike_price:.0f} {option_type.upper()} ')
            padding, chart_1, chart_2, chart_3 = st.columns([0.1, 2, 2, 2 ])
            with padding: 
                st.write("")
            with chart_1:
                fig_selected_symbol =  plot_underlying_price_vs_entry_value(recent_public_trades_df, btc_price, premium)
                st.plotly_chart(fig_selected_symbol )
            with chart_2:
                #whales_in_option_fig = plot_whales(recent_public_trades_df, min_count=2, min_avg_size=5, max_marker_size=30, showlegend=False)
                #st.plotly_chart(whales_in_option_fig )
                price_vs_date =  plot_price_vs_entry_date(recent_public_trades_df)
                st.plotly_chart(price_vs_date)
                
            with chart_3:
                
                open_by_expiration_radar = plot_radar_chart(df_options_for_strike)
                st.plotly_chart(open_by_expiration_radar) 
                
                
                                                                                 
            
            
            # Fetch and display the current price
            Other_tab = f'All Other Options for {strike_price:.0f} {option_type.upper()}'
            profit_tabs = st.tabs(["Simulation", "Recent trades", Other_tab])

            with profit_tabs[0]:
                option_details_row = st.container()
                with  option_details_row:
                    col1, vertical_line, paddind_line, col2, col3 = st.columns([2, 0.1, 0.1, 4, 0.5])  # Adjust ratio for centering

                    with col1:
                        select_combo_row = st.container()
                        sececnd_chart_combo_row = st.container()
                        options_paying_row = st.container()

                        with select_combo_row :
                            
                            column1, column2 = st.columns(2)
                            with column1:
                                single_selected_symbol = st.checkbox(label=f'{option_symbol}', value=True, key="Single_option")

                            disable_combinations = True if st.session_state.most_profitable_df.empty else False
                                
                            with column2:
                                combo_profit_df = pd.DataFrame()
                                combo_selected_symbol = st.checkbox(label="with Comboination", value=False, key="combo_option_selectto", disabled=disable_combinations)
                            
                        
                        with sececnd_chart_combo_row:
                            selected_combo = None 
                            if not st.session_state.most_profitable_df.empty and combo_selected_symbol:
                                
                                try:
                                    most_profitable_df = st.session_state.most_profitable_df  # Safe reference
                                    most_profitable_df_copy = most_profitable_df.copy()
                                    most_profitable_df_copy = most_profitable_df_copy.drop(columns=['Underlying Price'], errors='ignore')

                                    combo_symbol_list = most_profitable_df_copy.columns.to_list()
                                    
                                    selected_combo = st.selectbox(label="Select Combo Option:", options=combo_symbol_list, key="Select_combination")

                                    if selected_combo:
                                        try:
                                            selected_combo_instrumnet, selected_combo_side = fetch_data.extract_instrument_info(selected_combo)
                                            
                                            if selected_combo_instrumnet:
                                                combo_detail, _ = fetch_data.fetch_option_data(selected_combo_instrumnet)

                                                if not combo_detail.empty:
                                                    combo_strike_price = combo_detail['strike_price'].values[0]
                                                    combo_option_type = combo_detail['option_type'].values[0]
                                                    combo_expiration_date_str = combo_detail['expiration_date'].values[0]
                                                    combo_expiration_date = pd.to_datetime(combo_expiration_date_str).date()  # Ensure it's converted to date

                                                    combo_data_for_plot = most_profitable_df[selected_combo]
                                                    combo_premium = combo_data_for_plot["Premium"]

                                                    now_utc = datetime.now(timezone.utc).date()

                                                    # Compute total days to expiration (at least 1 to avoid zero)
                                                    combo_time_to_expiration_days = max((combo_expiration_date - now_utc).days, 1)
                                                
                                                    if selected_combo_side == 'BUY':
                                                        combo_breakeven_sell = None
                                                        if combo_option_type == 'call':
                                                            combo_breakeven_buy = combo_premium + combo_strike_price
                                                        elif combo_option_type == 'put':
                                                            combo_breakeven_buy = combo_strike_price - combo_premium

                                                    elif selected_combo_side == 'SELL':
                                                        combo_breakeven_buy = None
                                                        if combo_option_type == 'call':
                                                            combo_breakeven_sell = combo_premium + combo_strike_price
                                                        elif combo_option_type == 'put':
                                                            combo_breakeven_sell = combo_strike_price - combo_premium
                                                        
                                                    combo_days_ahead_slider = st.slider(
                                                        f'Days ahead for {selected_combo_instrumnet}',
                                                        min_value=0,
                                                        max_value=combo_time_to_expiration_days,
                                                        value=1,
                                                        help="Simulate the option's mark-to-market value after X days (before expiration).",
                                                        key="Combo_slider"
                                                    )

                                                    combo_profit_df = calculate_option_profit(combo_detail, combo_days_ahead_slider, combo_quantity, risk_free_rate, change_in_iv, selected_combo_instrumnet)
                                                
                                                else:
                                                    st.error("Error: The fetched option detail is empty.")
                                                    combo_premium = 0
                                                    combo_profit_df = pd.DataFrame()
                                            else:
                                                st.warning("Error: Unable to extract instrument information.")
                                                
                                        except Exception as e:
                                            st.error(f"An error occurred while fetching option data: {e}")
 

                                except Exception as e:
                                    st.error(f"An error occurred while processing the most profitable dataframe: {e}")

                            #else:
                             #   st.warning("No data available or no symbol selected, change Filters ")
                        
                        with options_paying_row:
                            st.markdown("---")

                            if single_selected_symbol:


                                fee_cap = 0.125 * premium
                                initial_fee = 0.0003 * btc_price * quantity
                                final_fee_selected_option = min(initial_fee, fee_cap)

                            else :  
                                final_fee_selected_option = 0
                                premium = 0

                            
                            if selected_combo and selected_combo_instrumnet:
                                fee_cap = 0.125 * combo_premium
                                initial_fee = 0.0003 * btc_price * combo_quantity
                                final_fee_combo= min(initial_fee, fee_cap)

                            else : 
                                final_fee_combo = 0
                                combo_premium = 0


                            total_premium = premium + combo_premium
                            total_fees = final_fee_selected_option + final_fee_combo
   
                            options_title_row = st.container()
                            option_row_1 = st.container()
                            option_row_2 = st.container()
                            total_row = st.container()

                            c1, c2, c3 = st.columns(3)
                            with options_title_row:
                                with c1:
                                    st.markdown(f"<p style='font-size: 14px;'>Options</p>", unsafe_allow_html=True) 
                                with c2: 
                                    st.markdown(f"<p style='font-size: 14px;'>Premium (USD)</p>", unsafe_allow_html=True) 
                                with c3:
                                    st.markdown(f"<p style='font-size: 14px;'>Fee (USD)</p>", unsafe_allow_html=True) 


                                
                            if single_selected_symbol:    
                                with option_row_1:
                                    with c1:
                                        st.markdown(f"<p style='font-size: 14px;'>{option_symbol}</p>", unsafe_allow_html=True) 
                                    with c2:
                                        st.markdown(f"<p style='font-size: 14px;'>{premium:,.1f}</p>", unsafe_allow_html=True) 
                                    with c3: 
                                        st.markdown(f"<p style='font-size: 14px;'>{final_fee_selected_option:,.1f}</p>", unsafe_allow_html=True) 

                            if selected_combo and selected_combo_instrumnet:
                                with option_row_2:
                                    with c1:
                                        st.markdown(f"<p style='font-size: 14px;'>{selected_combo_instrumnet}</p>", unsafe_allow_html=True) 
                                    with c2: 
                                        st.markdown(f"<p style='font-size: 14px;'>{combo_premium:,.1f}</p>", unsafe_allow_html=True) 
                                    with c3: 
                                        st.markdown(f"<p style='font-size: 14px;'>{final_fee_combo:,.1f}</p>", unsafe_allow_html=True) 
                            
                            if selected_combo and single_selected_symbol:
                                with total_row:
                                    with c1:
                                        st.markdown(f"<p style='font-size: 14px;'>Total</p>", unsafe_allow_html=True) 
                                    with c2:
                                        st.markdown(f"<p style='font-size: 14px;'>{total_premium:,.1f}</p>", unsafe_allow_html=True) 
                                    with c3: 
                                        st.markdown(f"<p style='font-size: 14px;'>{total_fees:,.1f}</p>", unsafe_allow_html=True) 


                    # vertical Line
                    with vertical_line: 
                        st.markdown("<div style='height: 500px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  

                    with paddind_line:
                        st.write("")      


                    with col2:
                        chart_tabs= st.tabs(["Profit Chart", "Profit Table"])
                        with chart_tabs[0]:

                            if not st.session_state.most_profitable_df.empty and not combo_profit_df.empty and  combo_selected_symbol and single_selected_symbol:
                                
                                profit_fig_with_combo = plot_option_profit(results_df,
                                                                        combo_profit_df,
                                                                        option_symbol, 
                                                                        selected_combo_instrumnet, 
                                                                        days_ahead_slider,
                                                                        combo_days_ahead_slider,
                                                                        option_side,
                                                                        selected_combo_side, 
                                                                        breakeven_buy, 
                                                                        breakeven_sell,
                                                                        combo_breakeven_buy,
                                                                        combo_breakeven_sell)
                                st.plotly_chart(profit_fig_with_combo)

                            if not st.session_state.most_profitable_df.empty and  combo_selected_symbol and not single_selected_symbol :

                                results_df = pd.DataFrame()
                                profit_fig_with_combo = plot_option_profit(results_df,
                                                                        combo_profit_df,
                                                                        None, 
                                                                        selected_combo_instrumnet, 
                                                                        days_ahead_slider,
                                                                        combo_days_ahead_slider,
                                                                        option_side,
                                                                        selected_combo_side,
                                                                        None, 
                                                                        None,
                                                                        combo_breakeven_buy,
                                                                        combo_breakeven_sell)
                                st.plotly_chart(profit_fig_with_combo)

                            if single_selected_symbol and disable_combinations:
                                combo_profit_df = pd.DataFrame()
                                combo_days_ahead_slider = 0
                                profit_fig = plot_option_profit(results_df,
                                                                combo_profit_df,
                                                                option_symbol, 
                                                                None,  
                                                                days_ahead_slider,
                                                                combo_days_ahead_slider, 
                                                                option_side,
                                                                None,
                                                                breakeven_buy, 
                                                                breakeven_sell, 
                                                                None, 
                                                                None)
                                st.plotly_chart(profit_fig)
                        
                        with chart_tabs[1]:
                            selected_option_df = results_df.copy()
                            selected_option_df = selected_option_df.filter(regex="Underlying|Day")
                            selected_option_df = selected_option_df.rename(columns=lambda col: f'{option_symbol}' if "Day" in col else col)

                            if selected_combo and not combo_profit_df.empty:
                                # Create a copy of combo_profit_df and filter it
                                combo_df = combo_profit_df.copy()
                                combo_df = combo_df.filter(regex=f"(?=.*Day)(?=.*{selected_combo_side})")
                                combo_df_filtered = combo_df.rename(columns=lambda col: f'{selected_combo_instrumnet}' if "Day" in col else col)
                                selected_option_df = pd.concat([selected_option_df, combo_df_filtered], axis=1)  # Concatenate the DataFrames

                            # Display the selected option DataFrame
                            st.dataframe(selected_option_df, use_container_width=True, hide_index=True)


                    with col3:
                        st.write("")
                 
                st.markdown("---")


            with profit_tabs[1]:
                 
                 if not recent_public_trades_df.empty:
                    st.subheader(f'Recent Trades for {strike_price:.0f} {option_type.upper()}')
                    st.dataframe(recent_public_trades_df, use_container_width=True, hide_index=True)
                 else :
                     st.warning(f'No Trade History for {strike_price:.0f} {option_type.upper()}')

            with profit_tabs[2]:
                df_options_for_strike = df_options_for_strike.drop(columns=['strike_price', 'option_type'], errors='ignore')
                st.dataframe(df_options_for_strike, use_container_width=True, hide_index=True)



#--------------------------------------------------------------
#-----------------------Combinations ----------------------------
#-------------------------------------------------------------
        with main_tabs[2]:

            if st.session_state.most_profitable_df.shape[1] > 1:  # Assuming at least 'Underlying Price' and one profit column exists
                    most_profitable_df = st.session_state.most_profitable_df
                    num_combos = most_profitable_df.shape[1]
                    num_combo_to_show = 10  # Number of columns to show per tab
                    num_tabs = (num_combos // num_combo_to_show) + (num_combos % num_combo_to_show > 0)

                            # Create tab names based on the number of combinations
                    tab_names = [f"Combination {i + 1}" for i in range(num_tabs)]

                            # Create the tabs dynamically
                    combo_tabs = st.tabs(tab_names)

                            # Loop through each tab and display the corresponding data
                    for i, tab in enumerate(combo_tabs):
                        with tab:
                                    # Copy the main DataFrame to avoid modifying original
                            display_df = most_profitable_df.copy()
                                    
                                    # Isolate and remove the 'Underlying Price' column
                            underlying_price = display_df.pop('Underlying Price')

                                    # Determine the columns to display in this tab
                            start_col = i * num_combo_to_show
                            end_col = start_col + num_combo_to_show

                                    # Slice the DataFrame for the current tab.
                            columns_to_display = display_df.columns[start_col:end_col]
                                    
                                    # Create a new DataFrame for display without the 'Underlying Price'
                            sliced_df = display_df[list(columns_to_display)]
                                    
                                    # Insert the 'Underlying Price' column at the front
                            sliced_df.insert(0, 'Underlying Price', underlying_price)

                                    # Render the DataFrame with Streamlit
                            styled_results = style_combined_results(sliced_df)  # Pass the full DataFrame

                                    # Render the styled DataFrame in the tab using markdown
                            st.markdown(styled_results.to_html(escape=False), unsafe_allow_html=True)
                
            else:
                st.warning("No combinations meet the criteria, Press Analyze then Filter button.")
            

     


def style_combined_results(combined_results):
    """
    Apply conditional formatting to the combined results DataFrame with richer color distinctions.

    Parameters:
        combined_results (pd.DataFrame): The DataFrame to apply styles on.

    Returns:
        pd.io.formats.style.Styler: A styled DataFrame for better insights.
    """
    def color_profits(value):
        """
        Color profits based on their values using a gradient:
        - Green for positive profits
        - Yellow for values transitioning around zero
        - Red for negative profits
        """
        if value > 200:
            r = 0  # No red
            g = 255  # Full green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Bright green

        elif 0 < value <= 200:
            # Gradient from green (at 200) to yellow (at 0)
            r = int((255 * (200 - value)) / 200)  # More red as value decreases
            g = 255  # Green stays full
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Green to yellow gradient

        elif value == 0:
            # Pure yellow for profit of zero
            r = 255  # Full red
            g = 255  # Full green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Opaque yellow

        elif -100 < value < 0:
            # Gradient from yellow (at 0) to red (at -100)
            r = 255  # Full red
            g = int((255 * (100 + value)) / 100)  # Green decreases as it goes more negative
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Yellow to red gradient

        else:
            # Solid red for values lower than -100
            r = 255  # Full red
            g = 0  # No green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: white'  # Solid red

    # Create a masking function to skip the first row and the "Premium" row
    def apply_color(row):
        # Check if the row name is "Premium", and skip coloring if it is
        if row.name == 'Premium':
            return [''] * len(row)  # No styling
        else:
            return [color_profits(value) for value in row]

    # Apply the styling using Styler.apply() on all columns except the first one
    styled_df = combined_results.style.apply(apply_color, axis=1, subset=combined_results.columns[1:])  # Skip the 'Underlying Price' column

    # Format numeric values with specific precision
    formatted_dict = {
        'Underlying Price': '{:.0f}',  # 0 decimal for underlying price
    }

    # Apply formatting for the 'Underlying Price' column and 1 decimal for other profit columns
    for col in combined_results.columns[1:]:  # Assuming first two are not profit columns
        formatted_dict[col] = '{:.1f}'  # 1 decimal for profit columns

    # Format the styled DataFrame
    styled_df = styled_df.format(formatted_dict)

    return styled_df
      
def run_app():
    # Check if the data refresh thread has been started; if not, initialize it.
    if 'data_refresh_thread' not in st.session_state:
        st.session_state.data_refresh_thread = None

    if st.session_state.data_refresh_thread is None or not st.session_state.data_refresh_thread.is_alive():
        # Start the options data refresh thread
        st.session_state.data_refresh_thread = threading.Thread(target=start_fetching_data_from_api)
        st.session_state.data_refresh_thread.start()

    # Run your Streamlit application
    app()



if __name__ == "__main__":
    run_app()
    # Check for our custom environment flag.
    #if os.environ.get("STREAMLIT_RUN") != "1":
        # Set the flag so that the subprocess knows it's already launched.
       # os.environ["STREAMLIT_RUN"] = "1"
        # Launch the Streamlit app using the current Python interpreter.
       # subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
   # else:
        # We are already running under "streamlit run" – proceed with your app.
        # Start the background thread once (if not already started).
        # Now call your app() function to render the Streamlit interface.
        