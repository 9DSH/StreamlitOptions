import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import numpy as np

def plot_option_profit(results_df, 
                       combo_df, 
                       selected_option_name, 
                       combo_option_name, 
                       days_ahead_slider,
                       combo_days_ahead_slider, 
                       selected_option_side,
                       combo_option_side, 
                       breakeven_buy,
                       breakeven_sell, 
                       combo_breakeven_buy, 
                       combo_breakeven_sell):
    
    fig = go.Figure()

    # Initialize maximum profit values
    max_profit_buy = 0
    max_profit_sell = 0
    min_profit_buy =0
    min_profit_sell = 0
    
    combo_max_profit_buy = 0
    combo_max_profit_sell = 0
    combo_min_profit_buy = 0
    combo_min_profit_sell = 0
    
    # Validate the presence of results_df and combo_df
    if results_df is not None and not results_df.empty:
        if f'Day {days_ahead_slider} Profit (BUY)' in results_df.columns:
            max_profit_buy = max(results_df[f'Day {days_ahead_slider} Profit (BUY)'].max(), 0)
            min_profit_buy = min(results_df[f'Day {days_ahead_slider} Profit (BUY)'].min(), 0)
        if f'Day {days_ahead_slider} Profit (SELL)' in results_df.columns:
            max_profit_sell = max(results_df[f'Day {days_ahead_slider} Profit (SELL)'].max(), 0)
            min_profit_sell = min(results_df[f'Day {days_ahead_slider} Profit (SELL)'].min(), 0)

    if combo_df is not None and not combo_df.empty:
        if f'Day {combo_days_ahead_slider} Profit (BUY)' in combo_df.columns:
            combo_max_profit_buy = max(combo_df[f'Day {combo_days_ahead_slider} Profit (BUY)'].max(), 0)
            combo_min_profit_buy = min(combo_df[f'Day {combo_days_ahead_slider} Profit (BUY)'].min(), 0)
        if f'Day {combo_days_ahead_slider} Profit (SELL)' in combo_df.columns:
            combo_max_profit_sell = max(combo_df[f'Day {combo_days_ahead_slider} Profit (SELL)'].max(), 0)
            combo_min_profit_sell = min(combo_df[f'Day {combo_days_ahead_slider} Profit (SELL)'].min(), 0)

    # Determine the maximum Y-values for annotations
    max_buy_value = max(max_profit_buy, combo_max_profit_buy)
    max_sell_value = max(max_profit_sell, combo_max_profit_sell)

    # Determine the minimum Y-values for annotations
    min_buy_value = min(min_profit_buy, combo_min_profit_buy)
    min_sell_value = min(min_profit_sell, combo_min_profit_sell)

    # Create a common function to add traces with appropriate labels
    def add_traces(df, is_combo, option_side, days_ahead):
        # Determine the color
        color = 'yellow' if is_combo else 'red'  # Combo lines in yellow; results_df lines in red

        if option_side == 'BUY':
            # PnL for BUY
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df[f'Day {days_ahead} Profit (BUY)'],
                mode='lines',
                name=f'PnL {selected_option_name} (BUY)' if not is_combo else f'PnL {combo_option_name} (BUY)',
                line=dict(color=color, width=2),  # Solid line
                hovertemplate=(f'{selected_option_name} (BUY)' if not is_combo else f'{combo_option_name} (BUY)') + '<br>' +
                              '<b>PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Expiration PnL for BUY
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df['Expiration Profit (BUY)'],
                mode='lines',
                name=f'Expiry PnL {selected_option_name} (BUY)' if not is_combo else f'Expiry PnL {combo_option_name} (BUY)',
                line=dict(color=color, dash='dash'),  # Dashed line
                hovertemplate=(f'{selected_option_name} (BUY)' if not is_combo else f'{combo_option_name} (BUY)') + '<br>' +
                              '<b>Expiry PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Breakeven line for BUY from results_df if it's not None
            if breakeven_buy is not None:
                fig.add_shape(
                    type="line",
                    x0=breakeven_buy,
                    y0=min_buy_value,
                    x1=breakeven_buy,
                    y1=max_buy_value,  # Positioning up to the maximum Y-value
                    line=dict(color='rgba(255, 0, 0, 0.7)' , width=2, dash="dot")
                )
                # Add label for breakeven of results_df
                fig.add_annotation(
                    x=breakeven_buy,
                    y=max_buy_value ,  # Offset above the max Y-value
                    text=f'Breakeven: {breakeven_buy:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

            # Breakeven line for BUY from combo_df if it's not None
            if combo_breakeven_buy is not None:
                fig.add_shape(
                    type="line",
                    x0=combo_breakeven_buy,
                    y0=min_buy_value,
                    x1=combo_breakeven_buy,
                    y1=max_buy_value,  # Positioning up to the maximum Y-value
                    line=dict(color="rgba(255, 255, 0, 0.7)", width=2, dash="dot")  # Yellow for combo breakeven
                )
                # Add label for breakeven of combo_df
                fig.add_annotation(
                    x=combo_breakeven_buy,
                    y=max_buy_value * 1.10,  # Offset above the max Y-value
                    text=f'Breakeven: {combo_breakeven_buy:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

        elif option_side == 'SELL':
            # PnL for SELL
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df[f'Day {days_ahead} Profit (SELL)'],
                mode='lines',
                name=f'PnL {selected_option_name} (SELL)' if not is_combo else f'PnL {combo_option_name} (SELL)',
                line=dict(color=color, width=2),  # Solid line
                hovertemplate=(f'{selected_option_name} (SELL)' if not is_combo else f'{combo_option_name} (SELL)') + '<br>' +
                              '<b>PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Expiration PnL for SELL
            fig.add_trace(go.Scatter(
                x=df['Underlying Price'], 
                y=df['Expiration Profit (SELL)'],
                mode='lines',
                name=f'Expiry PnL {selected_option_name} (SELL)' if not is_combo else f'Expiry PnL {combo_option_name} (SELL)',
                line=dict(color=color, dash='dash'),  # Dashed line
                hovertemplate=(f'{selected_option_name} (SELL)' if not is_combo else f'{combo_option_name} (SELL)') + '<br>' +
                              '<b>Expiry PnL</b>: %{y:.2f}<br>' +
                              '<extra></extra>'
            ))

            # Breakeven line for SELL from results_df if it's not None
            if breakeven_sell is not None:
                fig.add_shape(
                    type="line",
                    x0=breakeven_sell,
                    y0=min_sell_value,
                    x1=breakeven_sell,
                    y1=max_sell_value,  # Positioning up to the maximum Y-value
                    line=dict(color='rgba(255, 0, 0, 0.7)' , width=2, dash="dot")
                )
                # Add label for breakeven of results_df
                fig.add_annotation(
                    x=breakeven_sell,
                    y=min_sell_value * 0.95,  # Offset above the max Y-value
                    text=f'Breakeven: {breakeven_sell:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

            # Breakeven line for SELL from combo_df if it's not None
            if combo_breakeven_sell is not None:
                fig.add_shape(
                    type="line",
                    x0=combo_breakeven_sell,
                    y0=min_sell_value,
                    x1=combo_breakeven_sell,
                    y1=max_sell_value,  # Positioning up to the maximum Y-value
                    line=dict(color="rgba(255, 255, 0, 0.7)", width=2, dash="dot")
                )
                # Add label for breakeven of combo_df
                fig.add_annotation(
                    x=combo_breakeven_sell,
                    y=max_sell_value * 0.95,  # Offset above the max Y-value
                    text=f'Breakeven: {combo_breakeven_sell:.2f}',
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-10,
                    font=dict(size=12),  # Set font size to 12 for better visibility
                    bgcolor='rgba(255, 255, 255, 0)',  # Transparent background
                    bordercolor='black',
                    borderwidth=1
                )

    # Add traces for Buy Options from the combo_df only if combo_df is not empty
    if combo_option_side == "BUY" and combo_df is not None and not combo_df.empty:
        add_traces(combo_df, True, 'BUY', combo_days_ahead_slider)

    # Add traces for Sell Options from the combo_df only if combo_df is not empty
    if combo_option_side == "SELL" and combo_df is not None and not combo_df.empty:
        add_traces(combo_df, True, 'SELL', combo_days_ahead_slider)

    # Add traces for Buy Options from the results_df only if results_df is not empty
    if selected_option_side == "BUY" and results_df is not None and not results_df.empty:
        add_traces(results_df, False, 'BUY', days_ahead_slider)

    # Add traces for Sell Options from the results_df only if results_df is not empty
    if selected_option_side == "SELL" and results_df is not None and not results_df.empty:
        add_traces(results_df, False, 'SELL', days_ahead_slider)

    # Update layout for the figure
    fig.update_layout(
        xaxis_title='Underlying Price',
        yaxis_title='Profit',
        legend_title='Options',
        hovermode='x unified',  # Aligns hover information across multiple traces
    )

    return fig


def plot_stacked_calls_puts(df):
    """
    Plot a stacked column chart of total Calls and total Puts against Strike Price.

    Parameters:
        df (pd.DataFrame): DataFrame containing options data with 'Strike Price', 'Option Type', and 'Side' columns.
    """
    # Create a DataFrame to hold counts of Calls and Puts by Strike Price
    plot_data = df.copy()

    # Create columns to identify Call and Put options
    plot_data['Is Call'] = plot_data['Option Type'].str.lower() == 'call'
    plot_data['Is Put'] = plot_data['Option Type'].str.lower() == 'put'

    # Group by Strike Price and Option Type to sum counts, buys, and sells
    grouped_data = plot_data.groupby(['Strike Price', 'Option Type']).agg(
        Total_Calls=('Is Call', 'sum'),                                    # Count Calls
        Total_Puts=('Is Put', 'sum'),                                     # Count Puts
        Buy_Total=('Side', lambda x: (x == 'BUY').sum()),               # Total Buys
        Sell_Total=('Side', lambda x: (x == 'SELL').sum())              # Total Sells
    ).reset_index()

    # Create an interactive stacked bar chart
    fig = go.Figure()

    # Loop through each option type to create bars
    for opt_type in ['Call', 'Put']:
        option_data = grouped_data[grouped_data['Option Type'] == opt_type]
        
        fig.add_trace(go.Bar(
            x=option_data['Strike Price'],
            y=option_data['Total_Calls'] if opt_type == 'Call' else option_data['Total_Puts'],
            name=f'Total {opt_type}s',
            marker=dict(color='gray' if opt_type == 'Call' else 'red', line=dict(color='rgba(0, 0, 0, 0)', width=0)),  # No border
            hovertemplate=f'Strike Price: %{{x}}<br>Total {opt_type}s: %{{y}}<br>Total Buys: %{{customdata[0]}}<br>Total Sells: %{{customdata[1]}}<extra></extra>',  # Update hover information for each type
            customdata=option_data[['Buy_Total', 'Sell_Total']].values  # Pass custom data for hover
        ))

    # Update layout settings
    fig.update_layout(
        title='Open Interest by Strike',
        xaxis_title='Strike Price',
        yaxis_title='Total Number',
        barmode='stack',  # Set the bar mode to stack
        template="plotly_white"  # Clean background
    )

    return fig

def plot_strike_price_vs_size(filtered_df):
    fig = go.Figure()

    # Create a hover template that includes Strike Price, Size, Instrument, and Side
    hover_text = (
        "Strike Price: " + filtered_df['Strike Price'].astype(str) + "<br>" +
        "Size: " + filtered_df['Size'].astype(str) + "<br>" +
        "Instrument: " + filtered_df['Instrument'] + "<br>" +
        "Underlying Price: " + filtered_df['Underlying Price'].astype(str) + "<br>" +
        "Entry Date: " + filtered_df['Entry Date'].astype(str) + "<br>" +
        "Side: " + filtered_df['Side']

    )

    fig.add_trace(go.Scatter(
        x=filtered_df['Strike Price'],
        y=filtered_df['Size'],
        mode='markers',
        marker=dict(color='red', size=10, opacity=0.7),
        name='Strike Size',
        hoverinfo='text',  # Use custom hover text
        hovertext=hover_text  # Set the custom hover text
    ))

    # Update the layout of the plot
    fig.update_layout(
        title='Size by Strike Price ',
        xaxis_title='Strike Price',
        yaxis_title='Size',
        showlegend=True,
        template="plotly_white"  # Use a clean white template
    )
    return fig

def plot_radar_chart(df_options_for_strike):
    # Check if required columns exist
    if 'expiration_date' not in df_options_for_strike.columns or 'open_interest' not in df_options_for_strike.columns:
        print("DataFrame must contain 'expiration_date' and 'open_interest' columns.")
        return

    # Convert 'expiration_date' to datetime
    exp_dates = pd.to_datetime(df_options_for_strike['expiration_date'])

    # Prepare labels:
    # 'categories' for plotting (in original format)
    categories = exp_dates.dt.strftime('%m/%d/%Y').tolist()
    # 'formatted_categories' for display (e.g., '4 July 25')
    formatted_categories = exp_dates.dt.strftime('%#d %B').tolist()  # Use '%-d' on Unix-based systems

    # Extract values for the radar chart
    values = df_options_for_strike['open_interest'].tolist()

    # Close the radar chart by repeating the first value
    values += values[:1]
    categories += categories[:1]
    formatted_categories += formatted_categories[:1]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Open Interest',
        line=dict(color='red', width=2)  # Change color of the line to red
    ))

    # Update the layout of the radar chart
    fig.update_layout(
        title='Open Interest by Expiration Date',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',  # Set plot background to black
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 10],  # Adjust the range as necessary
            ),
            angularaxis=dict(
                tickcolor='white',  # Color of the angular ticks
                tickfont=dict(color='white'),  # Tick font color
                tickvals=categories,       # Original values for proper plotting
                ticktext=formatted_categories  # Formatted text for display
            )
        ),
        showlegend=True
    )

    return fig

def plot_public_profit_sums(summed_df):
    """
    Plot the Underlying Price against the Sum of Profits using Plotly.
    
    Parameters:
        summed_df (pd.DataFrame): DataFrame containing 'Underlying Price' and 'Sum of Profits'.
    """
    fig = go.Figure()

    # Add a line trace
    fig.add_trace(go.Scatter(
        x=summed_df['Underlying Price'],
        y=summed_df['Sum of Profits'],
        mode='lines+markers',
        name='Sum of Profits',
        line=dict(shape='linear'),  # Change line shape to linear
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Underlying Price vs. Sum of Profits',
        xaxis_title='Underlying Price',
        yaxis_title='Sum of Profits',
        template='plotly_white'
    )

    return fig

def plot_most_traded_instruments(most_traded):
    """
    Plots a pie chart of the most traded instruments.
    """
    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(labels=most_traded['Instrument'], values=most_traded['Size'], 
                                   hole=0.6)])  # Optional hole for a donut chart
                                   
    fig.update_traces(hoverinfo='label+percent', textinfo='value+percent+label')
    fig.update_layout(title_text='Top 10 Most Traded Instruments by contracts',
                      title_font_size=24)
    
    return fig


def plot_underlying_price_vs_entry_value(df, custom_price=None, custom_entry_value=None):
    # Helper function to format date with error handling
    def format_entry_date(date_obj):
        try:
            # Check if date_obj is a datetime instance
            if isinstance(date_obj, datetime):
                # Format datetime to a desired string format
                return date_obj.strftime('%Y-%m-%d %H:%M:%S')
            else:
                return 'Invalid Date'
        except Exception as e:  # Catching general exception for robustness, can be more specific
            return 'Invalid Date'

    # Create a scatter plot of Entry Value against Underlying Price
    fig = go.Figure()

    # Separate data for BUY and SELL with corresponding colors
    df_sell = df[df['Side'] == 'SELL']
    df_buy = df[df['Side'] == 'BUY']

    # Prepare custom data with formatted Entry Date
    df_sell.loc[:, 'Formatted Entry Date'] = df_sell['Entry Date'].apply(format_entry_date)
    df_buy.loc[:, 'Formatted Entry Date'] = df_buy['Entry Date'].apply(format_entry_date)
    

    # Add SELL data points (red)
    fig.add_trace(go.Scatter(
        x=df_sell['Entry Value'],
        y=df_sell['Underlying Price'],
        mode='markers',
        marker=dict(size=10, 
                    color='red', 
                    opacity=0.6,       
                    line=dict(        
                        color='black', # Color of the border
                        width=1         # Width of the border
                )),
        name='SELL',
        hovertemplate=(  
            '<b>Underlying Price:</b> %{y:.1f}<br>'  
            '<b>Premium:</b> %{x:.1f}<br>'  
            '<b>Entry Date:</b> %{customdata[1]}<br>'  
            '<b>Size:</b> %{customdata[2]}<br>'  
            '<extra></extra>'  
        ),
        customdata=df_sell[['Instrument', 'Formatted Entry Date', 'Size']].values
    ))

    # Add BUY data points (white)
    fig.add_trace(go.Scatter(
        x=df_buy['Entry Value'],
        y=df_buy['Underlying Price'],
        mode='markers',
        marker=dict(
            size=10, 
            color='white', 
            opacity=0.6,       
            line=dict(        
                color='black', # Color of the border
                width=1         # Width of the border
        )),
        name='BUY',
        hovertemplate=(  
            '<b>Underlying Price:</b> %{y:.1f}<br>'  
            '<b>Premium:</b> %{x:.1f}<br>'  
            '<b>Entry Date:</b> %{customdata[1]}<br>'  
            '<b>Size:</b> %{customdata[2]}<br>'  
            '<extra></extra>'  
        ),
        customdata=df_buy[['Instrument', 'Formatted Entry Date', 'Size']].values
    ))

    # Custom point if provided
    if custom_price is not None and custom_entry_value is not None:
        fig.add_trace(go.Scatter(
            x=[custom_entry_value],
            y=[custom_price],
            mode='markers',
            marker=dict(size=15, 
                        color='green', 
                        symbol='circle',       
                        line=dict(        
                            color='black', # Color of the border
                            width=1         # Width of the border
                    )),
            name='Your Option',
            hovertemplate='<b>Your Entry Price:</b> ' + '{:.1f}'.format(custom_price)  + '<br>' +
                          '<b>Your Premium:</b> ' + '{:.1f}'.format(custom_entry_value) + '<br>' +
                          '<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title='Underlying Price vs Premium',
        xaxis_title='Premium',    
        yaxis_title='Underlying Price',       
        template='plotly_dark',
        hoverlabel=dict(bgcolor='black', font_color='white')
    )

    return fig

def plot_price_vs_entry_date(df):
    # Ensure 'Entry Date' is in datetime format
    df['Entry Date'] = pd.to_datetime(df['Entry Date'])

    # Create traces for BUY and SELL
    buy_df = df[df['Side'] == 'BUY']
    sell_df = df[df['Side'] == 'SELL']

    fig = go.Figure()

    # Add BUY trace with transparency
    fig.add_trace(go.Scatter(
        x=buy_df['Entry Date'],
        y=buy_df['Price (USD)'],  # Use 'Price (USD)' here
        mode='lines',  # Use lines
        name='BUY',
        line=dict(color='white', width=2),  # White line
        marker=dict(size=8),
        hovertext=(
            'Underlying Price: ' + buy_df['Underlying Price'].map('{:.1f}'.format).astype(str) + '<br>' +
            'Entry Date: ' + buy_df['Entry Date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '<br>' +  # Include time
            'Price (USD): ' + buy_df['Price (USD)'].map('{:.1f}'.format) + '<br>' +  # Updated here
            'Size: ' + buy_df['Size'].astype(str) + '<br>' +
            'Side: ' + buy_df['Side']
        ),
        hoverinfo='text',
        hoverlabel=dict(bgcolor='black', font=dict(color='white')),  # Set hover background to black
        opacity=0.6  # Set trace transparency here
    ))

    # Add SELL trace with transparency
    fig.add_trace(go.Scatter(
        x=sell_df['Entry Date'],
        y=sell_df['Price (USD)'],  # Use 'Price (USD)' here
        mode='lines',  # Use lines
        name='SELL',
        line=dict(color='red', width=2),  # Red line
        marker=dict(size=8),
        hovertext=(
            'Underlying Price: ' + sell_df['Underlying Price'].map('{:.1f}'.format).astype(str) + '<br>' +
            'Entry Date: ' + sell_df['Entry Date'].dt.strftime('%Y-%m-%d %H:%M:%S') + '<br>' +  # Include time
            'Price (USD): ' + sell_df['Price (USD)'].map('{:.1f}'.format) + '<br>' +  # Updated here
            'Size: ' + sell_df['Size'].astype(str) + '<br>' +
            'Side: ' + sell_df['Side']
        ),
        hoverinfo='text',
        hoverlabel=dict(bgcolor='black', font=dict(color='white')),  # Set hover background to black
        opacity=0.6  # Set trace transparency here
    ))

    # Update layout with transparent background
    fig.update_layout(
        title='Option Price vs Entry Date',
        xaxis_title='Entry Date',
        yaxis_title='Price (USD)',  # Updated y-axis label
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the entire figure
        font=dict(color='white'),  # White font for labels
        xaxis=dict(
            showgrid=False,
            title_standoff=10,
            zerolinecolor='gray'  # Gray zero line
        ),
        yaxis=dict(
            showgrid=False,
            title_standoff=10,
            zerolinecolor='gray'  # Gray zero line
        ),
        hovermode='closest'
    )

    # Show the plot
    return fig

def plot_identified_whale_trades(df, min_marker_size, max_marker_size, min_opacity, max_opacity, showlegend=True):
    # Ensure 'Entry Date' is in datetime format
    df['Entry Date'] = pd.to_datetime(df['Entry Date'])

    # Initialize a list to hold filtered trades
    filtered_trades = []

    # Step 1: Identify outliers based on the IQR method for each strike price
    strike_prices = df['Strike Price'].unique()

    for strike in strike_prices:
        trades_for_strike = df[df['Strike Price'] == strike]

        # Calculate the IQR
        Q1 = trades_for_strike['Entry Value'].quantile(0.25)
        Q3 = trades_for_strike['Entry Value'].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound_value = Q1 - 1.5 * IQR
        upper_bound_value = Q3 + 1.5 * IQR

        # Select outliers that are larger than the upper bound
        outliers = trades_for_strike[trades_for_strike['Entry Value'] > upper_bound_value]

        # Track valid outliers only if they exceed the average size of remaining trades
        remaining_df = trades_for_strike[~trades_for_strike.index.isin(outliers.index)]
        avg_premium_remaining = remaining_df['Entry Value'].mean() if not remaining_df.empty else 0

        valid_premium_outliers = outliers[outliers['Entry Value'] > avg_premium_remaining]

        # Filter upper size 
        Q1 = valid_premium_outliers['Size'].quantile(0.25)
        Q3 = valid_premium_outliers['Size'].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Select outliers that are larger than the upper bound
        outliers_size = valid_premium_outliers[valid_premium_outliers['Size'] > upper_bound]

        # Track valid outliers only if they exceed the average size of remaining trades
        remaining_df_size = valid_premium_outliers[~valid_premium_outliers.index.isin(outliers_size.index)]
        avg_size_remaining = remaining_df_size['Size'].mean() if not remaining_df_size.empty else 0

        valid_outliers = outliers_size[outliers_size['Size'] > avg_size_remaining]

        # Append valid outliers to the filtered trades
        filtered_trades.append(valid_outliers)

    # Combine all filtered trades into a single DataFrame
    filtered_trades_df = pd.concat(filtered_trades) if filtered_trades else pd.DataFrame()

    # Step 4: Further filter to consider only trades with non-null BlockTrade IDs
    block_trade_df = filtered_trades_df[filtered_trades_df['BlockTrade IDs'].notnull()]
    other_trades_df = filtered_trades_df[filtered_trades_df['BlockTrade IDs'].isnull()]

    # Step 5: Group by Entry Date and Strike Price, counting instances and summing sizes
    def group_trades(trades):
        grouped = trades.groupby(['Entry Date', 'Strike Price', 'Side', 'Option Type', 'Expiration Date']).agg(
            total_size=('Entry Value', 'sum'),
            instances=('Entry Value', 'size')
        ).reset_index()
        return grouped

    grouped_block_trades = group_trades(block_trade_df)
    grouped_other_trades = group_trades(other_trades_df)

    # Combine grouped DataFrames
    combined_grouped = pd.concat([grouped_block_trades, grouped_other_trades], ignore_index=True)

    # Function for scaling marker size and opacity
    def compute_marker_size_and_opacity(group_instances, total_size, max_instances, max_total_size):
        # Calculate marker size
        size_scaling = np.interp(group_instances, 
                                  [1, max_instances], 
                                  [min_marker_size, max_marker_size])
        
        # Calculate opacity
        opacity_scaling = np.clip(total_size / max_total_size, min_opacity, max_opacity)
        
        return size_scaling, opacity_scaling

    # Step 6: Visualize the data
    fig = go.Figure()
    
    max_instances = combined_grouped['instances'].max() if not combined_grouped.empty else 1
    max_total_size = combined_grouped['total_size'].max() if not combined_grouped.empty else 1

    for _, group in combined_grouped.iterrows():
        entry_date = group['Entry Date']
        strike_price = group['Strike Price']
        total_size = group['total_size']
        instances_count = group['instances']
        option_type = group['Option Type']
        option_side = group['Side']
        option_expiration = group['Expiration Date']

        # Calculate marker size and opacity
        group_marker_size, opacity = compute_marker_size_and_opacity(instances_count, total_size, max_instances, max_total_size)

        # Check if the current group is in the block trades group
        is_block_trade = not block_trade_df[(block_trade_df['Entry Date'] == entry_date) & 
                                             (block_trade_df['Strike Price'] == strike_price)].empty

        # Set color based on block trade status
        color = 'red' if is_block_trade else 'yellow'
        text = 'white' if is_block_trade else 'black'
        
        # Construct the hover template
        hover_template = (
            'Entry Date: ' + entry_date.strftime("%Y-%m-%d %H:%M:%S") + '<br>' +
            'Strike Price: ' + str(strike_price) + '<br>' +
            'Side: ' + str(option_side) + '<br>' +
            'Type: ' + str(option_type) + '<br>' +
            'Total Premium: ' + f'{total_size:.1f}' + '<br>' +
            'Expiration Date: ' + str(option_expiration) + '<br>' +
            'Instances: ' + str(instances_count) +
            '<extra></extra>'  # Extra will suppress default hover info
        )

        # Add a trace for this specific Entry Date and Strike Price
        fig.add_trace(go.Scatter(
            x=[entry_date],
            y=[strike_price],
            mode='markers',
            marker=dict(size=group_marker_size, opacity=opacity, color=color),
            name=f'Strike: {strike_price} - Instances: {instances_count}',
            hovertemplate=hover_template,
            hoverinfo='text',
            hoverlabel=dict(bgcolor=color, font=dict(color=text))  # Set background color of hover label
        ))

    # Update layout of the figure
    fig.update_layout(
        xaxis_title='Entry Date',
        yaxis_title='Strike Price',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),
        showlegend=showlegend,
        xaxis=dict(showgrid=False, title_standoff=10, zerolinecolor='gray'),
        yaxis=dict(showgrid=False, title_standoff=10, zerolinecolor='gray'),
        hovermode='closest'
    )

    # Show the plot
    return fig