import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import gradio as gr
import time # For API rate limiting

warnings.filterwarnings('ignore')

# --- Configuration ---
DEFAULT_CRYPTO_CURRENCY = 'usd'
DEFAULT_DAYS = 364 # Default date range in days
INITIAL_CAPITAL = 10000 # Initial capital for backtesting
COMMISSION_RATE = 0.001 # Transaction commission rate, e.g., 0.1%

# Retaining PREDEFINED_CRYPTOS as requested
PREDEFINED_CRYPTOS = {
    "ethereum": {"coingecko_id": "ethereum", "github_owner": "ethereum", "github_repo": "go-ethereum"},
    "bitcoin": {"coingecko_id": "bitcoin", "github_owner": "bitcoin", "github_repo": "bitcoin"},
    "solana": {"coingecko_id": "solana", "github_owner": "solana-labs", "github_repo": "solana"},
    "polkadot": {"coingecko_id": "polkadot", "github_owner": "paritytech", "github_repo": "polkadot-sdk"},
    "cardano": {"coingecko_id": "cardano", "github_owner": "input-output-hk", "github_repo": "cardano-node"},
}

# --- GitHub Token (Sensitive, should be handled securely in production) ---
# IMPORTANT: Replace with your actual GitHub Token or use environment variables for production.
# GITHUB_TOKEN = 'ghp_YOUR_ACTUAL_GITHUB_TOKEN_HERE' 
GITHUB_TOKEN = '' # Example token, please replace!
headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}

# --- Function to get CoinGecko ID list (Top N by Market Cap, excluding predefined) ---
def get_coingecko_id_list(top_n=100):
    """
    Fetches a list of top N cryptocurrency IDs by market capitalization from CoinGecko API,
    excluding those already present in PREDEFINED_CRYPTOS.
    Args:
        top_n (int): The number of top cryptocurrencies to fetch.
    Returns:
        list: A sorted list of cryptocurrency IDs.
    """
    print(f"\n--- Fetching top {top_n} cryptocurrency IDs from CoinGecko (excluding predefined) ---")
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": DEFAULT_CRYPTO_CURRENCY,
        "order": "market_cap_desc",
        "per_page": top_n + len(PREDEFINED_CRYPTOS) * 2, # Fetch more to account for predefined ones and ensure enough top_n
        "page": 1
    }

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        predefined_ids = set(PREDEFINED_CRYPTOS.keys())
        filtered_ids = []
        for coin in data:
            if coin['id'] not in predefined_ids:
                filtered_ids.append(coin['id'])
            if len(filtered_ids) >= top_n: # Stop once we have enough after filtering
                break
        
        filtered_ids.sort() # Sort alphabetically
        print(f"Successfully fetched {len(filtered_ids)} CoinGecko IDs (top {top_n} excluding predefined).")
        return filtered_ids
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to fetch CoinGecko ID list: {e}")
        return []

# Fetch the dynamic list once when the script starts
DYNAMIC_COINGECKO_IDS = get_coingecko_id_list(top_n=100)

# Construct the final list for the dropdown: PREDEFINED + DYNAMIC + Manual Option
ALL_DROPDOWN_CHOICES = list(PREDEFINED_CRYPTOS.keys()) + DYNAMIC_COINGECKO_IDS + ["Manual GitHub Repo & CoinGecko ID"]
# Fallback if ALL_DROPDOWN_CHOICES becomes empty unexpectedly
if not ALL_DROPDOWN_CHOICES:
    ALL_DROPDOWN_CHOICES = ["bitcoin", "ethereum", "solana", "Manual GitHub Repo & CoinGecko ID"]
    print("Warning: Dropdown choices are empty, using default fallback list.")


# --- 1. Fetch Cryptocurrency Price Data (CoinGecko API) ---
def get_crypto_prices(crypto_id, currency, start_date, end_date):
    print(f"\n--- Starting to fetch {crypto_id.upper()} price data ({currency.upper()}) ---")
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart/range?vs_currency={currency}&from={start_timestamp}&to={end_timestamp}"
    print(f"DEBUG(Price API): Request URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data or 'prices' not in data or not data['prices']:
            print(f"Warning: No price data fetched for {crypto_id.upper()} from CoinGecko. Check CoinGecko ID or date range.")
            return pd.Series(dtype='float64')

        prices = []
        for price_data in data['prices']:
            timestamp, price = price_data
            prices.append({'date': datetime.fromtimestamp(timestamp / 1000), 'price': price})

        df = pd.DataFrame(prices)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')
        df = df.set_index('date').sort_index()
        print(f"Successfully fetched {len(df)} price data points for {crypto_id.upper()}.")
        return df['price']

    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error occurred while fetching {crypto_id.upper()} price: {e} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
        return pd.Series(dtype='float64')
    except requests.exceptions.RequestException as e: # Catch all other request errors
        print(f"Error: Failed to connect to CoinGecko API: {e}")
        return pd.Series(dtype='float64')
    except Exception as e:
        print(f"Error: An unknown error occurred while fetching {crypto_id.upper()} price: {e}")
        return pd.Series(dtype='float64')

# --- 2. Fetch GitHub Commit Data (GitHub API) ---
def get_github_commits(owner, repo, start_date, end_date):
    print(f"\n--- Starting to fetch GitHub Commit data for {owner}/{repo} ---")
    commits_data = []
    page = 1
    per_page = 100

    api_start_date = (start_date - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    api_end_date = (end_date + timedelta(days=1)).replace(hour=23, minute=59, second=59, microsecond=999999)

    since_date_str = api_start_date.isoformat(timespec='seconds') + 'Z'
    until_date_str = api_end_date.isoformat(timespec='seconds') + 'Z'
    print(f"DEBUG(GitHub API): Search range: from {since_date_str} to {until_date_str}")

    try:
        while True:
            url = f"https://api.github.com/repos/{owner}/{repo}/commits?per_page={per_page}&page={page}&since={since_date_str}&until={until_date_str}"
            print(f"DEBUG(GitHub API): Request URL: {url}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            commits = response.json()
            print(f"DEBUG(GitHub API): Received {len(commits)} commits for page {page}.")

            if not commits:
                print(f"DEBUG(GitHub API): No more commits found for page {page}, breaking loop.")
                break

            for commit in commits:
                commit_date_str = commit['commit']['author']['date']
                commit_date = datetime.strptime(commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
                # Only keep commits within the specified date range
                if start_date.replace(hour=0, minute=0, second=0, microsecond=0) <= commit_date.replace(hour=0, minute=0, second=0, microsecond=0) <= end_date.replace(hour=0, minute=0, second=0, microsecond=0):
                    commits_data.append({'date': commit_date})

            if len(commits) < per_page:
                print(f"DEBUG(GitHub API): Less than {per_page} commits received, assuming last page.")
                break

            page += 1
            time.sleep(0.1) # Add a small delay to avoid hitting rate limits too quickly

        if not commits_data:
            print(f"Warning: No Commit data fetched from GitHub repository {owner}/{repo} within the specified date range ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}), despite expanded search range.")
            return pd.Series(dtype='int64')

        df = pd.DataFrame(commits_data)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')
        commit_counts = df.groupby('date').size().reindex(pd.date_range(start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
                                                                        end=end_date.replace(hour=0, minute=0, second=0, microsecond=0),
                                                                        freq='D'), fill_value=0)
        print(f"Successfully fetched {len(df)} GitHub Commit data points, summarized into {len(commit_counts)} days.")
        return commit_counts

    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error occurred while fetching GitHub Commits: {e} (Status code: {response.status_code if 'response' in locals() else 'N/A'})")
        if 'response' in locals() and response.status_code == 404:
            print("Please check if GitHub Owner and Repository names are correct.")
        elif 'response' in locals() and response.status_code == 403:
            print(f"GitHub API rate limit might have been reached (Status code: {response.status_code}). Consider setting GITHUB_TOKEN.")
        return pd.Series(dtype='int64')
    except requests.exceptions.RequestException as e: # Catch all other request errors
        print(f"Error: Failed to connect to GitHub API: {e}")
        return pd.Series(dtype='int64')
    except Exception as e:
        print(f"Error: An unknown error occurred while fetching GitHub Commits: {e}")
        return pd.Series(dtype='int64')

# --- 3. Strategy Definitions ---
def simple_commit_threshold_strategy(commit_series, buy_threshold, sell_threshold):
    """
    Simple trading strategy based on Commit count thresholds.
    Buys when Commit count exceeds buy threshold, sells when below sell threshold.
    Assumes full position buy/sell.
    """
    signals = pd.Series(0, index=commit_series.index, dtype=int)
    holding = 0 # 0: no position, 1: holding position

    if commit_series.empty:
        return signals

    for i in range(len(commit_series)):
        current_date = commit_series.index[i]
        current_commits = commit_series.iloc[i]

        if holding == 0: # No position
            if current_commits >= buy_threshold:
                signals.iloc[i] = 1 # Buy signal
                holding = 1
            else:
                signals.iloc[i] = 0 # Remain no position
        elif holding == 1: # Holding position
            if current_commits <= sell_threshold:
                signals.iloc[i] = -1 # Sell signal
                holding = 0
            else:
                signals.iloc[i] = 0 # Continue holding
    
    return signals

def commit_sma_strategy(commit_series, short_period, long_period):
    """
    Strategy based on Commit count SMA golden cross/death cross.
    Golden cross (short-period SMA crosses above long-period SMA) triggers buy.
    Death cross (short-period SMA crosses below long-period SMA) triggers sell.
    """
    signals = pd.Series(0, index=commit_series.index, dtype=int)
    holding = 0 # 0: no position, 1: holding position

    if commit_series.empty or len(commit_series) < long_period:
        return signals # Not enough data to calculate SMA

    short_sma = commit_series.rolling(window=short_period, min_periods=1).mean()
    long_sma = commit_series.rolling(window=long_period, min_periods=1).mean()

    for i in range(1, len(commit_series)):
        # Ensure SMA values are not NaN
        if pd.isna(short_sma.iloc[i]) or pd.isna(long_sma.iloc[i]):
            continue

        # Golden Cross: Short-period SMA crosses above long-period SMA
        if short_sma.iloc[i-1] <= long_sma.iloc[i-1] and short_sma.iloc[i] > long_sma.iloc[i] and holding == 0:
            signals.iloc[i] = 1 # Buy signal
            holding = 1
        # Death Cross: Short-period SMA crosses below long-period SMA
        elif short_sma.iloc[i-1] >= long_sma.iloc[i-1] and short_sma.iloc[i] < long_sma.iloc[i] and holding == 1:
            signals.iloc[i] = -1 # Sell signal
            holding = 0
        else:
            signals.iloc[i] = 0 # No action
    
    return signals


# --- 4. Backtesting Engine ---
def run_backtest(price_series, signals, initial_capital=INITIAL_CAPITAL, commission_rate=COMMISSION_RATE):
    """
    Performs a simple backtest, calculates cumulative returns and buy/sell points,
    and compares with a Buy and Hold strategy. Calculates detailed trade metrics.
    Returns two cumulative return series: one with commission, one without.
    """
    portfolio_value_with_commission = pd.Series(initial_capital, index=price_series.index)
    portfolio_value_no_commission = pd.Series(initial_capital, index=price_series.index)
    
    current_cash_wc = initial_capital # wc = with commission
    holding_shares_wc = 0 
    
    current_cash_nc = initial_capital # nc = no commission
    holding_shares_nc = 0

    buy_points = []
    sell_points = []
    trades_info = [] # List to store details of completed trades {buy_date, buy_price, sell_date, sell_price, profit_wc, profit_nc, holding_days, commission_cost}

    if price_series.empty or signals.empty:
        # Calculate Buy and Hold Return even if strategy data is empty
        buy_and_hold_return = 0.0
        if not price_series.empty:
            first_price = price_series.iloc[0]
            last_price = price_series.iloc[-1]
            if first_price > 0:
                buy_and_hold_shares = initial_capital / first_price * (1 - commission_rate)
                final_buy_and_hold_value = buy_and_hold_shares * last_price * (1 - commission_rate)
                buy_and_hold_return = (final_buy_and_hold_value / initial_capital - 1) * 100
        
        return (pd.Series(1.0, index=price_series.index), 
                pd.Series(1.0, index=price_series.index), 
                [], [], {
            "Error": "Price or signal data is empty, cannot backtest.",
            "Buy and Hold Strategy Total Return": f"{buy_and_hold_return:.2f}%",
            "Total Trades": 0,
            "Win Rate": "N/A",
            "Average P/L per Trade": "N/A",
            "Profit Factor": "N/A",
            "Average Holding Period": "N/A",
            "Trading Frequency (trades/day)": "N/A",
            "Total Commission Paid": "0.00 USD"
        })

    # Track current open trade
    open_trade = None # {'buy_date': date, 'buy_price': price, 'shares_wc': shares_wc, 'shares_nc': shares_nc}

    # --- Execute the selected strategy backtest ---
    for i in range(len(price_series)):
        current_date = price_series.index[i]
        current_price = price_series.iloc[i]
        signal = signals.iloc[i]

        # Ensure price is not NaN
        if pd.isna(current_price):
            if i > 0:
                portfolio_value_with_commission.iloc[i] = portfolio_value_with_commission.iloc[i-1]
                portfolio_value_no_commission.iloc[i] = portfolio_value_no_commission.iloc[i-1]
            continue

        # Handle Buy Signal
        if signal == 1 and open_trade is None: # Buy signal and currently no open trade
            if current_cash_wc > 0:
                # With commission
                shares_to_buy_wc = current_cash_wc / current_price
                commission_cost_buy_wc = shares_to_buy_wc * current_price * commission_rate
                holding_shares_wc = shares_to_buy_wc * (1 - commission_rate)
                current_cash_wc = 0

                # No commission
                shares_to_buy_nc = current_cash_nc / current_price
                holding_shares_nc = shares_to_buy_nc
                current_cash_nc = 0

                open_trade = {
                    'buy_date': current_date, 
                    'buy_price': current_price, 
                    'shares_wc': holding_shares_wc,
                    'shares_nc': holding_shares_nc
                }
                buy_points.append({'date': current_date, 'price': current_price})

        # Handle Sell Signal
        elif signal == -1 and open_trade is not None: # Sell signal and currently holding position
            # With commission
            revenue_wc = open_trade['shares_wc'] * current_price
            commission_cost_sell_wc = revenue_wc * commission_rate
            current_cash_wc = revenue_wc * (1 - commission_rate)
            
            # No commission
            revenue_nc = open_trade['shares_nc'] * current_price
            current_cash_nc = revenue_nc

            profit_wc = (revenue_wc * (1-commission_rate)) - (open_trade['shares_wc'] * open_trade['buy_price'] * (1+commission_rate)) # Approx. P/L including initial commission
            # More accurate profit calculation for with commission:
            initial_cost_wc = initial_capital if i==0 else portfolio_value_with_commission.iloc[i-1] # This is not accurate, should track actual cost basis per trade
            
            # Recalculating P/L for this specific trade
            # Value of shares bought (including cost of commission for buying)
            actual_buy_cost_wc = (open_trade['shares_wc'] / (1-commission_rate)) * open_trade['buy_price'] 
            actual_sell_proceeds_wc = open_trade['shares_wc'] * current_price * (1-commission_rate)
            trade_profit_wc = actual_sell_proceeds_wc - actual_buy_cost_wc
            
            # P/L without commission for this trade
            trade_profit_nc = (open_trade['shares_nc'] * current_price) - (open_trade['shares_nc'] * open_trade['buy_price'])

            total_commission_cost_for_trade = (open_trade['shares_wc'] / (1-commission_rate)) * open_trade['buy_price'] * commission_rate + open_trade['shares_wc'] * current_price * commission_rate

            trades_info.append({
                'buy_date': open_trade['buy_date'],
                'buy_price': open_trade['buy_price'],
                'sell_date': current_date,
                'sell_price': current_price,
                'profit_wc': trade_profit_wc,
                'profit_nc': trade_profit_nc,
                'holding_days': (current_date - open_trade['buy_date']).days,
                'commission_cost': total_commission_cost_for_trade
            })
            
            holding_shares_wc = 0
            holding_shares_nc = 0
            open_trade = None # Close the trade
            sell_points.append({'date': current_date, 'price': current_price})
        
        # Calculate daily net worth
        if holding_shares_wc > 0:
            portfolio_value_with_commission.iloc[i] = current_cash_wc + (holding_shares_wc * current_price)
            portfolio_value_no_commission.iloc[i] = current_cash_nc + (holding_shares_nc * current_price)
        else:
            portfolio_value_with_commission.iloc[i] = current_cash_wc
            portfolio_value_no_commission.iloc[i] = current_cash_nc
            
    # If there's an open trade at the end of the period, close it at the last price
    if open_trade is not None:
        last_price = price_series.iloc[-1]
        last_date = price_series.index[-1]
        
        # With commission
        revenue_wc = open_trade['shares_wc'] * last_price
        commission_cost_sell_wc = revenue_wc * commission_rate
        current_cash_wc = revenue_wc * (1 - commission_rate)

        # No commission
        revenue_nc = open_trade['shares_nc'] * last_price
        current_cash_nc = revenue_nc

        actual_buy_cost_wc = (open_trade['shares_wc'] / (1-commission_rate)) * open_trade['buy_price'] 
        actual_sell_proceeds_wc = open_trade['shares_wc'] * last_price * (1-commission_rate)
        trade_profit_wc = actual_sell_proceeds_wc - actual_buy_cost_wc
        
        trade_profit_nc = (open_trade['shares_nc'] * last_price) - (open_trade['shares_nc'] * open_trade['buy_price'])

        total_commission_cost_for_trade = (open_trade['shares_wc'] / (1-commission_rate)) * open_trade['buy_price'] * commission_rate + open_trade['shares_wc'] * last_price * commission_rate

        trades_info.append({
            'buy_date': open_trade['buy_date'],
            'buy_price': open_trade['buy_price'],
            'sell_date': last_date,
            'sell_price': last_price,
            'profit_wc': trade_profit_wc,
            'profit_nc': trade_profit_nc,
            'holding_days': (last_date - open_trade['buy_date']).days,
            'commission_cost': total_commission_cost_for_trade
        })
        portfolio_value_with_commission.iloc[-1] = current_cash_wc
        portfolio_value_no_commission.iloc[-1] = current_cash_nc

    # Calculate cumulative returns for both with/without commission
    cumulative_returns_wc = portfolio_value_with_commission / initial_capital
    cumulative_returns_nc = portfolio_value_no_commission / initial_capital

    # --- Calculate Performance Metrics ---
    total_return_wc = (cumulative_returns_wc.iloc[-1] - 1) * 100 if not cumulative_returns_wc.empty else 0
    total_return_nc = (cumulative_returns_nc.iloc[-1] - 1) * 100 if not cumulative_returns_nc.empty else 0

    # Max Drawdown for with commission
    max_peak_wc = cumulative_returns_wc.expanding(min_periods=1).max()
    drawdown_wc = (cumulative_returns_wc / max_peak_wc) - 1
    max_drawdown_wc = drawdown_wc.min() * 100 if not drawdown_wc.empty else 0

    # Calculate Buy and Hold Strategy Returns
    buy_and_hold_return = 0.0
    if not price_series.empty and price_series.iloc[0] > 0:
        first_price = price_series.iloc[0]
        last_price = price_series.iloc[-1]
        buy_and_hold_shares = initial_capital / first_price * (1 - commission_rate)
        final_buy_and_hold_value = buy_and_hold_shares * last_price * (1 - commission_rate)
        buy_and_hold_return = (final_buy_and_hold_value / initial_capital - 1) * 100
        if pd.isna(buy_and_hold_return):
            buy_and_hold_return = 0.0

    # Trade-specific metrics
    total_trades = len(trades_info)
    winning_trades = [t for t in trades_info if t['profit_wc'] > 0]
    losing_trades = [t for t in trades_info if t['profit_wc'] < 0]

    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
    
    avg_profit_loss = (sum(t['profit_wc'] for t in trades_info) / total_trades) if total_trades > 0 else 0
    
    total_profit = sum(t['profit_wc'] for t in winning_trades)
    total_loss = abs(sum(t['profit_wc'] for t in losing_trades))
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') # Infinity if no losses

    avg_holding_period = (sum(t['holding_days'] for t in trades_info) / total_trades) if total_trades > 0 else 0
    
    total_days_backtested = (price_series.index[-1] - price_series.index[0]).days + 1 if len(price_series) > 1 else 1
    trading_frequency = total_trades / total_days_backtested if total_days_backtested > 0 else 0

    total_commission_paid = sum(t['commission_cost'] for t in trades_info)


    performance_metrics = {
        "Selected Strategy Total Return (with commissions)": f"{total_return_wc:.2f}%",
        "Selected Strategy Total Return (without commissions)": f"{total_return_nc:.2f}%",
        "Max Drawdown": f"{max_drawdown_wc:.2f}%",
        "Buy and Hold Strategy Total Return": f"{buy_and_hold_return:.2f}%",
        "Initial Capital": f"{initial_capital} {DEFAULT_CRYPTO_CURRENCY.upper()}",
        "Final Capital (with commissions)": f"{portfolio_value_with_commission.iloc[-1]:.2f} {DEFAULT_CRYPTO_CURRENCY.upper()}",
        "--- Trade Statistics ---": "", # Separator
        "Total Trades": total_trades,
        "Winning Trades": len(winning_trades),
        "Losing Trades": len(losing_trades),
        "Win Rate": f"{win_rate:.2f}%",
        "Average P/L per Trade": f"{avg_profit_loss:.2f} {DEFAULT_CRYPTO_CURRENCY.upper()}",
        "Profit Factor": f"{profit_factor:.2f}" if profit_factor != float('inf') else "Infinity",
        "Average Holding Period": f"{avg_holding_period:.2f} days",
        "Trading Frequency (trades/day)": f"{trading_frequency:.4f}",
        "Total Commission Paid": f"{total_commission_paid:.2f} {DEFAULT_CRYPTO_CURRENCY.upper()}"
    }
    
    return cumulative_returns_wc, cumulative_returns_nc, buy_points, sell_points, performance_metrics


# --- Gradio Interface Function ---
def analyze_crypto_activity(crypto_selection, manual_coingecko_id, manual_owner, manual_repo, 
                            start_date_input, end_date_input, 
                            strategy_choice, buy_threshold_input, sell_threshold_input,
                            short_sma_period_input, long_sma_period_input,
                            apply_commission_to_plot): # New parameter for commission toggle
    print("\n--- DEBUG: analyze_crypto_activity function started ---")
    print(f"DEBUG: crypto_selection: {crypto_selection}")
    print(f"DEBUG: manual_coingecko_id: {manual_coingecko_id}")
    print(f"DEBUG: manual_owner: {manual_owner}")
    print(f"DEBUG: manual_repo: {manual_repo}")
    print(f"DEBUG: start_date_input (type: {type(start_date_input)}): {start_date_input}")
    print(f"DEBUG: end_date_input (type: {type(end_date_input)}): {end_date_input}")
    print(f"DEBUG: strategy_choice: {strategy_choice}")
    print(f"DEBUG: buy_threshold_input: {buy_threshold_input}")
    print(f"DEBUG: sell_threshold_input: {sell_threshold_input}")
    print(f"DEBUG: short_sma_period_input: {short_sma_period_input}") 
    print(f"DEBUG: long_sma_period_input: {long_sma_period_input}")   
    print(f"DEBUG: apply_commission_to_plot: {apply_commission_to_plot}")

    start_dt = start_date_input if start_date_input is not None else datetime.now() - timedelta(days=DEFAULT_DAYS)
    end_dt = end_date_input if end_date_input is not None else datetime.now()

    if start_dt >= end_dt:
        return None, None, "", "End date must be later than start date."

    coingecko_id_to_use = ""
    github_owner_to_use = ""
    github_repo_to_use = ""

    if crypto_selection == "Manual GitHub Repo & CoinGecko ID":
        print("DEBUG: 'Manual GitHub Repo & CoinGecko ID' selected.")
        if not manual_owner or not manual_repo or not manual_coingecko_id:
            return None, None, "", "Please provide GitHub Owner, Repository name, and CoinGecko ID when 'Manual GitHub Repo & CoinGecko ID' is selected."
        coingecko_id_to_use = manual_coingecko_id
        github_owner_to_use = manual_owner
        github_repo_to_use = manual_repo
    elif crypto_selection in PREDEFINED_CRYPTOS:
        print(f"DEBUG: Predefined cryptocurrency selected: {crypto_selection}")
        crypto_info = PREDEFINED_CRYPTOS[crypto_selection]
        coingecko_id_to_use = crypto_info["coingecko_id"]
        github_owner_to_use = crypto_info["github_owner"]
        github_repo_to_use = crypto_info["github_repo"]
    else: # Crypto from dynamic top 100 list (not predefined)
        print(f"DEBUG: Dynamically loaded CoinGecko ID selected (not predefined): {crypto_selection}")
        if not manual_owner or not manual_repo:
             return None, None, "", f"For non-default cryptocurrency '{crypto_selection}', please provide the corresponding GitHub Owner and Repository names."
        coingecko_id_to_use = crypto_selection 
        github_owner_to_use = manual_owner 
        github_repo_to_use = manual_repo 

    print(f"DEBUG: Final CoinGecko ID to use: {coingecko_id_to_use}")
    print(f"DEBUG: Final GitHub Owner to use: {github_owner_to_use}")
    print(f"DEBUG: Final GitHub Repo to use: {github_repo_to_use}")

    print("DEBUG: Starting data fetching...")
    price_series = get_crypto_prices(coingecko_id_to_use, DEFAULT_CRYPTO_CURRENCY, start_dt, end_dt)
    commit_series = get_github_commits(github_owner_to_use, github_repo_to_use, start_dt, end_dt)

    if price_series.empty:
        return None, None, "", "Cannot plot: Failed to fetch cryptocurrency price data. Please check CoinGecko ID or date range."
    if commit_series.empty:
        return None, None, "", "Cannot plot: Failed to fetch GitHub Commit data. Please check GitHub Owner/Repo or date range."

    print("DEBUG: Starting data alignment logic...")
    floored_start_date = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    floored_end_date = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    full_date_range = pd.date_range(start=floored_start_date, end=floored_end_date, freq='D')

    price_series_reindexed = price_series.reindex(full_date_range)
    commit_series_reindexed = commit_series.reindex(full_date_range, fill_value=0)

    combined_df = pd.DataFrame({'price': price_series_reindexed, 'commits': commit_series_reindexed})
    combined_df.dropna(subset=['price'], inplace=True) 

    price_series_aligned = combined_df['price']
    commit_series_aligned = combined_df['commits']

    if price_series_aligned.empty or commit_series_aligned.empty:
        return None, None, "", "Warning: Data became empty after aligning dates and removing no-price data points. This might be due to large time differences in original data or many missing values in one dataset. Please check start and end date settings, or try a shorter date range."

    print("DEBUG: Data ready, starting plot generation and backtesting...")
    
    # --- Execute Strategy Backtest ---
    strategy_signals = pd.Series(0, index=price_series_aligned.index, dtype=int)
    
    if strategy_choice == "Simple Commit Threshold Strategy":
        if buy_threshold_input is None or sell_threshold_input is None:
            return None, None, "", "Please provide both buy and sell thresholds for 'Simple Commit Threshold Strategy'."
        if buy_threshold_input < sell_threshold_input:
            return None, None, "", "Error: Buy threshold must be greater than sell threshold for an effective strategy."
        print(f"DEBUG: Executing 'Simple Commit Threshold Strategy' (Buy: {buy_threshold_input}, Sell: {sell_threshold_input})...")
        strategy_signals = simple_commit_threshold_strategy(commit_series_aligned, buy_threshold_input, sell_threshold_input)
        
    elif strategy_choice == "Commit SMA Strategy": 
        if short_sma_period_input is None or long_sma_period_input is None:
            return None, None, "", "Please provide both short and long SMA periods for 'Commit SMA Strategy'."
        if not isinstance(short_sma_period_input, (int, float)) or short_sma_period_input <= 0:
            return None, None, "", "Error: Short SMA period must be a positive integer."
        if not isinstance(long_sma_period_input, (int, float)) or long_sma_period_input <= 0:
            return None, None, "", "Error: Long SMA period must be a positive integer."
        
        short_sma_period = int(short_sma_period_input)
        long_sma_period = int(long_sma_period_input)

        if short_sma_period >= long_sma_period:
            return None, None, "", "Error: Short SMA period must be less than long SMA period."
        if len(commit_series_aligned) < long_sma_period:
            return None, None, "", f"Error: Insufficient data to calculate SMA (requires at least {long_sma_period} days of Commit data)."
        print(f"DEBUG: Executing 'Commit SMA Strategy' (Short period: {short_sma_period}, Long period: {long_sma_period})...")
        strategy_signals = commit_sma_strategy(commit_series_aligned, short_sma_period, long_sma_period)
    
    elif strategy_choice == "No Strategy":
        strategy_signals = pd.Series(0, index=price_series_aligned.index, dtype=int) # No trade signals

    # Run backtest, get both commission-inclusive and commission-exclusive returns
    cumulative_returns_wc, cumulative_returns_nc, buy_points, sell_points, performance_metrics = run_backtest(
        price_series_aligned, strategy_signals, INITIAL_CAPITAL, COMMISSION_RATE
    )
    
    # Determine which cumulative returns to plot based on UI toggle
    if apply_commission_to_plot:
        cumulative_returns_to_plot = cumulative_returns_wc
        strategy_label = 'Selected Strategy Cumulative Return (with commissions)'
    else:
        cumulative_returns_to_plot = cumulative_returns_nc
        strategy_label = 'Selected Strategy Cumulative Return (without commissions)'
    
    if strategy_choice == "No Strategy":
        cumulative_returns_to_plot = pd.Series(1.0, index=price_series_aligned.index) # Keep flat for No Strategy

    # --- Plot Main Chart (Price and Commits) ---
    fig1, ax1 = plt.subplots(figsize=(14, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{coingecko_id_to_use.capitalize()} Price ({DEFAULT_CRYPTO_CURRENCY.upper()})', color=color)
    ax1.plot(price_series_aligned.index, price_series_aligned.values, color=color, label='Price Trend')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot buy/sell points
    buy_dates = [p['date'] for p in buy_points]
    buy_prices = [p['price'] for p in buy_points]
    sell_dates = [p['date'] for p in sell_points]
    sell_prices = [p['price'] for p in sell_points]

    if buy_points:
        ax1.scatter(buy_dates, buy_prices, marker='^', s=100, color='green', label='Buy Point', zorder=5)
    if sell_points:
        ax1.scatter(sell_dates, sell_prices, marker='v', s=100, color='red', label='Sell Point', zorder=5)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('GitHub Daily Commits', color=color)
    ax2.bar(commit_series_aligned.index, commit_series_aligned.values, color=color, alpha=0.6, width=0.8, label='Daily Commits')
    ax2.tick_params(axis='y', labelcolor=color)

    fig1.suptitle(f'{coingecko_id_to_use.capitalize()} Price Trend vs. {github_owner_to_use}/{github_repo_to_use} Daily Commits\n({start_dt.strftime("%Y-%m-%d")} to {end_dt.strftime("%Y-%m-%d")})', fontsize=16)
    fig1.autofmt_xdate()
    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Combine legends, avoid duplicates
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    
    unique_labels_map = {}
    for handle, label in zip(lines + bars, labels + bar_labels):
        unique_labels_map[label] = handle
    
    final_handles = list(unique_labels_map.values())
    final_labels = list(unique_labels_map.keys())

    ax2.legend(final_handles, final_labels, loc='upper left')

    # --- Plot Strategy Cumulative Returns ---
    fig2, ax_ret = plt.subplots(figsize=(14, 6))
    ax_ret.plot(cumulative_returns_to_plot.index, cumulative_returns_to_plot.values, color='purple', label=strategy_label)
    
    # Add Buy and Hold strategy cumulative returns
    if not price_series_aligned.empty:
        buy_and_hold_cumulative_returns = price_series_aligned / price_series_aligned.iloc[0]
        ax_ret.plot(buy_and_hold_cumulative_returns.index, buy_and_hold_cumulative_returns.values, 
                    color='orange', linestyle='--', label='Buy and Hold Strategy Cumulative Return')

    ax_ret.set_title('Strategy Cumulative Return Curve')
    ax_ret.set_xlabel('Date')
    ax_ret.set_ylabel(f'Cumulative Return (x {DEFAULT_CRYPTO_CURRENCY.upper()})', color='purple')
    ax_ret.tick_params(axis='y', labelcolor='purple')
    ax_ret.grid(True, linestyle='--', alpha=0.6)
    ax_ret.legend(loc='upper left')
    fig2.autofmt_xdate()
    fig2.tight_layout()

    # --- Prepare Performance Metrics Text ---
    performance_text = "\n".join([f"{k}: {v}" for k, v in performance_metrics.items()])

    print("DEBUG: Charts generated, preparing to return results.")
    return fig1, fig2, performance_text, "Charts generated successfully."

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Cryptocurrency Price & GitHub Development Activity Tracker with Backtesting
        Explore the relationship between cryptocurrency price trends and daily GitHub development commits by selecting a cryptocurrency or manually entering a GitHub repository.
        Includes strategy backtesting functionality, showing buy/sell points and strategy returns.
        """
    )

    with gr.Row():
        with gr.Column():
            crypto_choice = gr.Dropdown(
                label="Select Cryptocurrency (predefined first, then top 100 by market cap) or Manually Enter GitHub Repo & CoinGecko ID",
                choices=ALL_DROPDOWN_CHOICES,
                value=list(PREDEFINED_CRYPTOS.keys())[0] if PREDEFINED_CRYPTOS else (ALL_DROPDOWN_CHOICES[0] if ALL_DROPDOWN_CHOICES else None),
                interactive=True
            )
            manual_coingecko_id = gr.Textbox(
                label="Manual CoinGecko ID (e.g., dogecoin)",
                placeholder="Enter CoinGecko ID",
                interactive=True,
                visible=False # Default hidden
            )
            manual_github_owner = gr.Textbox(
                label="GitHub Project Owner (e.g., ethereum)",
                placeholder="Enter GitHub Owner name",
                interactive=True,
                visible=False # Default hidden
            )
            manual_github_repo = gr.Textbox(
                label="GitHub Repository Name (e.g., go-ethereum)",
                placeholder="Enter GitHub Repository name",
                interactive=True,
                visible=False # Default hidden
            )
            start_date_picker = gr.DateTime(
                label="Start Date",
                value=datetime.now() - timedelta(days=DEFAULT_DAYS),
                interactive=True,
                type="datetime"
            )
            end_date_picker = gr.DateTime(
                label="End Date",
                value=datetime.now(),
                interactive=True,
                type="datetime"
            )
            
            gr.Markdown("### Strategy Backtesting Settings")
            strategy_choice = gr.Dropdown(
                label="Select Backtesting Strategy",
                choices=["No Strategy", "Simple Commit Threshold Strategy", "Commit SMA Strategy"], 
                value="No Strategy",
                interactive=True
            )
            buy_threshold_input = gr.Number(
                label="Buy Commit Threshold", 
                value=50, # Default value, adjustable
                info="Buy when daily commits reach or exceed this value",
                interactive=True, 
                visible=False
            )
            sell_threshold_input = gr.Number(
                label="Sell Commit Threshold", 
                value=10, # Default value, adjustable
                info="Sell when daily commits fall below or equal to this value",
                interactive=True, 
                visible=False
            )
            short_sma_period_input = gr.Number( 
                label="Short Commit SMA Period",
                value=5, # Default value
                info="Period (days) for calculating short-term Simple Moving Average",
                interactive=True,
                visible=False
            )
            long_sma_period_input = gr.Number( 
                label="Long Commit SMA Period",
                value=10, # Default value
                info="Period (days) for calculating long-term Simple Moving Average",
                interactive=True,
                visible=False
            )

            apply_commission_checkbox = gr.Checkbox(
                label="Apply Commissions to Strategy Return Curve",
                value=True, # Default to true
                interactive=True,
                info="Toggle to see strategy return curve with/without commissions."
            )

            analyze_button = gr.Button("Analyze and Generate Charts")
        
        with gr.Column():
            output_plot_price_commits = gr.Plot(label="Price Trend vs. GitHub Commit Count (with Buy/Sell Points)")
            output_plot_returns = gr.Plot(label="Strategy Cumulative Return Curve")
            output_performance_metrics = gr.Textbox(label="Strategy Performance Metrics", interactive=False)
            output_message = gr.Textbox(label="Status/Message", interactive=False)

    # Function to toggle visibility of manual input fields and pre-fill if predefined
    def toggle_manual_input_visibility_and_fill(choice):
        if choice == "Manual GitHub Repo & CoinGecko ID":
            return (gr.update(visible=True, value=""),
                    gr.update(visible=True, value=""),
                    gr.update(visible=True, value=""))
        elif choice in PREDEFINED_CRYPTOS:
            crypto_info = PREDEFINED_CRYPTOS[choice]
            return (gr.update(visible=False, value=crypto_info["coingecko_id"]), 
                    gr.update(visible=False, value=crypto_info["github_owner"]),
                    gr.update(visible=False, value=crypto_info["github_repo"]))
        else: 
            return (gr.update(visible=False, value=choice), 
                    gr.update(visible=True, value=""), 
                    gr.update(visible=True, value="")) 

    crypto_choice.change(
        toggle_manual_input_visibility_and_fill,
        inputs=crypto_choice,
        outputs=[manual_coingecko_id, manual_github_owner, manual_github_repo]
    )

    # Function to toggle visibility of strategy parameters
    def toggle_strategy_params_visibility(strategy_choice_value):
        buy_thresh_vis = False
        sell_thresh_vis = False
        short_sma_vis = False
        long_sma_vis = False

        if strategy_choice_value == "Simple Commit Threshold Strategy":
            buy_thresh_vis = True
            sell_thresh_vis = True
        elif strategy_choice_value == "Commit SMA Strategy":
            short_sma_vis = True
            long_sma_vis = True
        
        return (gr.update(visible=buy_thresh_vis), 
                gr.update(visible=sell_thresh_vis), 
                gr.update(visible=short_sma_vis), 
                gr.update(visible=long_sma_vis))

    strategy_choice.change(
        toggle_strategy_params_visibility,
        inputs=strategy_choice,
        outputs=[buy_threshold_input, sell_threshold_input, short_sma_period_input, long_sma_period_input]
    )

    analyze_button.click(
        analyze_crypto_activity,
        inputs=[
            crypto_choice, manual_coingecko_id, manual_github_owner, manual_github_repo,
            start_date_picker, end_date_picker,
            strategy_choice, buy_threshold_input, sell_threshold_input, 
            short_sma_period_input, long_sma_period_input,
            apply_commission_checkbox # Pass the checkbox value
        ],
        outputs=[
            output_plot_price_commits,
            output_plot_returns,
            output_performance_metrics,
            output_message
        ]
    )

demo.launch()