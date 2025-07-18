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

# Retaining PREDEFINED_CRYPTOS as requested
PREDEFINED_CRYPTOS = {
    "ethereum": {"coingecko_id": "ethereum", "github_owner": "ethereum", "github_repo": "go-ethereum"},
    "bitcoin": {"coingecko_id": "bitcoin", "github_owner": "bitcoin", "github_repo": "bitcoin"},
    "solana": {"coingecko_id": "solana", "github_owner": "solana-labs", "github_repo": "solana"},
    "polkadot": {"coingecko_id": "polkadot", "github_owner": "paritytech", "github_repo": "polkadot-sdk"},
    "cardano": {"coingecko_id": "cardano", "github_owner": "input-output-hk", "github_repo": "cardano-node"},
}

# --- GitHub Token (Sensitive, should be handled securely in production) ---
GITHUB_TOKEN = 'xxx' # Replace with your token or remove if not using
headers = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}

# --- Modified: Function to get CoinGecko ID list (Top N by Market Cap, excluding predefined) ---
def get_coingecko_id_list(top_n=100):
    """
    Fetches a list of top N cryptocurrency IDs by market capitalization from CoinGecko API,
    excluding those already present in PREDEFINED_CRYPTOS.
    Args:
        top_n (int): The number of top cryptocurrencies to fetch.
    Returns:
        list: A sorted list of cryptocurrency IDs.
    """
    print(f"\n--- 開始從 CoinGecko 獲取市值前 {top_n} 的加密貨幣 ID 清單 (排除預設) ---")
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": DEFAULT_CRYPTO_CURRENCY,
        "order": "market_cap_desc",
        "per_page": top_n + len(PREDEFINED_CRYPTOS) * 2, # Fetch more to account for predefined ones and ensure enough top_n
        "page": 1
    }

    try:
        response = requests.get(url, params=params)
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
        print(f"成功獲取到 {len(filtered_ids)} 個 CoinGecko ID (市值前 {top_n} 排除預設)。")
        return filtered_ids
    except requests.exceptions.RequestException as e:
        print(f"錯誤：獲取 CoinGecko ID 清單失敗：{e}")
        return []

# Fetch the dynamic list once when the script starts
DYNAMIC_COINGECKO_IDS = get_coingecko_id_list(top_n=100)

# Construct the final list for the dropdown: PREDEFINED + DYNAMIC + Manual Option
ALL_DROPDOWN_CHOICES = list(PREDEFINED_CRYPTOS.keys()) + DYNAMIC_COINGECKO_IDS + ["手動輸入 GitHub 儲存庫與 CoinGecko ID"]
# Fallback if ALL_DROPDOWN_CHOICES becomes empty unexpectedly
if not ALL_DROPDOWN_CHOICES:
    ALL_DROPDOWN_CHOICES = ["bitcoin", "ethereum", "solana", "手動輸入 GitHub 儲存庫與 CoinGecko ID"]
    print("警告：下拉選單選項為空，將使用預設的備用清單。")


# --- 1. Fetch Cryptocurrency Price Data (CoinGecko API) ---
def get_crypto_prices(crypto_id, currency, start_date, end_date):
    print(f"\n--- 開始抓取 {crypto_id.upper()} 價格數據 ({currency.upper()}) ---")
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_id}/market_chart/range?vs_currency={currency}&from={start_timestamp}&to={end_timestamp}"
    print(f"DEBUG(Price API): Request URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if not data or 'prices' not in data or not data['prices']:
            print(f"警告：未從 CoinGecko 獲取到 {crypto_id.upper()} 的價格數據。請檢查 CoinGecko ID 或日期範圍。")
            return pd.Series(dtype='float64')

        prices = []
        for price_data in data['prices']:
            timestamp, price = price_data
            prices.append({'date': datetime.fromtimestamp(timestamp / 1000), 'price': price})

        df = pd.DataFrame(prices)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')
        df = df.set_index('date').sort_index()
        print(f"成功抓取到 {len(df)} 筆 {crypto_id.upper()} 價格數據。")
        return df['price']

    except requests.exceptions.HTTPError as e:
        print(f"錯誤：抓取 {crypto_id.upper()} 價格時發生 HTTP 錯誤：{e} (狀態碼: {response.status_code if 'response' in locals() else 'N/A'})")
        return pd.Series(dtype='float64')
    except requests.exceptions.RequestException as e: # Catch all other request errors
        print(f"錯誤：連接 CoinGecko API 失敗：{e}")
        return pd.Series(dtype='float64')
    except Exception as e:
        print(f"錯誤：抓取 {crypto_id.upper()} 價格時發生未知錯誤：{e}")
        return pd.Series(dtype='float64')

# --- 2. Fetch GitHub Commit Data (GitHub API) ---
def get_github_commits(owner, repo, start_date, end_date):
    print(f"\n--- 開始抓取 GitHub 儲存庫 {owner}/{repo} 的 Commit 數據 ---")
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
                if start_date.replace(hour=0, minute=0, second=0, microsecond=0) <= commit_date.replace(hour=0, minute=0, second=0, microsecond=0) <= end_date.replace(hour=0, minute=0, second=0, microsecond=0):
                    commits_data.append({'date': commit_date})

            if len(commits) < per_page:
                print(f"DEBUG(GitHub API): Less than {per_page} commits received, assuming last page.")
                break

            page += 1
            time.sleep(0.1) # Add a small delay to avoid hitting rate limits too quickly

        if not commits_data:
            print(f"警告：儘管嘗試擴大搜尋範圍，但未在指定日期 ({start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}) 內從 GitHub 儲存庫 {owner}/{repo} 獲取到 Commit 數據。")
            return pd.Series(dtype='int64')

        df = pd.DataFrame(commits_data)
        df['date'] = pd.to_datetime(df['date']).dt.floor('D')
        commit_counts = df.groupby('date').size().reindex(pd.date_range(start=start_date.replace(hour=0, minute=0, second=0, microsecond=0),
                                                                        end=end_date.replace(hour=0, minute=0, second=0, microsecond=0),
                                                                        freq='D'), fill_value=0)
        print(f"成功抓取到 {len(df)} 筆 GitHub Commit 數據，整理為 {len(commit_counts)} 天的統計。")
        return commit_counts

    except requests.exceptions.HTTPError as e:
        print(f"錯誤：抓取 GitHub Commit 時發生 HTTP 錯誤：{e} (狀態碼: {response.status_code if 'response' in locals() else 'N/A'})")
        if 'response' in locals() and response.status_code == 404:
            print("請檢查 GitHub Owner 和 Repository 名稱是否正確。")
        elif 'response' in locals() and response.status_code == 403:
            print(f"可能已達到 GitHub API 速率限制 (狀態碼: {response.status_code})。建議您設定 GITHUB_TOKEN。")
        return pd.Series(dtype='int64')
    except requests.exceptions.RequestException as e: # Catch all other request errors
        print(f"錯誤：連接 GitHub API 失敗：{e}")
        return pd.Series(dtype='int64')
    except Exception as e:
        print(f"錯誤：抓取 GitHub Commit 時發生未知錯誤：{e}")
        return pd.Series(dtype='int64')

# --- Gradio Interface Function ---
def analyze_crypto_activity(crypto_selection, manual_coingecko_id, manual_owner, manual_repo, start_date_input, end_date_input):
    print("\n--- DEBUG: analyze_crypto_activity 函數開始執行 ---")
    print(f"DEBUG: crypto_selection: {crypto_selection}")
    print(f"DEBUG: manual_coingecko_id: {manual_coingecko_id}")
    print(f"DEBUG: manual_owner: {manual_owner}")
    print(f"DEBUG: manual_repo: {manual_repo}")
    print(f"DEBUG: start_date_input (type: {type(start_date_input)}): {start_date_input}")
    print(f"DEBUG: end_date_input (type: {type(end_date_input)}): {end_date_input}")

    start_dt = start_date_input if start_date_input is not None else datetime.now() - timedelta(days=DEFAULT_DAYS)
    end_dt = end_date_input if end_date_input is not None else datetime.now()

    if start_dt >= end_dt:
        return None, "結束日期必須晚於開始日期。"

    coingecko_id_to_use = ""
    github_owner_to_use = ""
    github_repo_to_use = ""

    if crypto_selection == "手動輸入 GitHub 儲存庫與 CoinGecko ID":
        print("DEBUG: 選擇了 '手動輸入 GitHub 儲存庫與 CoinGecko ID'。")
        if not manual_owner or not manual_repo or not manual_coingecko_id:
            return None, "請在選擇 '手動輸入 GitHub 儲存庫與 CoinGecko ID' 時，同時提供 GitHub Owner、Repository 名稱和 CoinGecko ID。"
        coingecko_id_to_use = manual_coingecko_id
        github_owner_to_use = manual_owner
        github_repo_to_use = manual_repo
    elif crypto_selection in PREDEFINED_CRYPTOS:
        print(f"DEBUG: 選擇了預定義的加密貨幣: {crypto_selection}")
        crypto_info = PREDEFINED_CRYPTOS[crypto_selection]
        coingecko_id_to_use = crypto_info["coingecko_id"]
        github_owner_to_use = crypto_info["github_owner"]
        github_repo_to_use = crypto_info["github_repo"]
    else: # Crypto from dynamic top 100 list (not predefined)
        print(f"DEBUG: 選擇了動態載入的 CoinGecko ID (非預定義): {crypto_selection}")
        # manual_coingecko_id 的值實際上是 crypto_selection，但 UI 上是隱藏的
        # 我們需要檢查 manual_owner 和 manual_repo 是否有值
        if not manual_owner or not manual_repo:
             # 如果 manual_owner 或 manual_repo 為空，表示使用者沒有輸入，需要提示
             return None, f"對於非預設的加密貨幣 '{crypto_selection}'，請提供對應的 GitHub Owner 和 Repository 名稱。"
        coingecko_id_to_use = crypto_selection # 直接使用下拉選單選擇的值
        github_owner_to_use = manual_owner # 從 UI 輸入獲取
        github_repo_to_use = manual_repo # 從 UI 輸入獲取


    print(f"DEBUG: 最終使用的 CoinGecko ID: {coingecko_id_to_use}")
    print(f"DEBUG: 最終使用的 GitHub Owner: {github_owner_to_use}")
    print(f"DEBUG: 最終使用的 GitHub Repo: {github_repo_to_use}")

    print("DEBUG: 開始抓取數據...")
    price_series = get_crypto_prices(coingecko_id_to_use, DEFAULT_CRYPTO_CURRENCY, start_dt, end_dt)
    commit_series = get_github_commits(github_owner_to_use, github_repo_to_use, start_dt, end_dt)

    if price_series.empty:
        return None, "無法繪製圖表：未能成功抓取到加密貨幣價格數據。請檢查 CoinGecko ID 或日期範圍。"
    if commit_series.empty:
        return None, "無法繪製圖表：未能成功抓取到 GitHub Commit 數據。請檢查 GitHub Owner/Repo 或日期範圍。"

    print("DEBUG: 開始數據對齊邏輯...")
    floored_start_date = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    floored_end_date = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    full_date_range = pd.date_range(start=floored_start_date, end=floored_end_date, freq='D')

    price_series_reindexed = price_series.reindex(full_date_range)
    commit_series_reindexed = commit_series.reindex(full_date_range, fill_value=0)

    combined_df = pd.DataFrame({'price': price_series_reindexed, 'commits': commit_series_reindexed})
    combined_df.dropna(inplace=True)

    price_series_aligned = combined_df['price']
    commit_series_aligned = combined_df['commits']

    if price_series_aligned.empty or commit_series_aligned.empty:
        return None, "警告：在對齊日期後，數據變為空。可能因為原始數據的時間點差異太大，或某一方數據有較多缺失。請檢查起始和結束日期設定，或嘗試縮短日期範圍。"

    print("DEBUG: 數據準備就緒，開始繪製圖表...")
    fig, ax1 = plt.subplots(figsize=(14, 8))

    color = 'tab:blue'
    ax1.set_xlabel('日期')
    ax1.set_ylabel(f'{coingecko_id_to_use.capitalize()} 價格 ({DEFAULT_CRYPTO_CURRENCY.upper()})', color=color)
    ax1.plot(price_series_aligned.index, price_series_aligned.values, color=color, label='價格走勢')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()

    color = 'tab:red'
    ax2.set_ylabel('GitHub 每日 Commit 數', color=color)
    ax2.bar(commit_series_aligned.index, commit_series_aligned.values, color=color, alpha=0.6, width=0.8, label='每日 Commit')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle(f'{coingecko_id_to_use.capitalize()} 價格走勢 vs. {github_owner_to_use}/{github_repo_to_use} 每日 Commit 數\n({start_dt.strftime("%Y-%m-%d")} 至 {end_dt.strftime("%Y-%m-%d")})', fontsize=16)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines + bars, labels + bar_labels, loc='upper left')

    print("DEBUG: 圖表繪製完成，準備返回結果。")
    return fig, "圖表生成成功。"

# --- Gradio UI Layout ---
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 加密貨幣價格與 GitHub 開發活動追蹤器
        透過選擇加密貨幣或手動輸入 GitHub 儲存庫，來探索其價格走勢與每日開發提交 (Commit) 數量的關係。
        """
    )

    with gr.Row():
        with gr.Column():
            crypto_choice = gr.Dropdown(
                label="選擇加密貨幣 (預設優先顯示，其次為市值前100) 或手動輸入 GitHub 儲存庫與 CoinGecko ID",
                choices=ALL_DROPDOWN_CHOICES,
                value=list(PREDEFINED_CRYPTOS.keys())[0] if PREDEFINED_CRYPTOS else (ALL_DROPDOWN_CHOICES[0] if ALL_DROPDOWN_CHOICES else None),
                interactive=True
            )
            manual_coingecko_id = gr.Textbox(
                label="手動輸入 CoinGecko ID (例如: dogecoin)",
                placeholder="輸入 CoinGecko ID",
                interactive=True,
                visible=False # Default hidden
            )
            manual_github_owner = gr.Textbox(
                label="GitHub 專案擁有者 (例如: ethereum)",
                placeholder="輸入 GitHub 擁有者名稱",
                interactive=True,
                visible=False # Default hidden
            )
            manual_github_repo = gr.Textbox(
                label="GitHub 儲存庫名稱 (例如: go-ethereum)",
                placeholder="輸入 GitHub 儲存庫名稱",
                interactive=True,
                visible=False # Default hidden
            )
            start_date_picker = gr.DateTime(
                label="開始日期",
                value=datetime.now() - timedelta(days=DEFAULT_DAYS),
                interactive=True,
                type="datetime"
            )
            end_date_picker = gr.DateTime(
                label="結束日期",
                value=datetime.now(),
                interactive=True,
                type="datetime"
            )

            analyze_button = gr.Button("分析並生成圖表")
        with gr.Column():
            output_plot = gr.Plot(label="價格走勢 vs. GitHub Commit 數")
            output_message = gr.Textbox(label="狀態/訊息", interactive=False)

    # Function to toggle visibility of manual input fields and pre-fill if predefined
    def toggle_manual_input_visibility_and_fill(choice):
        if choice == "手動輸入 GitHub 儲存庫與 CoinGecko ID":
            # Show all manual fields, clear values
            return (gr.update(visible=True, value=""),
                    gr.update(visible=True, value=""),
                    gr.update(visible=True, value=""))
        elif choice in PREDEFINED_CRYPTOS:
            # Hide manual fields. For manual_coingecko_id, set value to predefined.
            # For owner/repo, set value to predefined and hide them.
            crypto_info = PREDEFINED_CRYPTOS[choice]
            return (gr.update(visible=False, value=crypto_info["coingecko_id"]), # Coingecko ID is part of predefined
                    gr.update(visible=False, value=crypto_info["github_owner"]),
                    gr.update(visible=False, value=crypto_info["github_repo"]))
        else: # Choice from dynamic top 100 list (not predefined)
            # Hide manual_coingecko_id (since choice itself is the ID)
            # Show and clear manual_github_owner/repo
            return (gr.update(visible=False, value=choice), # Pass selected ID as value, but keep hidden
                    gr.update(visible=True, value=""), # Clear owner and show
                    gr.update(visible=True, value="")) # Clear repo and show

    crypto_choice.change(
        toggle_manual_input_visibility_and_fill,
        inputs=crypto_choice,
        outputs=[manual_coingecko_id, manual_github_owner, manual_github_repo]
    )

    analyze_button.click(
        analyze_crypto_activity,
        inputs=[crypto_choice, manual_coingecko_id, manual_github_owner, manual_github_repo, start_date_picker, end_date_picker],
        outputs=[output_plot, output_message]
    )

demo.launch()