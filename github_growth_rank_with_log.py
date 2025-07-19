import requests
import time
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import sys
import random
import logging
import requests
from urllib.parse import urlparse
import sys
from tqdm import tqdm

sys.stdout.reconfigure(encoding='utf-8')
from dotenv import load_dotenv

load_dotenv()

def get_top_symbols(limit=100, quote_asset='USDT'):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(url)
    data = response.json()

    # 過濾出 quoteAsset 是 USDT 的交易對
    usdt_pairs = [item for item in data if item['symbol'].endswith(quote_asset)]

    # 根據成交額 (quoteVolume) 排序
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)

    # 取前 N 名
    top_symbols = [item['symbol'][:-4].lower() for item in sorted_pairs[:limit]]

    return top_symbols

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log.txt", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_HEADERS = {
    'Accept': 'application/vnd.github+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

SUCCESS_FILE = 'success_ids.json'

def load_success_ids():
    try:
        with open(SUCCESS_FILE, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_success_ids(success_ids):
    with open(SUCCESS_FILE, 'w') as f:
        json.dump(list(success_ids), f)

def fetch_with_retry(url, headers, max_retries=999999999):
    attempt = 0
    while attempt < max_retries:
        # 正常情況
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response
        # 正在計算或是限流
        elif response.status_code == 429 or response.status_code == 202:
            wait = (2 ** attempt) * 2 + random.uniform(0.5, 2)
            status_msg = "GitHub 數據正在計算中 (202 Accepted)" if response.status_code == 202 else "被限流"
            logging.warning(f"⏳ {status_msg}，等待 {wait:.1f} 秒後重試... (URL: {url})")
            time.sleep(wait)
            attempt += 1
        # 失敗情況
        else:
            error_message = f"⚠️ 失敗: {response.status_code} - {url}"
            try:
                error_details = response.json()
                error_message += f" - Details: {error_details}"
            except requests.exceptions.JSONDecodeError:
                error_message += f" - Response: {response.text[:200]}..." # Log first 200 chars of non-JSON response
            logging.error(error_message)
            return None # For any other error, return None immediately
    # 達到最大重試
    logging.error(f"❌ 達到最大重試次數 ({max_retries})，放棄請求。URL: {url}")
    return None

COINGECKO_API_BASE_URL = "https://api.coingecko.com/api/v3"

def fetch_coingecko_coin_id(coin_symbol):
    """
    Fetches CoinGecko ID for a given coin symbol.
    """
    time.sleep(10)
    url = f"{COINGECKO_API_BASE_URL}/search?query={coin_symbol}"
    response = fetch_with_retry(url, {}) # CoinGecko public API usually doesn't require headers for search
    if response:
        data = response.json()
        if 'coins' in data and len(data['coins']) > 0:
            # Try to find an exact match for the symbol
            for coin in data['coins']:
                if coin['symbol'].lower() == coin_symbol.lower():
                    return coin['id']
            # If no exact symbol match, return the first one
            return data['coins'][0]['id']
        else:
            logging.warning(f"❌ CoinGecko 搜尋 {coin_symbol} 無結果。URL: {url}")
    return None

def fetch_coingecko_github_link(coingecko_id):
    """
    Fetches GitHub link for a given CoinGecko ID.
    """
    time.sleep(10)
    url = f"{COINGECKO_API_BASE_URL}/coins/{coingecko_id}"
    
    response = fetch_with_retry(url, {}) # CoinGecko public API usually doesn't require headers
    if response:
        data = response.json()
        if 'links' in data and 'repos_url' in data['links'] and 'github' in data['links']['repos_url'] and len(data['links']['repos_url']['github']) > 0:
            return data['links']['repos_url']['github'][0] # Take the first GitHub URL
        else:
            logging.warning(f"❌ CoinGecko 找不到 {coingecko_id} 的 GitHub repo 連結。URL: {url}")
    return None

def get_most_starred_repo(org_url):
    # 解析組織名稱
    parsed = urlparse(org_url)
    org = parsed.path.strip('/')

    # GitHub API 端點
    api_url = f'https://api.github.com/orgs/{org}/repos?per_page=100'

    headers = GITHUB_HEADERS

    response = requests.get(api_url, headers=headers)
    if response.status_code != 200:
        print(f'❌ Failed to fetch repos: {response.status_code} - {response.text}')
        return

    repos = response.json()
    if not repos:
        print('🚫 No repositories found.')
        return

    # 找到 star 最多的 repo
    top_repo = max(repos, key=lambda r: r.get('stargazers_count', 0))
    repo_url = top_repo['html_url']
    return repo_url

def fetch_github_repo_stats(owner, repo):
    time.sleep(10)
    repo_url = f'https://api.github.com/repos/{owner}/{repo}'
    repo_response = fetch_with_retry(repo_url, GITHUB_HEADERS)
    stars = 0
    forks = 0
    if repo_response:
        try:
            repo_data = repo_response.json()
            stars = repo_data.get('stargazers_count', 0)
            forks = repo_data.get('forks_count', 0)
        except json.JSONDecodeError: # More specific exception for JSON errors
            logging.error(f"❌ JSON decode error fetching repo stats for {owner}/{repo}: {repo_response.text}")
        except Exception as e:
            logging.error(f"❌ Unexpected error fetching repo stats for {owner}/{repo}: {e}")

    commits_url = f'https://api.github.com/repos/{owner}/{repo}/stats/commit_activity'
    logging.info(f"嘗試獲取提交活動: {commits_url}") # Debugging: Log the commit activity URL
    commits_response = fetch_with_retry(commits_url, GITHUB_HEADERS)
    commits = None
    if commits_response:
        try:
            weeks = commits_response.json()
            logging.info(f"獲取到提交活動數據類型: {type(weeks)}") # Debugging: Log the type of 'weeks'
            # logging.info(f"獲取到提交活動數據內容: {json.dumps(weeks, indent=2)}") # Debugging: Log full content (careful with large responses)

            if isinstance(weeks, list):
                # Ensure each item in the list has a 'total' key before accessing it
                if all(isinstance(w, dict) and 'total' in w for w in weeks):
                    commits = [w['total'] for w in weeks[-30:]]  # 最後30週
                    logging.info(f"成功提取 {len(commits)} 週的提交總數。")
                else:
                    logging.warning(f"⚠️ 提交活動數據列表中的元素格式不正確，缺少 'total' 鍵或不是字典: {weeks[:5]}...") # Log first few for brevity
            elif isinstance(weeks, dict) and 'message' in weeks and 'documentation_url' in weeks:
                logging.warning(f"⚠️ GitHub API 返回錯誤消息或計算中: {weeks.get('message', '未知錯誤')}")
                # This could be the 202 Accepted case where data is still being computed
            else:
                logging.warning(f"⚠️ 提交活動數據不是預期的列表類型，而是: {type(weeks)} - {weeks}")
        except json.JSONDecodeError:
            logging.error(f"❌ JSON decode error fetching commit activity for {owner}/{repo}: {commits_response.text}")
        except Exception as e:
            # This is where 'unhashable type: 'slice'' would likely be caught if `weeks` is not a list
            logging.error(f"❌ Unexpected error processing commit activity for {owner}/{repo}: {e}")
            # If `weeks` is not a list, `weeks[-30:]` will raise an error.
            # The 'unhashable type: 'slice'' typically happens when you try to use a slice object (like `[-30:]`)
            # as a key in a dictionary or in some other context where only hashable types are allowed.
            # However, in this specific line `weeks[-30:]`, if `weeks` is not a sequence, it would typically
            # raise a TypeError like "object is not subscriptable".
            # The "unhashable type: 'slice'" error might occur *if* `weeks` somehow becomes an unhashable
            # type itself and is then passed to a function that expects a hashable type, which is less likely here.
            # More likely it's a TypeError from subscripting a non-sequence.
            # Regardless, the checks above should help.

    return commits, stars, forks

def calc_commit_growth(commits):
    if len(commits) < 6:
        return None
    early = sum(commits[:15])
    late = sum(commits[-15:])
    if early == 0:
        return float('inf') if late > 0 else 0
    return round((late - early) / early * 100, 2)

def main():
    # 1.取得100種幣的symbol
    top_symbols = get_top_symbols()
    logging.info(f"找到 {len(top_symbols)} 個熱門幣種符號 前五為:{top_symbols[:5]}")

    now = datetime.now().strftime('%Y%m%d')
    csv_filename = f'commit_growth_{now}.csv'
    commit_growth_csv = pd.read_csv(csv_filename)
    file_exists = os.path.isfile(csv_filename)

    for symbol in tqdm(top_symbols):
        # 1.確定該symbol是否已經有資料了
        if symbol in commit_growth_csv['symbol'].values.tolist():
            logging.info(f"找到 {symbol} 存在csv中跳過")
            continue
        logging.info(f"{symbol} 不存在csv中 開始跑分析 嘗試取得CoinGecko ID")

        # 2.根據symbol找CoinGecko ID
        coingecko_id = fetch_coingecko_coin_id(symbol)
        if not coingecko_id:
            logging.warning(f"❌ 無法找到 {symbol} 的 CoinGecko ID")
            continue
        logging.info(f"找到 {symbol} 的 CoinGecko ID:{coingecko_id}")
            
        # 3.根據CoinGecko ID找github_link
        repo_url = fetch_coingecko_github_link(coingecko_id)
        if not repo_url:
            logging.warning(f"❌ 無法找到 {coingecko_id} 的 GitHub repo 連結")
            continue
        logging.info(f"找到 {symbol} 的 GitHub repo 連結:{repo_url}")

        # 4.嘗試解析github_link
        try:
            if isinstance(repo_url, list) and len(repo_url) > 0:
                repo_url = repo_url[0]
            if not isinstance(repo_url, str):
                logging.warning(f"[WARNING] 無法解析 repo: {repo_url} (不是字串)")
                continue
            
            def repo_url_split(repo_url):
                github_prefix = "github.com/"
                start_index = repo_url.find(github_prefix)
                owner = None  # Initialize owner and repo to None
                repo = None   # Initialize owner and repo to None

                if start_index != -1:
                    path_after_github = repo_url[start_index + len(github_prefix):]
                    parts = path_after_github.split('/')
                    if len(parts) >= 2:
                        owner = parts[0]
                        repo = parts[1]
                        # Clean up potential .git suffix
                        if repo.endswith('.git'):
                            repo = repo[:-4]
                    else:
                        logging.warning(f"[WARNING] 無法從 URL 提取 owner/repo (格式不符): {repo_url}")
                        logging.info(f"尋找最有可能的repo: {repo_url}")
                        # Capture the result of the recursive call
                        most_starred_repo_url = get_most_starred_repo(repo_url)
                        if most_starred_repo_url:
                            logging.info(f"找到可能的repo: {most_starred_repo_url}")
                            # Recurse and return the values directly
                            owner, repo = repo_url_split(most_starred_repo_url)
                        else:
                            logging.warning(f"[WARNING] 無法找到最有可能的repo for: {repo_url}")
                else:
                    logging.warning(f"[WARNING] 無法從 URL 提取 owner/repo (無 github.com/): {repo_url}")
                return owner, repo
            
            owner,repo = repo_url_split(repo_url)
        
        except Exception as e:
            logging.warning(f"[WARNING] 無法解析 repo: error code:{e} repo_url:{repo_url}")
            continue
        
        if not owner or not repo:
            logging.warning(f"❌ 無法獲取 {coingecko_id} 的 GitHub owner/repo 資訊 repo_url:{repo_url}")
            continue
        
        # 4.根據owner, repo拿到commits, stars, forks
        commits, stars, forks = fetch_github_repo_stats(owner, repo)
        if commits is None:
            logging.warning(f"拿取 {coingecko_id} 的 commits, stars, forks 失敗")
            continue
        logging.info(f"拿取 {coingecko_id} 的 commits, stars, forks 成功")
        
        
        # 5.計算commit成長
        growth = calc_commit_growth(commits)
        if growth is None:
            logging.warning(f"計算 growth 失敗 coingecko_id:{coingecko_id}")
            continue
        logging.info(f"計算 growth 成功 coingecko_id:{coingecko_id}")
        
        
        # 6.新增一筆紀錄
        new_record = [{
            'coin_id': coingecko_id,
            'symbol': symbol,
            'repo': f'{owner}/{repo}',
            'growth_%': growth,
            'recent_15w': sum(commits[-15:]),
            'early_15w': sum(commits[:15]),
            'stars': stars,
            'forks': forks,
        }]

        if new_record:
            new_df = pd.DataFrame(new_record)
            new_df.to_csv(csv_filename, mode='a', header=not file_exists, index=False)
            if not file_exists: # After the first write, the file exists
                file_exists = True
            logging.info(f"[INFO] 儲存 1 筆結果 {symbol} 到 {csv_filename}")
        else:
            logging.warning("[WARNING] {symbol} 無有效結果")

if __name__ == '__main__':
    main()

