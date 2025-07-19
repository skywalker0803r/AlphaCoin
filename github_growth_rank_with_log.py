import requests
import time
import pandas as pd
import os
from datetime import datetime, timedelta
import json
import sys
import random
import logging
from urllib.parse import urlparse
from tqdm import tqdm
from dotenv import load_dotenv

# 設定輸出編碼
sys.stdout.reconfigure(encoding='utf-8')

# 載入環境變數
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log.txt", encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

# 從環境變數獲取 API Keys
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')

if not GITHUB_TOKEN:
    logging.error("GITHUB_TOKEN 環境變數未設定。請檢查您的 .env 文件。")
    sys.exit(1)
if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
    logging.warning("GOOGLE_API_KEY 或 GOOGLE_CSE_ID 環境變數未設定。Google 搜尋備用方案將無法使用。")

GITHUB_HEADERS = {
    'Accept': 'application/vnd.github+json',
    'Authorization': f'token {GITHUB_TOKEN}'
}

SUCCESS_FILE = 'success_symbols.csv'
CSV_FILE = 'github_growth_rank.csv'

def get_top_symbols(limit=100, quote_asset='USDT'):
    """從幣安 API 獲取市值前 N 名的幣種符號。"""
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # 檢查 HTTP 錯誤
        data = response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"從幣安 API 獲取數據失敗: {e}")
        return []

    usdt_pairs = [item for item in data if item['symbol'].endswith(quote_asset)]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
    top_symbols = [item['symbol'][:-len(quote_asset)].lower() for item in sorted_pairs[:limit]]
    return top_symbols

# 修改這函數 如果error是202或429 就是說只需等待再重試就能跑的就重試 若是404那種就直接return None
def fetch_with_retry(url, headers=None, max_retries=99999, initial_delay=5, backoff_factor=2, return_headers=False):
    """
    帶有重試機制的 HTTP 請求函數。
    對於 202 和 429 狀態碼，會進行重試。
    對於 404 及其他 4xx 狀態碼，會直接返回 None。
    新增 return_headers 參數，如果為 True，則返回 response.headers。
    """
    for i in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)

            if response.status_code == 202:  # GitHub API 數據正在計算中
                delay = initial_delay * (backoff_factor ** i) + random.uniform(0, 2)
                logging.info(f"⏳ GitHub 數據正在計算中 (202 Accepted)，等待 {delay:.2f} 秒後重試... (第 {i+1} 次重試)")
                time.sleep(delay)
                continue
            elif response.status_code == 429:  # 請求過於頻繁
                delay = initial_delay * (backoff_factor ** i) + random.uniform(0, 5) # 429 可以給更長的延遲
                logging.warning(f"Too Many Requests (429)，等待 {delay:.2f} 秒後重試... (第 {i+1} 次重試)")
                time.sleep(delay)
                continue
            elif 400 <= response.status_code < 500: # 4xx 客戶端錯誤，除了 429
                if response.status_code == 404:
                    logging.error(f"請求 {url} 失敗: 資源未找到 (404 Not Found)。不再重試。")
                else:
                    logging.error(f"請求 {url} 失敗: 客戶端錯誤 {response.status_code}。不再重試。")
                return None # 對於 4xx 錯誤，直接返回 None

            response.raise_for_status()  # 對於非 2xx 狀態碼（這裡主要是 5xx 錯誤），拋出 HTTPError

            # 新增邏輯：如果 return_headers 為 True，則返回 response.headers
            if return_headers:
                return response.headers

            try:
                return response.json()
            except json.JSONDecodeError as e:
                logging.error(f"解析 JSON 失敗: {e}, URL: {url}, 響應內容: {response.text}")
                if i < max_retries - 1:
                    delay = initial_delay * (backoff_factor ** i) + random.uniform(0, 2)
                    logging.warning(f"JSON 解析失敗，重試 {url} (第 {i+1} 次), 等待 {delay:.2f} 秒...")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"達到最大重試次數，放棄請求 {url} (JSON 解析失敗)。")
                    return None

        except requests.exceptions.RequestException as e:
            logging.error(f"請求 {url} 失敗: {e}")
            if i < max_retries - 1:
                delay = initial_delay * (backoff_factor ** i) + random.uniform(0, 2)
                logging.warning(f"重試 {url} (第 {i+1} 次), 等待 {delay:.2f} 秒...")
                time.sleep(delay)
            else:
                logging.error(f"達到最大重試次數，放棄請求 {url}。")
    return None

def get_coingecko_id(symbol):
    """根據符號獲取 CoinGecko ID。"""
    logging.info("使用get_coingecko_id先等10秒避免限流")
    time.sleep(random.uniform(10, 15))
    url = f"https://api.coingecko.com/api/v3/search?query={symbol}"
    # 修改點：使用 fetch_with_retry
    data = fetch_with_retry(url) # fetch_with_retry 會返回 None 或 JSON 數據
    if data and data.get('coins'): # 使用 .get() 避免 KeyError
        # 嘗試找到精確匹配的symbol或id
        for coin in data['coins']:
            if coin['symbol'].lower() == symbol.lower() or coin['id'].lower() == symbol.lower():
                return coin['id']
        # 如果沒有精確匹配，返回第一個結果
        return data['coins'][0]['id']
    logging.error(f"從 CoinGecko 搜尋 ID 失敗或無結果: {symbol}")
    return None

def fetch_coingecko_github_repo_url(coingecko_id):
    """從 CoinGecko API 獲取 GitHub 儲存庫連結。"""
    logging.info("使用fetch_coingecko_github_repo_url先等10秒避免限流")
    time.sleep(random.uniform(10, 15))
    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}"
    # 修改點：使用 fetch_with_retry
    data = fetch_with_retry(url)
    if data and 'links' in data and 'repos_url' in data['links'] and data['links']['repos_url']:
        # 可能有兩種路徑
        github_urls = [link for link in data['links']['repos_url']["github"] if link and "github.com" in link]
        if github_urls:
            return github_urls[0] # 返回第一個 GitHub 連結
        else:
            github_urls = [link for link in data['links']['repos_url'] if link and "github.com" in link]
            if github_urls:
                return github_urls[0] # 返回第一個 GitHub 連結
    logging.warning(f"❌ CoinGecko 找不到 {coingecko_id} 的 GitHub repo 連結。URL: {url}")
    return None

def Google_Search_github_repo(query, limit=5):
    """使用 Google Custom Search API 搜尋 GitHub 儲存庫連結。"""
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        logging.warning("Google API Keys 未設定，跳過 Google 搜尋。")
        return None

    search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={query}&num={limit}"
    logging.info("避免被google限流 延遲10秒再用google搜尋")
    time.sleep(10) # Keep the delay as it's a specific requirement for Google Search

    # Use fetch_with_retry for the Google Custom Search API call
    data = fetch_with_retry(search_url)

    if data and 'items' in data:
        for item in data['items']:
            link = item.get('link')
            if link and "github.com" in link and "/tree/" not in link and "/blob/" not in link and "/wiki/" not in link and "/topics/" not in link:
                # 嘗試過濾掉非 repo 根目錄的連結
                # 並且確保不是 Gist 或 Pages 這種
                parsed_url = urlparse(link)
                path_parts = [p for p in parsed_url.path.split('/') if p]
                if len(path_parts) >= 2: # 至少有 owner/repo
                    logging.info(f"✅ Google 搜尋找到 GitHub 連結: {link} (Query: {query})")
                    return link
        logging.warning(f"🔍 Google 搜尋未找到相關的 GitHub 儲存庫連結 (Query: {query})")
    else:
        logging.info(f"🔍 Google 搜尋無結果 (Query: {query})")
    return None


def extract_github_owner_repo(repo_url):
    """從 GitHub URL 中提取 owner 和 repository 名稱，並處理組織 URL。"""
    if not repo_url:
        return None, None
    parsed_url = urlparse(repo_url)
    path_parts = [p for p in parsed_url.path.split('/') if p]

    if len(path_parts) >= 2:
        owner = path_parts[0]
        repo = path_parts[1]
        logging.info(f"✅ 成功從 URL 提取 owner:{owner}, repo:{repo} from {repo_url}")
        return owner, repo
    elif len(path_parts) == 1:
        owner = path_parts[0]
        # 如果只有 owner，嘗試查找該 owner 下最有可能的 repo
        logging.warning(f"⚠️ 僅找到 GitHub owner: {owner} from {repo_url}。嘗試查找最可能的儲存庫...")
        return find_most_likely_repo(owner)
    
    logging.warning(f"❌ 無法從 URL 提取 owner/repo (格式不符): {repo_url}")
    return None, None

def find_most_likely_repo(owner):
    """嘗試從 GitHub 組織或用戶頁面找到 commit 數最高的儲存庫。"""

    # 嘗試作為用戶查找
    # GitHub API doesn't directly support sorting by commit count for repo listings.
    # We'll fetch all repos and then individually check commit counts.
    # For simplicity, we'll fetch a reasonable number of repos to check.
    # A more robust solution might involve pagination if there are many repos.
    url = f"https://api.github.com/users/{owner}/repos?per_page=100" # Fetch up to 100 repos
    repos_data = fetch_with_retry(url, GITHUB_HEADERS)

    if repos_data and isinstance(repos_data, list):
        max_commits = -1
        most_committed_repo = None

        for repo in repos_data:
            repo_name = repo['name']
            commits_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits?per_page=1"
            # We only need the 'Link' header to get the total number of commits
            commits_response_headers = fetch_with_retry(commits_url, GITHUB_HEADERS, return_headers=True)

            if commits_response_headers and 'Link' in commits_response_headers:
                link_header = commits_response_headers['Link']
                # Extract the last page number from the Link header
                import re
                last_page_match = re.search(r'page=(\d+)>; rel="last"', link_header)
                if last_page_match:
                    total_commits = int(last_page_match.group(1))
                    if total_commits > max_commits:
                        max_commits = total_commits
                        most_committed_repo = repo_name
            else:
                # If no 'Link' header or other issue, try fetching the first page of commits
                # and count them. This is less efficient but a fallback.
                commits_data = fetch_with_retry(commits_url, GITHUB_HEADERS)
                if commits_data and isinstance(commits_data, list):
                    # If we can't get the total from 'Link', we'll assume the count on the first page
                    # is the total if per_page=1 is used and we are only looking for one commit.
                    # This is a simplification and might not be accurate for very large repos.
                    # For a truly accurate count without 'Link' header, you'd need to paginate
                    # through all commits.
                    # For this purpose, we assume if the first commit is returned, there's at least one.
                    if len(commits_data) > 0:
                        # This fallback is imperfect as it doesn't give total count.
                        # It primarily serves to check if there are *any* commits if 'Link' fails.
                        # We'll skip this fallback for now as it makes the logic complex for true "max commits".
                        pass

        if most_committed_repo:
            logging.info(f"✅ 找到用戶 {owner} commit 數最高的儲存庫: {most_committed_repo} (總提交數: {max_commits})")
            return owner, most_committed_repo

    # 嘗試作為組織查找 (與用戶邏輯相同)
    url = f"https://api.github.com/orgs/{owner}/repos?per_page=100" # Fetch up to 100 repos
    repos_data = fetch_with_retry(url, GITHUB_HEADERS)

    if repos_data and isinstance(repos_data, list):
        max_commits = -1
        most_committed_repo = None

        for repo in repos_data:
            repo_name = repo['name']
            commits_url = f"https://api.github.com/repos/{owner}/{repo_name}/commits?per_page=1"
            commits_response_headers = fetch_with_retry(commits_url, GITHUB_HEADERS, return_headers=True)

            if commits_response_headers and 'Link' in commits_response_headers:
                link_header = commits_response_headers['Link']
                import re
                last_page_match = re.search(r'page=(\d+)>; rel="last"', link_header)
                if last_page_match:
                    total_commits = int(last_page_match.group(1))
                    if total_commits > max_commits:
                        max_commits = total_commits
                        most_committed_repo = repo_name

        if most_committed_repo:
            logging.info(f"✅ 找到組織 {owner} commit 數最高的儲存庫: {most_committed_repo} (總提交數: {max_commits})")
            return owner, most_committed_repo

    logging.warning(f"❌ 無法為 owner: {owner} 找到 commit 數最高的儲存庫。")
    return None, None

def fetch_github_repo_stats(owner, repo):
    """從 GitHub API 獲取儲存庫的提交活動、星數和 Fork 數。"""
    stats_url = f"https://api.github.com/repos/{owner}/{repo}/stats/commit_activity"
    repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"

    logging.info(f"嘗試獲取提交活動: {stats_url} 延遲10秒避免限流")
    time.sleep(random.uniform(10, 15))
    commit_activity = fetch_with_retry(stats_url, GITHUB_HEADERS)
    
    # 處理 404 Not Found 的情況
    if commit_activity is None:
        logging.warning(f"❌ 無法獲取 {owner}/{repo} 的提交活動，可能儲存庫不存在或私有。")
        return None, None, None

    # 將每週的提交數提取出來
    commits_per_week = [week['total'] for week in commit_activity] if commit_activity else []
    
    logging.info(f"嘗試獲取儲存庫資訊: {repo_info_url}")
    repo_info = fetch_with_retry(repo_info_url, GITHUB_HEADERS)
    
    stars = repo_info.get('stargazers_count') if repo_info else None
    forks = repo_info.get('forks_count') if repo_info else None

    if not commits_per_week or stars is None or forks is None:
        logging.warning(f"❌ 未能獲取 {owner}/{repo} 的所有必要統計數據。")
        return None, None, None

    return commits_per_week, stars, forks

def calc_commit_growth(commits_per_week):
    """計算 GitHub 提交活動的成長率。"""
    if len(commits_per_week) < 30: # 至少需要 30 週數據來計算前後 15 週
        logging.warning(f"數據不足 (只有 {len(commits_per_week)} 週)，無法計算成長率。")
        return None

    recent_15w = sum(commits_per_week[-15:])
    early_15w = sum(commits_per_week[-30:-15])

    if early_15w == 0:
        if recent_15w == 0:
            return 0.0
        else:
            return float('inf') # 避免除以零，表示無限成長
    
    growth = ((recent_15w - early_15w) / early_15w) * 100
    return growth

def load_processed_symbols(csv_filename):
    """載入已處理的幣種列表。"""
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        return set(df['symbol'].unique())
    return set()

def update_success_file(symbol):
    """更新成功處理的幣種列表。"""
    with open(SUCCESS_FILE, 'a', encoding='utf-8') as f:
        f.write(symbol + '\n')

def load_success_symbols():
    """載入已成功處理的幣種。"""
    if os.path.exists(SUCCESS_FILE):
        with open(SUCCESS_FILE, 'r', encoding='utf-8') as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def main():
    # 確保 CSV 檔案存在，如果不存在則創建帶有標頭的檔案
    csv_filename = CSV_FILE
    file_exists = os.path.exists(csv_filename)
    if not file_exists:
        # 定義 CSV 檔案的標頭
        header = ['coin_id', 'symbol', 'repo', 'growth_%', 'recent_15w', 'early_15w', 'stars', 'forks']
        df_empty = pd.DataFrame(columns=header)
        df_empty.to_csv(csv_filename, index=False, encoding='utf-8')
        logging.info(f"創建新的 CSV 檔案: {csv_filename}")

    # 載入已處理過的幣種列表
    processed_symbols_in_csv = load_processed_symbols(csv_filename)
    logging.info(f"從 {csv_filename} 載入 {len(processed_symbols_in_csv)} 個已處理符號。")

    # 獲取熱門幣種
    top_symbols = get_top_symbols(limit=100)
    if not top_symbols:
        logging.error("未能獲取熱門幣種列表，程式終止。")
        return

    logging.info(f"找到 {len(top_symbols)} 個熱門幣種符號 前五為:{top_symbols[:5]}")

    for symbol in tqdm(top_symbols, desc="處理幣種"):
        if symbol in processed_symbols_in_csv:
            logging.info(f"找到 {symbol} 存在csv中跳過")
            continue

        logging.info(f"{symbol} 不存在csv中 開始跑分析 嘗試取得CoinGecko ID")
        coingecko_id = get_coingecko_id(symbol)
        if not coingecko_id:
            logging.warning(f"❌ 無法找到 {symbol} 的 CoinGecko ID，跳過。")
            continue
        logging.info(f"找到 {symbol} 的 CoinGecko ID:{coingecko_id}")

        # 1. 嘗試從 CoinGecko 獲取 GitHub URL
        repo_url = fetch_coingecko_github_repo_url(coingecko_id)
        
        # 2. 如果 CoinGecko 沒有提供，嘗試 Google 搜尋
        if not repo_url and GOOGLE_API_KEY and GOOGLE_CSE_ID:
            logging.info(f"CoinGecko 未找到 {coingecko_id} 的 GitHub 連結，嘗試 Google 搜尋...")
            # 嘗試用 CoinGecko ID 和 幣種符號進行搜尋
            repo_url = Google_Search_github_repo(f"{coingecko_id} github")
            if not repo_url:
                repo_url = Google_Search_github_repo(f"{symbol} github")
            if not repo_url:
                logging.warning(f"❌ 無法從 Google 搜尋找到 {coingecko_id} 或 {symbol} 的 GitHub repo 連結。")
        
        if not repo_url:
            logging.warning(f"❌ 無法找到 {coingecko_id} 的 GitHub repo 連結")
            continue

        logging.info(f"找到 {coingecko_id} 的 GitHub repo 連結:{repo_url}")

        # 3. 提取 owner 和 repo
        owner, repo = extract_github_owner_repo(repo_url)
        if not owner or not repo:
            logging.warning(f"❌ 無法獲取 {coingecko_id} 的 GitHub owner/repo 資訊 repo_url:{repo_url}")
            continue
        
        # 4. 根據 owner, repo 拿到 commits, stars, forks
        commits, stars, forks = fetch_github_repo_stats(owner, repo)
        if commits is None:
            logging.warning(f"拿取 {coingecko_id} 的 commits, stars, forks 失敗")
            continue
        logging.info(f"拿取 {coingecko_id} 的 commits, stars, forks 成功")
        
        # 5. 計算 commit 成長
        growth = calc_commit_growth(commits)
        if growth is None:
            logging.warning(f"計算 growth 失敗 coingecko_id:{coingecko_id}")
            continue
        logging.info(f"計算 growth 成功 coingecko_id:{coingecko_id}")
        
        # 6. 新增一筆紀錄
        new_record = [{
            'coin_id': coingecko_id,
            'symbol': symbol,
            'repo': f'{owner}/{repo}',
            'growth_%': growth,
            'recent_15w': sum(commits[-15:]),
            'early_15w': sum(commits[-30:-15]), # 修正為正確的早期15週
            'stars': stars,
            'forks': forks,
        }]

        if new_record:
            new_df = pd.DataFrame(new_record)
            new_df.to_csv(csv_filename, mode='a', header=False, index=False, encoding='utf-8') # header=False 因為第一次已寫入
            processed_symbols_in_csv.add(symbol) # 將新處理的符號添加到已處理集合中
            time.sleep(random.uniform(1, 3)) # 增加隨機延遲，避免過快請求
            logging.info(f"✅ 成功將 {symbol} 的數據寫入 {csv_filename}")

    logging.info("所有熱門幣種處理完畢。")

if __name__ == "__main__":
    main()