import os
import re
import time
import datetime
from typing import  List, Set, Tuple, Dict
import ast


from tqdm import tqdm 
import pandas as pd
import requests
import numpy as np
import os
import math

#=== 預售屋社區資料、買賣資料取得 ===
# 實價登錄網址組合
def build_complete_urls(base_url, url_fragments):
    """將base URL與fragments組合成完整URL字典
    
    Args:
        base_url (str): 基礎URL
        url_fragments (dict): URL片段字典，key為城市名稱，value為URL片段
        
    Returns:
        dict: 完整URL字典, key為城市名稱, value為完整URL
    """
    return {city: base_url + fragment for city, fragment in url_fragments.items()}

# 由實價登錄網站取得資料
def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 若有錯誤狀況，會引發例外
        data = response.json()
        return pd.DataFrame(data)
    except Exception as e:
        print(f"取得資料時發生錯誤：{e}")
        return pd.DataFrame()  # 回傳空的 DataFrame

# 取得全台資料後合併，並新增縣市及輸入日期欄位
def combined_df(url, input_time):
    # 建立一個空的列表存放各區的 DataFrame
    df_list = []
    city_counts = {}  # 用於記錄每個縣市的資料筆數
    
    print("開始處理各縣市資料：")
    
    # 迴圈走訪所有 URL，並新增表示地區和輸入時間的欄位
    for city_name, uni_url in url.items():
        print(f"處理 {city_name} 中...", end="", flush=True)
        
        df_temp = fetch_data(uni_url)
        if not df_temp.empty:
            df_temp["city_name"] = city_name      # 加入來源區域欄位，便於後續分析
            df_temp["input_time"] = input_time    # 加入從變數名稱提取的時間

            # 立刻重排欄位，讓 city_name 成為第一欄
            cols = df_temp.columns.tolist()
            cols = ["city_name"] + [c for c in cols if c != "city_name"]
            df_temp = df_temp[cols]
            
            # 記錄此縣市的資料筆數
            row_count = len(df_temp)
            city_counts[city_name] = row_count
            # 直接打印當前縣市的資料筆數
            print(f" 找到 {row_count} 筆資料")
        else:
            print(" 找到 0 筆資料")
            
        df_list.append(df_temp)
        time.sleep(1)  # 每次發出請求後暫停 1 秒

    # 利用 pd.concat 合併所有 DataFrame（重置索引）
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 顯示合併後的總筆數
    total_rows = len(combined_df)
    print(f"\n合併後資料總筆數: {total_rows} 筆")
    
    return combined_df

# 讀取儲存的原始資料
def csv_extractor(file_name: str) -> pd.DataFrame:
    try:
        total_rows = sum(1 for _ in open(file_name, 'r', encoding='utf8'))
        print(f'total_rows: {total_rows}')
        
        df_list = []
        with tqdm(total=total_rows, desc="Extracting rows") as pbar:
            for chunk in pd.read_csv(file_name, encoding='utf8', chunksize=10000):
                df_list.append(chunk)
                pbar.update(len(chunk))
        
        df = pd.concat(df_list, ignore_index=True)
        print('\nExtracting finished')
        return df
    except FileNotFoundError:
        print(f'Error: The file {file_name} was not found.')
    except Exception as e:
        print(f'An error occurred: {e}')
    return pd.DataFrame()


#=== 01預售社區資料整理 ===

# 新增行政區欄位：由坐落地址欄位折分出行政區
def parse_admin_region(address):
    # 若不是字串或字串長度為0，直接回傳 None 或空值
    if not isinstance(address, str) or not address:
        return None
    
    # 判斷第二個字是否為「區」
    # 注意：Python 字串的索引從 0 開始
    if len(address) >= 2 and address[1] == "區":
        return address[:2]
    # 判斷第三個字是否為「區」
    elif len(address) >= 3 and address[2] == "區":
        return address[:3]
    # 其餘情況取前三個字(市/鄉/鎮)
    elif len(address) >= 3:
        return address[:3]
    else:
        # 若字串不足三個字，就直接回傳原字串
        return address
    
# 由備查編號清單取出所有的備查編號
def extract_mixed_alphanumeric_ids(text):
    if pd.isna(text):
        return ''
    ids = re.findall(r'\b[A-Z0-9]{10,16}\b', text)
    return ', '.join(ids)

# 比對編欄位及編號清單欄位進而取出建設公司
def extract_company_name(row):
    """
    比對id與idlist中的ID，提取匹配的公司名稱
    """
    target_id = row['編號']
    idlist = row['編號列表']
    
    # 如果idlist是字串格式的列表，需要先轉換
    if isinstance(idlist, str):
        try:
            # 嘗試用ast.literal_eval轉換字串格式的列表
            idlist = ast.literal_eval(idlist)
        except:
            # 如果轉換失敗，返回None
            return None
    
    # 遍歷idlist中的每個項目
    for item in idlist:
        # 按逗號分割字串
        parts = item.split(',')
        if len(parts) >= 3:  # 確保有足夠的部分
            item_id = parts[0].strip()  # ID部分
            company_name = parts[2].strip()  # 公司名稱部分
            
            # 比對ID
            if item_id == target_id:
                return company_name
    
    # 如果沒有找到匹配的ID，返回None
    return None

# 定義函式：尋找自售期間及代銷期間的起始日，若沒有則回傳 None
def find_first_sale_time(text):
    """
    尋找自售期間及代銷期間的起始日，支援多種日期格式
    
    支援格式：
    - 1110701 (7位數字)
    - 111年07月01日 / 111年7月1日 / 111年8月1號
    - 111/07/01 / 111/7/1
    
    Returns:
        str: 標準化的7位數字日期格式 (如: '1110701')，若沒有找到則回傳 None
    """
    if not isinstance(text, str):
        return None
    
    # 1. 先檢查是否已經是7位數字格式
    seven_digit_match = re.search(r"\d{7}", text)
    if seven_digit_match:
        return seven_digit_match.group(0)
    
    # 2. 檢查 "111年07月01日"、"111年7月1日" 或 "111年8月1號" 格式
    year_month_day_match = re.search(r"(\d{3})年(\d{1,2})月(\d{1,2})[日號]", text)
    if year_month_day_match:
        year = year_month_day_match.group(1)
        month = year_month_day_match.group(2).zfill(2)  # 補零到2位
        day = year_month_day_match.group(3).zfill(2)    # 補零到2位
        return f"{year}{month}{day}"
    
    # 3. 檢查 "111/07/01" 或 "111/7/1" 格式
    slash_format_match = re.search(r"(\d{3})/(\d{1,2})/(\d{1,2})", text)
    if slash_format_match:
        year = slash_format_match.group(1)
        month = slash_format_match.group(2).zfill(2)    # 補零到2位
        day = slash_format_match.group(3).zfill(2)      # 補零到2位
        return f"{year}{month}{day}"
    
    # 4. 如果都沒有匹配，回傳 None
    return None

# 型態轉成datetime64
def convert_mixed_date_columns(df, roc_cols=[], ad_cols=[], roc_slash_cols=[]):
    def parse_roc_integer(val):
        if pd.isna(val): return pd.NaT
        try:
            val = str(int(val)).zfill(7)
            y, m, d = int(val[:3]) + 1911, int(val[3:5]), int(val[5:7])
            return pd.Timestamp(f"{y}-{m:02d}-{d:02d}")
        except: return pd.NaT

    def parse_ad_integer(val):
        if pd.isna(val): return pd.NaT
        try:
            val = str(int(val)).zfill(8)
            return pd.to_datetime(val, format="%Y%m%d", errors='coerce')
        except: return pd.NaT

    def parse_roc_slash(val):
        if pd.isna(val): return pd.NaT
        try:
            parts = str(val).split('/')
            y, m, d = int(parts[0]) + 1911, int(parts[1]), int(parts[2])
            return pd.Timestamp(f"{y}-{m:02d}-{d:02d}")
        except: return pd.NaT

    for col in roc_cols:
        df[col] = df[col].apply(parse_roc_integer)

    for col in ad_cols:
        df[col] = df[col].apply(parse_ad_integer)

    for col in roc_slash_cols:
        df[col] = df[col].apply(parse_roc_slash)

    return df


