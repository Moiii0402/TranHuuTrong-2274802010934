from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import re
from datetime import datetime
import os
import subprocess
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Đường dẫn file trên máy cục bộ và HDFS
BASE_PATH = "~/iDragonCloud/ProjectBigData"
INPUT_PATH = os.path.expanduser(f"{BASE_PATH}/Amazon-Products-Cleaned.csv")
PROCESSED_PATH = os.path.expanduser(f"{BASE_PATH}/amazon_processed_data.csv")
RESULTS_PATH = os.path.expanduser(f"{BASE_PATH}/results")
HDFS_BASE_PATH = "hdfs://localhost:9000"
HDFS_INPUT_PATH = f"{HDFS_BASE_PATH}/data/Amazon-Products-Cleaned.csv"
HDFS_RESULTS_PATH = f"{HDFS_BASE_PATH}/results"

# Hàm tạo thư mục trên HDFS với quyền
def create_hdfs_directory_with_permissions(hdfs_path, permissions="755"):
    logger.info(f"Tạo thư mục {hdfs_path} trên HDFS với quyền {permissions}...")
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_path], check=True)
        logger.info(f"Thư mục {hdfs_path} đã được tạo hoặc đã tồn tại.")
        subprocess.run(["hdfs", "dfs", "-chmod", permissions, hdfs_path], check=True)
        logger.info(f"Đã thiết lập quyền {permissions} cho {hdfs_path}.")
        result = subprocess.run(["hdfs", "dfs", "-ls", hdfs_path], capture_output=True, text=True)
        logger.info(f"Trạng thái thư mục: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi tạo thư mục hoặc thiết lập quyền trên HDFS: {e}")
        exit(1)

# Hàm tải file từ HDFS
def download_from_hdfs(hdfs_path, local_path):
    logger.info(f"Tải file từ HDFS {hdfs_path} về {local_path}...")
    try:
        subprocess.run(["hdfs", "dfs", "-get", "-f", hdfs_path, local_path], check=True)
        logger.info(f"Tải file từ HDFS thành công: {local_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi tải file từ HDFS: {e}")
        exit(1)

# Hàm đẩy file lên HDFS
def upload_to_hdfs(local_path, hdfs_path):
    logger.info(f"Đẩy file từ {local_path} lên HDFS tại {hdfs_path}...")
    try:
        subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_path], check=True)
        logger.info(f"Đẩy file lên HDFS thành công: {hdfs_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi đẩy file lên HDFS: {e}")
        exit(1)

# Tạo thư mục kết quả cục bộ nếu chưa tồn tại
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# Các hàm trích xuất dữ liệu từ Amazon
def get_title(soup):
    try:
        title = soup.find("span", attrs={"id":'productTitle'})
        title_value = title.text
        title_string = title_value.strip()
    except AttributeError:
        title_string = ""
    return title_string

def get_price(soup):
    try:
        price_symbol = soup.find("span", attrs={"class":'a-price-symbol'})
        price_whole = soup.find("span", attrs={"class":'a-price-whole'})
        price_fraction = soup.find("span", attrs={"class":'a-price-fraction'})
        if price_symbol and price_whole and price_fraction:
            symbol = price_symbol.text.strip()
            whole = price_whole.text.strip().replace(',', '')
            fraction = price_fraction.text.strip()
            price = f"{symbol}{whole}.{fraction}"
            return price
        price_element = soup.find("span", attrs={"class":'a-price'})
        if price_element:
            symbol = price_element.find("span", attrs={"class":'a-price-symbol'})
            whole = price_element.find("span", attrs={"class":'a-price-whole'})
            fraction = price_element.find("span", attrs={"class":'a-price-fraction'})
            if symbol and whole and fraction:
                price = f"{symbol.text.strip()}{whole.text.strip().replace(',', '')}.{fraction.text.strip()}"
                return price
        offscreen_price = soup.find("span", attrs={"class":'a-offscreen'})
        if offscreen_price:
            return offscreen_price.text.strip()
        deal_price = soup.find("span", attrs={'id':'priceblock_dealprice'})
        if deal_price:
            return deal_price.text.strip()
        our_price = soup.find("span", attrs={'id':'priceblock_ourprice'})
        if our_price:
            return our_price.text.strip()
    except Exception as e:
        print(f"Price extraction error: {str(e)}")
    return "not available"

def get_discount(soup):
    try:
        discount_element = soup.find("span", attrs={"class":"a-size-large a-color-price savingPriceOverride"})
        if discount_element:
            return discount_element.text.strip()
        savings_element = soup.find("span", attrs={"class":"a-size-base a-color-secondary"})
        if savings_element and "%" in savings_element.text:
            return savings_element.text.strip()
        strike_price = soup.find("span", attrs={"class":"a-text-strike"})
        current_price_element = soup.find("span", attrs={"class":"a-offscreen"})
        if strike_price and current_price_element:
            try:
                strike_price_value = clean_price(strike_price.text.strip())
                current_price_value = clean_price(current_price_element.text.strip())
                if strike_price_value > 0 and current_price_value > 0:
                    discount_percentage = ((strike_price_value - current_price_value) / strike_price_value) * 100
                    return f"{discount_percentage:.1f}%"
            except:
                pass
    except Exception as e:
        print(f"Discount extraction error: {str(e)}")
    return "No discount"

def get_rating(soup):
    try:
        rating = soup.find("i", attrs={'class':'a-icon a-icon-star a-star-4-5'}).string.strip()
    except AttributeError:
        try:
            rating = soup.find("span", attrs={'class':'a-icon-alt'}).string.strip()
        except:
            rating = ""
    return rating

def get_review_count(soup):
    try:
        review_count = soup.find("span", attrs={'id':'acrCustomerReviewText'}).string.strip()
    except AttributeError:
        review_count = ""
    return review_count

def get_availability(soup):
    try:
        available = soup.find("div", attrs={'id':'availability'})
        available = available.find("span").string.strip()
    except AttributeError:
        available = "Not Available"
    return available

def get_specs(soup):
    specs = {}
    try:
        details_section = soup.find("div", attrs={"id": "productDetails_techSpec_section_1"})
        if details_section:
            rows = details_section.find_all("tr")
            for row in rows:
                key = row.find("th").text.strip()
                value = row.find("td").text.strip()
                specs[key] = value
        if not specs:
            details_section = soup.find("div", attrs={"id": "detailBullets_feature_div"})
            if details_section:
                list_items = details_section.find_all("li")
                for item in list_items:
                    text = item.text.strip()
                    if ":" in text:
                        key_value = text.split(":", 1)
                        specs[key_value[0].strip()] = key_value[1].strip()
    except Exception as e:
        print(f"Spec extraction error: {str(e)}")
    return specs

def get_display_size(soup, title):
    try:
        specs = get_specs(soup)
        for key in specs:
            if "display" in key.lower() or "screen" in key.lower() or "monitor" in key.lower():
                if "inch" in specs[key].lower() or '"' in specs[key]:
                    return specs[key]
        display_pattern = r'(\d+\.?\d?)[\s-]?inch|(\d+\.?\d?)[\s-]?"'
        match = re.search(display_pattern, title, re.IGNORECASE)
        if match:
            size = match.group(1) if match.group(1) else match.group(2)
            return f"{size} inch"
    except Exception as e:
        print(f"Display size extraction error: {str(e)}")
    return "Not specified"

def get_brand(title):
    try:
        brands = [
            "HP", "Lenovo", "Acer", "ASUS", "Dell", "Apple", "jumper", "Microsoft",
            "MSI", "Samsung", "IST Computers", "AOC", "ACEMAGIC", "LG", "ApoloSign",
            "Alienware", "CHUWI", "GIGABYTE", "Machenike", "Gateway", "Razer",
            "Oemgenuine", "Hewlett Packard", "EXCaliberPC", "mCover", "Bmax",
            "Intel", "Panasonic", "Toshiba", "Toughbook", "Spigen"
        ]
        for brand in brands:
            pattern = r'\b' + re.escape(brand) + r'\b'
            if re.search(pattern, title, re.IGNORECASE):
                return brand
    except Exception as e:
        print(f"Brand extraction error: {str(e)}")
    return "Unknown"

def clean_price(price_str):
    try:
        if not isinstance(price_str, str):
            return np.nan
        price_str = re.sub(r'[^\d.]', '', price_str)
        if '..' in price_str:
            price_str = price_str.replace('..', '.')
        return float(price_str)
    except:
        return np.nan

def clean_rating(rating_str):
    try:
        if isinstance(rating_str, str):
            match = re.search(r'([\d.]+)', rating_str)
            if match:
                return float(match.group(1))
        return np.nan
    except:
        return np.nan

def clean_review_count(review_str):
    try:
        review_count = re.sub(r'[^\d]', '', review_str)
        return float(review_count)
    except:
        return np.nan

def categorize_by_price(price):
    if pd.isna(price):
        return "Unknown"
    elif price < 200:
        return "Budget"
    elif price < 500:
        return "Mid-range"
    elif price < 800:
        return "High-end"
    else:
        return "Premium"

def analyze_laptop_features(titles):
    ram_pattern = r'(\d+)\s*GB RAM'
    storage_pattern = r'(\d+)\s*(?:GB|TB) (?:SSD|HDD|Storage)'
    processor_patterns = [
        r'Intel (?:Core )?i(\d)',
        r'AMD Ryzen (\d)',
        r'Intel Celeron',
        r'Intel Pentium'
    ]
    features = {'ram': [], 'storage': [], 'processor_type': []}
    for title in titles:
        if not isinstance(title, str):
            features['ram'].append(np.nan)
            features['storage'].append(np.nan)
            features['processor_type'].append(np.nan)
            continue
        ram_match = re.search(ram_pattern, title, re.IGNORECASE)
        features['ram'].append(int(ram_match.group(1)) if ram_match else np.nan)
        storage_match = re.search(storage_pattern, title, re.IGNORECASE)
        features['storage'].append(int(storage_match.group(1)) if storage_match else np.nan)
        processor_found = False
        for pattern in processor_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                processor_found = True
                if 'Intel Core' in title or 'i3' in title or 'i5' in title or 'i7' in title or 'i9' in title:
                    features['processor_type'].append('Intel Core')
                elif 'AMD Ryzen' in title:
                    features['processor_type'].append('AMD Ryzen')
                elif 'Intel Celeron' in title:
                    features['processor_type'].append('Intel Celeron')
                elif 'Intel Pentium' in title:
                    features['processor_type'].append('Intel Pentium')
                break
        if not processor_found:
            features['processor_type'].append(np.nan)
    return pd.DataFrame(features)

def get_ram(soup, title):
    try:
        specs = get_specs(soup)
        for key in specs:
            if "ram" in key.lower() or "memory" in key.lower():
                ram_spec = specs[key]
                ram_match = re.search(r'(\d+)\s*GB', ram_spec, re.IGNORECASE)
                if ram_match:
                    return int(ram_match.group(1))
        ram_pattern = r'(\d+)\s*GB RAM'
        match = re.search(ram_pattern, title, re.IGNORECASE)
        if match:
            return int(match.group(1))
    except Exception as e:
        print(f"RAM extraction error: {str(e)}")
    return np.nan

def get_storage(soup, title):
    try:
        specs = get_specs(soup)
        for key in specs:
            if "storage" in key.lower() or "ssd" in key.lower() or "hdd" in key.lower() or "drive" in key.lower():
                storage_spec = specs[key]
                storage_match = re.search(r'(\d+)\s*(GB|TB)', storage_spec, re.IGNORECASE)
                if storage_match:
                    size = int(storage_match.group(1))
                    unit = storage_match.group(2).upper()
                    return f"{size} {unit}"
        storage_pattern = r'(\d+)\s*(GB|TB)\s*(SSD|HDD|Storage|ROM)'
        match = re.search(storage_pattern, title, re.IGNORECASE)
        if match:
            size = match.group(1)
            unit = match.group(2).upper()
            return f"{size} {unit}"
    except Exception as e:
        print(f"Storage extraction error: {str(e)}")
    return "Not specified"

def random_delay():
    delay = random.uniform(1, 5)
    time.sleep(delay)

def scrape_amazon_products(urls):
    all_data = {
        "title": [], "brand": [], "price": [], "rating": [], "reviews": [],
        "availability": [], "ram": [], "storage": [], "display_size": [], "discount": [], "url": []
    }
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.84 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 12_3_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.3 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.79 Safari/537.36'
    ]
    for page_num, URL in enumerate(urls, 1):
        print(f"Processing page {page_num} of {len(urls)}")
        headers = {'User-Agent': random.choice(user_agents), 'Accept-Language': 'en-US, en;q=0.5'}
        try:
            response = requests.get(URL, headers=headers)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page_num}. Status code: {response.status_code}")
                continue
            soup = BeautifulSoup(response.content, "html.parser")
            links = soup.find_all("a", attrs={'class':'a-link-normal s-no-outline'})
            links_list = [link.get('href') for link in links]
            for link in links_list:
                try:
                    random_delay()
                    product_url = "https://www.amazon.com" + link
                    product_headers = {'User-Agent': random.choice(user_agents), 'Accept-Language': 'en-US, en;q=0.5'}
                    product_response = requests.get(product_url, headers=product_headers)
                    if product_response.status_code != 200:
                        print(f"Failed to retrieve product. Status code: {product_response.status_code}")
                        continue
                    product_soup = BeautifulSoup(product_response.content, "html.parser")
                    title = get_title(product_soup)
                    all_data['title'].append(title)
                    all_data['brand'].append(get_brand(title))
                    all_data['price'].append(get_price(product_soup))
                    all_data['rating'].append(get_rating(product_soup))
                    all_data['reviews'].append(get_review_count(product_soup))
                    all_data['availability'].append(get_availability(product_soup))
                    all_data['ram'].append(get_ram(product_soup, title))
                    all_data['storage'].append(get_storage(product_soup, title))
                    all_data['display_size'].append(get_display_size(product_soup, title))
                    all_data['discount'].append(get_discount(product_soup))
                    all_data['url'].append(product_url)
                except Exception as e:
                    print(f"Error processing product: {str(e)}")
            if page_num < len(urls):
                time.sleep(random.uniform(5, 10))
        except Exception as e:
            print(f"Error on page {page_num}: {str(e)}")
    amazon_df = pd.DataFrame.from_dict(all_data)
    amazon_df['title'].replace('', np.nan, inplace=True)
    amazon_df = amazon_df.dropna(subset=['title'])
    return amazon_df

def process_and_analyze_data(df):
    processed_df = df.copy()
    processed_df['price_clean'] = processed_df['price'].apply(clean_price)
    processed_df['rating_clean'] = processed_df['rating'].apply(clean_rating)
    processed_df['reviews_clean'] = processed_df['reviews'].apply(clean_review_count)
    processed_df['price_category'] = processed_df['price_clean'].apply(categorize_by_price)
    processed_df = processed_df.drop_duplicates(subset=['title'])
    return processed_df

# Hàm vẽ biểu đồ và phân tích (tích hợp từ đoạn code trước)
def visualize_and_analyze(df):
    # ======= TIỀN XỬ LÝ THÊM (TỪ ĐOẠN CODE TRƯỚC) ======= #
    # Xử lý khoảng giá
    df['price_bin'] = pd.cut(df['price_clean'], bins=[0, 300, 600, 900, 1200, 2000])

    # ======= VẼ BIỂU ĐỒ ======= #

    # Biểu đồ 1: Trung bình đánh giá top 5 nhãn hàng
    plt.figure(figsize=(8, 5))
    df.groupby('brand')['rating_clean'].mean().sort_values(ascending=False).head(5).plot(kind='bar', color='skyblue')
    plt.title("Top 5 Brands by Average Rating")
    plt.ylabel("Average Rating")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "top5_brand_rating.png"))
    plt.close()

    # Biểu đồ 2: Đánh giá theo mức giá
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='price_bin', y='rating_clean', data=df)
    plt.title("Rating by Price Range")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "rating_by_price.png"))
    plt.close()

    # Biểu đồ 3: Lượng đánh giá theo mức giá
    plt.figure(figsize=(8, 5))
    df.groupby('price_bin')['reviews_clean'].sum().plot(kind='bar', color='lightgreen')
    plt.title("Total Reviews by Price Range")
    plt.ylabel("Total Reviews")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "reviews_by_price.png"))
    plt.close()

    # Biểu đồ 4: RAM theo sao đánh giá
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='ram', y='rating_clean', data=df)
    plt.title("Rating by RAM")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "rating_by_ram.png"))
    plt.close()

    # Biểu đồ 5: ROM theo sao đánh giá
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='storage', y='rating_clean', data=df)
    plt.title("Rating by Storage")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "rating_by_storage.png"))
    plt.close()

    # Biểu đồ 6: Màn hình theo sao đánh giá
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='display_size', y='rating_clean', data=df)
    plt.title("Rating by Display Size")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, "rating_by_display.png"))
    plt.close()

    # ======= PHÂN TÍCH VÀ TẠO BÁO CÁO (TỪ ĐOẠN CODE TRƯỚC) ======= #

    total_reviews = int(df['reviews_clean'].sum())
    avg_rating = df['rating_clean'].mean()
    reviews_per_brand = df.groupby("brand")["reviews_clean"].sum().sort_values(ascending=False)
    avg_price_per_brand = df.groupby("brand")["price_clean"].mean().sort_values(ascending=False)
    avg_rating_per_brand = df.groupby("brand")["rating_clean"].mean().sort_values(ascending=False)

    best_brand = avg_rating_per_brand.idxmax()
    best_price = df.loc[df['rating_clean'].idxmax(), 'price_clean']
    best_ram = df.groupby("ram")["rating_clean"].mean().idxmax()
    best_rom = df.groupby("storage")["rating_clean"].mean().idxmax()
    best_display = df.groupby("display_size")["rating_clean"].mean().idxmax()

    best_config = df.loc[df['rating_clean'].idxmax()][['title', 'brand', 'ram', 'storage', 'display_size', 'price_clean', 'rating_clean']]

    # Tạo file Markdown
    with open(os.path.join(RESULTS_PATH, "amazon_laptop_report.md"), "w", encoding="utf-8") as f:
        f.write(f"""\
# Amazon Laptop Market Analysis Report

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Tổng quan

- **Tổng số lượt đánh giá**: {total_reviews:,}
- **Trung bình sao đánh giá**: {avg_rating:.2f}

## Lượng đánh giá theo nhãn hàng (Top 5)
{reviews_per_brand.head().to_markdown()}

## Giá trung bình theo nhãn hàng (Top 5)
{avg_price_per_brand.head().to_markdown(floatfmt=".2f")}

## Hãng có trung bình đánh giá cao nhất
{avg_rating_per_brand.head(1).to_markdown(floatfmt=".2f")}

## Phân tích theo mức giá và cấu hình

- 💰 **Mức giá có đánh giá cao nhất**: ${best_price:.2f}
- 🔋 **RAM có đánh giá cao nhất**: {best_ram} GB
- 💽 **ROM có đánh giá cao nhất**: {best_rom}
- 🖥️ **Màn hình có đánh giá cao nhất**: {best_display} inches

## Cấu hình tốt nhất theo đánh giá cao nhất:
| Thuộc tính       | Giá trị |
|------------------|---------|
| Nhãn hàng        | {best_config['brand']} |
| RAM              | {best_ram} GB |
| ROM              | {best_rom} |
| Màn hình         | {best_display} inches |
| Giá              | ${best_config['price_clean']:.2f} |
| Đánh giá         | {best_config['rating_clean']} sao |

## Biểu đồ phân tích

![Top 5 Brand Ratings](top5_brand_rating.png)
![Rating by Price](rating_by_price.png)
![Reviews by Price](reviews_by_price.png)
![Rating by RAM](rating_by_ram.png)
![Rating by Storage](rating_by_storage.png)
![Rating by Display](rating_by_display.png)
        """)

def visualize_top5_brands(df):
    if 'brand' in df.columns and 'rating_clean' in df.columns:
        top5_brands = df.groupby('brand')['rating_clean'].mean().nlargest(5)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top5_brands.index, y=top5_brands.values, palette="Blues_r")
        plt.xlabel("Brand")
        plt.ylabel("Average Rating")
        plt.title("Top 5 Brands by Average Rating")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'top5_brands.png'))
        plt.close()
    else:
        print("Required columns 'brand' or 'rating_clean' not found in DataFrame.")

def visualize_rating_by_price(df):
    if 'price_clean' in df.columns and 'rating_clean' in df.columns:
        df = df.dropna(subset=['price_clean', 'rating_clean'])
        price_bins = pd.qcut(df['price_clean'], q=4, labels=["Budget", "Mid-range", "High-end", "Premium"])
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=price_bins, y=df['rating_clean'], palette="coolwarm")
        plt.xlabel("Price Category")
        plt.ylabel("Rating")
        plt.title("Rating Distribution by Price Category")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'rating_by_price.png'))
        plt.close()
    else:
        print("Required columns 'price_clean' or 'rating_clean' not found in DataFrame.")

def visualize_review_count_by_price(df):
    if 'price_clean' in df.columns and 'reviews_clean' in df.columns:
        df = df.dropna(subset=['price_clean', 'reviews_clean'])
        price_bins = pd.qcut(df['price_clean'], q=4, labels=["Budget", "Mid-range", "High-end", "Premium"])
        review_counts = df.groupby(price_bins)['reviews_clean'].sum()
        plt.figure(figsize=(10, 6))
        sns.barplot(x=review_counts.index, y=review_counts.values, palette="viridis")
        plt.xlabel("Price Category")
        plt.ylabel("Total Reviews")
        plt.title("Review Count by Price Category")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_PATH, 'review_count_by_price.png'))
        plt.close()
    else:
        print("Required columns 'price_clean' or 'reviews_clean' not found in DataFrame.")

def generate_analysis_report(df):
    price_stats = df['price_clean'].describe()
    rating_stats = df['rating_clean'].describe()
    price_category_dist = df['price_category'].value_counts(normalize=True) * 100
    rating_by_category = df.groupby('price_category')['rating_clean'].mean().sort_values(ascending=False)
    brand_dist = df['brand'].value_counts().head(5)
    ram_dist = df['ram'].value_counts().sort_index()
    display_sizes = [float(re.search(r'(\d+\.?\d?)', size).group(1)) for size in df['display_size'] if isinstance(size, str) and re.search(r'(\d+\.?\d?)', size)]
    most_common_display = pd.Series(display_sizes).mode()[0] if display_sizes else np.nan
    ssd_count = sum(df['storage'].str.contains('SSD', na=False, case=False) if isinstance(df['storage'], pd.Series) else 0)
    hdd_count = sum(df['storage'].str.contains('HDD', na=False, case=False) if isinstance(df['storage'], pd.Series) else 0)
    most_common_storage = "SSD" if ssd_count > hdd_count else "HDD"
    price_rating_corr = df['price_clean'].corr(df['rating_clean'])
    report = f"""
# Amazon Laptop Market Analysis Report

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## Executive Summary

This report analyzes {len(df)} laptops listed on Amazon. The data was collected through web scraping and provides insights into pricing, customer ratings, brands, and product features.

## Price Analysis

- **Average Price**: ${price_stats['mean']:.2f}
- **Minimum Price**: ${price_stats['min']:.2f}
- **Maximum Price**: ${price_stats['max']:.2f}
- **Median Price**: ${price_stats['50%']:.2f}

## Price Category Distribution

{price_category_dist.to_string()}

## Brand Analysis

**Top 5 Brands**:
{brand_dist.to_string()}

## Rating Analysis

- **Average Rating**: {rating_stats['mean']:.2f} out of 5
- **Minimum Rating**: {rating_stats['min']:.2f}
- **Maximum Rating**: {rating_stats['max']:.2f}

## Rating by Price Category (Descending Order)

{rating_by_category.to_string()}

## Product Features

**RAM Distribution**:
{ram_dist.to_string()}

**Display Size**:
- Most common display size: {most_common_display} inches

**Storage Type**:
- Most common storage type: {most_common_storage}

## Correlations

- **Price-Rating Correlation**: {price_rating_corr:.3f}

## Conclusion

This analysis provides valuable insights into the Amazon laptop market. The data shows the relationship between price, ratings, brands, and product features, which can help consumers make informed purchasing decisions and assist retailers in understanding market trends.
"""
    report_path = os.path.join(RESULTS_PATH, 'amazon_laptop_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    return report

def main():
    # Step 0: Tạo thư mục trên HDFS với quyền
    create_hdfs_directory_with_permissions(f"{HDFS_BASE_PATH}/data", "755")  # Thư mục lưu dữ liệu
    create_hdfs_directory_with_permissions(HDFS_RESULTS_PATH, "755")         # Thư mục lưu kết quả

    # Step 1: Define URLs for scraping
    urls = [
        "https://www.amazon.com/s?k=laptop&page=1",
        "https://www.amazon.com/s?k=laptop&page=2",
        "https://www.amazon.com/s?k=laptop&page=3", 
        "https://www.amazon.com/s?k=laptop&page=4",
        "https://www.amazon.com/s?k=laptop&page=5",
        "https://www.amazon.com/s?k=laptop&page=6",
        "https://www.amazon.com/s?k=laptop&page=7",
        "https://www.amazon.com/s?k=laptop&page=8",
        "https://www.amazon.com/s?k=laptop&page=9",
        "https://www.amazon.com/s?k=laptop&page=10",
        "https://www.amazon.com/s?k=laptop&page=11",
        "https://www.amazon.com/s?k=laptop&page=12",
        "https://www.amazon.com/s?k=laptop&page=13",
        "https://www.amazon.com/s?k=laptop&page=14",
        "https://www.amazon.com/s?k=laptop&page=15",
        "https://www.amazon.com/s?k=laptop&page=16",
        "https://www.amazon.com/s?k=laptop&page=17",
        "https://www.amazon.com/s?k=laptop&page=18",
        "https://www.amazon.com/s?k=laptop&page=19",
        "https://www.amazon.com/s?k=laptop&page=20"
        ]

    # Step 2: Scrape data
    print("Starting data scraping for 1 page...")
    try:
        amazon_df = scrape_amazon_products(urls)
        print(f"Scraped data for {len(amazon_df)} products.")
        raw_data_path = os.path.expanduser(f"{BASE_PATH}/amazon_raw_data.csv")
        amazon_df.to_csv(raw_data_path, header=True, index=False)
        upload_to_hdfs(raw_data_path, f"{HDFS_BASE_PATH}/data/amazon_raw_data.csv")
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        return

    # Step 3: Process and analyze data
    print("Processing and analyzing data...")
    try:
        processed_df = process_and_analyze_data(amazon_df)
        processed_df.to_csv(PROCESSED_PATH, header=True, index=False)
        upload_to_hdfs(PROCESSED_PATH, f"{HDFS_BASE_PATH}/data/amazon_processed_data.csv")
        print("Data processing complete.")
    except Exception as e:
        print(f"Error during data processing: {str(e)}")
        return

    # Step 4: Visualize and analyze (tích hợp cả hai phần)
    print("Generating visualizations and analysis...")
    try:
        # Gọi hàm visualize_and_analyze (tích hợp từ đoạn code trước)
        visualize_and_analyze(processed_df)

        # Gọi các hàm visualize cũ
        visualize_top5_brands(processed_df)
        visualize_rating_by_price(processed_df)
        visualize_review_count_by_price(processed_df)

        # Đẩy các biểu đồ lên HDFS
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'top5_brands.png'), f"{HDFS_RESULTS_PATH}/top5_brands.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'rating_by_price.png'), f"{HDFS_RESULTS_PATH}/rating_by_price.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'review_count_by_price.png'), f"{HDFS_RESULTS_PATH}/review_count_by_price.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'top5_brand_rating.png'), f"{HDFS_RESULTS_PATH}/top5_brand_rating.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'rating_by_price.png'), f"{HDFS_RESULTS_PATH}/rating_by_price.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'reviews_by_price.png'), f"{HDFS_RESULTS_PATH}/reviews_by_price.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'rating_by_ram.png'), f"{HDFS_RESULTS_PATH}/rating_by_ram.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'rating_by_storage.png'), f"{HDFS_RESULTS_PATH}/rating_by_storage.png")
        upload_to_hdfs(os.path.join(RESULTS_PATH, 'rating_by_display.png'), f"{HDFS_RESULTS_PATH}/rating_by_display.png")
        print("Visualizations generated successfully.")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")

    # Step 5: Generate reports
    print("Generating analysis reports...")
    try:
        # Báo cáo từ đoạn code trước
        report_path = os.path.join(RESULTS_PATH, 'amazon_laptop_report.md')
        upload_to_hdfs(report_path, f"{HDFS_RESULTS_PATH}/amazon_laptop_report.md")

        # Báo cáo từ đoạn code mới
        report = generate_analysis_report(processed_df)
        report_path = os.path.join(RESULTS_PATH, 'amazon_laptop_analysis_report.md')
        upload_to_hdfs(report_path, f"{HDFS_RESULTS_PATH}/amazon_laptop_analysis_report.md")
        print("Reports generated successfully.")
    except Exception as e:
        print(f"Error during report generation: {str(e)}")

if __name__ == "__main__":
    main()