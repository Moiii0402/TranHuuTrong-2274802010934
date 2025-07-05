# ============================================================
# PHẦN 1: THIẾT LẬP VÀ IMPORT THƯ VIỆN
# ============================================================
import logging
import pandas as pd
import numpy as np
import os
import subprocess
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import re
import traceback

# Kiểm tra và import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    print("Lỗi: Thư viện 'xgboost' chưa được cài đặt. Vui lòng cài đặt bằng lệnh: pip3 install xgboost")
    exit(1)

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# PHẦN 2: CẤU HÌNH ĐƯỜNG DẪN
# ============================================================
# Đường dẫn cục bộ
BASE_PATH = "~/iDragonCloud/ProjectBigData"
INPUT_PATH = os.path.expanduser(f"{BASE_PATH}/Amazon-Products-Cleaned.csv")
PROCESSED_PATH = os.path.expanduser(f"{BASE_PATH}/processed_data.csv")
RESULTS_PATH = os.path.expanduser(f"{BASE_PATH}/results")

# Đường dẫn HDFS
HDFS_BASE_PATH = "hdfs://localhost:9000"
HDFS_INPUT_PATH = f"{HDFS_BASE_PATH}/data/Amazon-Products-Cleaned.csv"
HDFS_RESULTS_PATH = f"{HDFS_BASE_PATH}/results"

# Tạo thư mục kết quả nếu chưa tồn tại
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# ============================================================
# PHẦN 3: HÀM TƯƠNG TÁC VỚI HDFS
# ============================================================
def download_from_hdfs(hdfs_path, local_path):
    """Tải file từ HDFS về máy cục bộ"""
    logger.info(f"Tải file từ HDFS {hdfs_path} về {local_path}...")
    try:
        subprocess.run(["hdfs", "dfs", "-get", "-f", hdfs_path, local_path], check=True)
        logger.info(f"Tải file từ HDFS thành công: {local_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi tải file từ HDFS: {e}")
        exit(1)

def upload_to_hdfs(local_path, hdfs_path):
    """Đẩy file từ máy cục bộ lên HDFS"""
    logger.info(f"Đẩy file từ {local_path} lên HDFS tại {hdfs_path}...")
    try:
        # Tạo thư mục trên HDFS nếu chưa tồn tại
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", HDFS_RESULTS_PATH], check=True)
        # Đẩy file lên HDFS (ghi đè nếu đã tồn tại)
        subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_path], check=True)
        logger.info(f"Đẩy file lên HDFS thành công: {hdfs_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi đẩy file lên HDFS: {e}")
        exit(1)

# ============================================================
# PHẦN 4: ĐỌC VÀ XỬ LÝ DỮ LIỆU BAN ĐẦU
# ============================================================
def load_and_process_data():
    """Đọc và xử lý dữ liệu ban đầu"""
    # Tải file từ HDFS
    download_from_hdfs(HDFS_INPUT_PATH, INPUT_PATH)
    
    # Đọc dữ liệu
    logger.info(f"Đọc dữ liệu từ {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        logger.error(f"Lỗi khi đọc dữ liệu: {e}")
        exit(1)
    
    # Xử lý dữ liệu
    logger.info("Xử lý dữ liệu...")
    try:
        # Chuyển đổi kiểu dữ liệu
        df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
        df['no_of_ratings'] = pd.to_numeric(df['no_of_ratings'], errors='coerce')
        df['discount_price'] = pd.to_numeric(df['discount_price'], errors='coerce')
        df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

        # Điền giá trị thiếu
        df['ratings'] = df['ratings'].fillna(df['ratings'].mean())
        df['no_of_ratings'] = df['no_of_ratings'].fillna(df['no_of_ratings'].median())
        df['discount_price'] = df['discount_price'].fillna(df['actual_price'].mean())
        df['actual_price'] = df['actual_price'].fillna(df['actual_price'].mean())
        df['sub_category'] = df['sub_category'].fillna("Unknown")
        df['name'] = df['name'].fillna("Unknown")

        # Tạo trường dẫn xuất
        df['quantity_sold'] = (df['no_of_ratings'] * df['ratings'] / 5).astype(int)
        df['revenue'] = df['discount_price'] * df['quantity_sold']
        
        # Lưu dữ liệu đã xử lý
        df.to_csv(PROCESSED_PATH, index=False, encoding='utf-8-sig')
        # Đẩy lên HDFS
        upload_to_hdfs(PROCESSED_PATH, f"{HDFS_BASE_PATH}/data/processed_data.csv")
        
        return df
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {e}")
        exit(1)

# ============================================================
# PHẦN 5: PHÂN TÍCH CƠ BẢN
# ============================================================
def basic_analysis(df):
    """Thực hiện các phân tích cơ bản về doanh thu và sản phẩm"""
    try:
        # 1. Tổng doanh thu của tất cả cửa hàng
        logger.info("Tính tổng doanh thu...")
        total_revenue = df['revenue'].sum()
        logger.info(f"Tổng doanh thu: {total_revenue:,.2f}")
        
        # Lưu kết quả
        total_revenue_df = pd.DataFrame({"Tổng doanh thu": [total_revenue]})
        total_revenue_df.to_csv(f"{RESULTS_PATH}/total_revenue.csv", index=False, encoding='utf-8-sig')
        upload_to_hdfs(f"{RESULTS_PATH}/total_revenue.csv", f"{HDFS_RESULTS_PATH}/total_revenue.csv")
        
        # 2. Top 5 sản phẩm có doanh thu cao nhất
        logger.info("Tìm top 5 sản phẩm có doanh thu cao nhất...")
        top_revenue_products = df[['name', 'main_category', 'revenue']].sort_values(by='revenue', ascending=False).head(5)
        logger.info("Top 5 sản phẩm có doanh thu cao nhất:")
        print(top_revenue_products)
        
        # Lưu kết quả
        top_revenue_products.to_csv(f"{RESULTS_PATH}/top_revenue_products.csv", index=False, encoding='utf-8-sig')
        upload_to_hdfs(f"{RESULTS_PATH}/top_revenue_products.csv", f"{HDFS_RESULTS_PATH}/top_revenue_products.csv")
        
        # 3. Số lượng sản phẩm trung bình bán ra trên mỗi cửa hàng
        logger.info("Tính số lượng sản phẩm trung bình bán ra trên mỗi cửa hàng...")
        avg_quantity_sold_by_store = df.groupby('main_category')['quantity_sold'].mean().reset_index()
        avg_quantity_sold_by_store.columns = ['Cửa hàng', 'Số lượng trung bình bán ra']
        logger.info("Số lượng sản phẩm trung bình bán ra trên mỗi cửa hàng:")
        print(avg_quantity_sold_by_store)
        
        # Lưu kết quả
        avg_quantity_sold_by_store.to_csv(f"{RESULTS_PATH}/avg_quantity_sold_by_store.csv", index=False, encoding='utf-8-sig')
        upload_to_hdfs(f"{RESULTS_PATH}/avg_quantity_sold_by_store.csv", f"{HDFS_RESULTS_PATH}/avg_quantity_sold_by_store.csv")
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện phân tích cơ bản: {e}")
        exit(1)

# ============================================================
# PHẦN 6: PHÂN TÍCH DOANH THU THEO CỬA HÀNG
# ============================================================
def analyze_store_revenue(df):
    """Phân tích doanh thu chi tiết theo cửa hàng"""
    logger.info("Phân tích doanh thu theo cửa hàng (main_category)...")
    try:
        # Tính doanh thu theo main_category (cửa hàng)
        revenue_by_store = df.groupby('main_category')['revenue'].sum().reset_index()
        revenue_by_store = revenue_by_store.sort_values(by='revenue', ascending=False)
        
        # Sản phẩm bán chạy theo cửa hàng
        top_products = df.groupby(['main_category', 'name'])['quantity_sold'].sum().reset_index()
        top_products = top_products.sort_values(by=['main_category', 'quantity_sold'], ascending=[True, False])
        
        # Lưu kết quả
        revenue_by_store.to_csv(f"{RESULTS_PATH}/revenue_by_store.csv", index=False, encoding='utf-8-sig')
        top_products.to_csv(f"{RESULTS_PATH}/top_products_by_store.csv", index=False, encoding='utf-8-sig')
        
        # Đẩy lên HDFS
        upload_to_hdfs(f"{RESULTS_PATH}/revenue_by_store.csv", f"{HDFS_RESULTS_PATH}/revenue_by_store.csv")
        upload_to_hdfs(f"{RESULTS_PATH}/top_products_by_store.csv", f"{HDFS_RESULTS_PATH}/top_products_by_store.csv")
        
        return revenue_by_store, top_products
    except Exception as e:
        logger.error(f"Lỗi khi phân tích doanh thu: {e}")
        return None, None

# ============================================================
# PHẦN 7: DỰ ĐOÁN TỒN KHO (MACHINE LEARNING) - CẢI TIẾN
# ============================================================
def prepare_inventory_data_optimized(df):
    """Chuẩn bị dữ liệu cho dự đoán tồn kho (tối ưu cho dữ liệu lớn)"""
    logger.info("Chuẩn bị dữ liệu cho dự đoán tồn kho (phiên bản tối ưu)...")
    
    try:
        # 1. Xử lý đặc trưng số học - chỉ sử dụng các đặc trưng quan trọng
        # Đặc biệt quan trọng cho sản phẩm điều hòa
        features_numeric = df[['ratings', 'no_of_ratings', 'discount_price', 'actual_price']].copy()
        
        # Tạo đặc trưng mới: tỷ lệ giảm giá
        features_numeric['discount_ratio'] = (df['actual_price'] - df['discount_price']) / df['actual_price']
        
        # Thêm đặc trưng về mức độ phổ biến của sản phẩm
        features_numeric['popularity_score'] = df['ratings'] * np.log1p(df['no_of_ratings'])
        
        # 2. Xử lý main_category và sub_category hiệu quả hơn
        # Lọc ra các danh mục phổ biến, còn lại gộp vào "Other"
        top_categories = df['sub_category'].value_counts().head(20).index
        df['sub_category_grouped'] = df['sub_category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        # One-hot encoding với sparse=True để tiết kiệm bộ nhớ
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        category_encoded = encoder.fit_transform(df[['main_category', 'sub_category_grouped']])
        
        # 3. Xử lý đặc trưng text từ tên sản phẩm
        logger.info("Trích xuất đặc trưng từ tên sản phẩm...")
        
        # Trích xuất thông tin từ tên sản phẩm điều hòa
        def extract_ac_features(name):
            features = {}
            
            # Trích xuất công suất (Ton)
            ton_match = re.search(r'(\d+\.?\d*)\s*Ton', name, re.IGNORECASE)
            features['capacity_ton'] = float(ton_match.group(1)) if ton_match else 0
            
            # Trích xuất star rating
            star_match = re.search(r'(\d+)\s*Star', name, re.IGNORECASE)
            features['star_rating'] = int(star_match.group(1)) if star_match else 0
            
            # Kiểm tra xem có phải là Inverter không
            features['is_inverter'] = 1 if 'inverter' in name.lower() else 0
            
            # Kiểm tra xem có phải là Split AC không
            features['is_split'] = 1 if 'split' in name.lower() else 0
            
            # Kiểm tra chất liệu đồng
            features['is_copper'] = 1 if 'copper' in name.lower() else 0
            
            # Kiểm tra tính năng chuyển đổi
            features['is_convertible'] = 1 if 'convertible' in name.lower() else 0
            
            return features
        
        # Áp dụng hàm trích xuất đặc trưng
        # Chỉ áp dụng cho các sản phẩm điều hòa để tối ưu hiệu suất
        ac_mask = df['sub_category'].str.contains('Conditioner', case=False, na=False)
        
        # Khởi tạo DataFrame cho đặc trưng điều hòa
        ac_features = pd.DataFrame({
            'capacity_ton': 0,
            'star_rating': 0,
            'is_inverter': 0,
            'is_split': 0,
            'is_copper': 0,
            'is_convertible': 0
        }, index=df.index)
        
        # Chỉ cập nhật đặc trưng cho các sản phẩm điều hòa
        for idx, row in df[ac_mask].iterrows():
            features = extract_ac_features(row['name'])
            for key, value in features.items():
                ac_features.loc[idx, key] = value
        
        # Kết hợp đặc trưng số học và đặc trưng điều hòa
        feature_df = pd.concat([features_numeric, ac_features], axis=1)
        
        # 4. Chuẩn hóa dữ liệu số học
        logger.info("Chuẩn hóa dữ liệu...")
        numeric_columns = feature_df.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        feature_df[numeric_columns] = scaler.fit_transform(feature_df[numeric_columns])
        
        # 5. Kết hợp tất cả đặc trưng
        # Chuyển sparse matrix thành DataFrame
        category_df = pd.DataFrame.sparse.from_spmatrix(
            category_encoded,
            index=df.index,
            columns=[f"cat_{i}" for i in range(category_encoded.shape[1])]
        )
        
        # Kết hợp tất cả đặc trưng
        final_features = pd.concat([feature_df, category_df], axis=1)
        
        # 6. Tạo biến mục tiêu dựa trên lịch sử bán hàng
        # Điều chỉnh hệ số dự trữ tồn kho cho từng danh mục
        inventory_factor_map = {
            'Air Conditioners': 1.8,  # Điều hòa cần tồn kho nhiều hơn do tính mùa vụ
            'Refrigerators': 1.5,
            'Washing Machines': 1.6,
            'default': 1.5
        }
        
        # Áp dụng hệ số dự trữ phù hợp cho từng danh mục
        df['inventory_factor'] = df['sub_category'].map(
            lambda x: inventory_factor_map.get(x, inventory_factor_map['default'])
        )
        
        # Tính lượng tồn kho cần thiết
        df['quantity'] = (df['quantity_sold'] * df['inventory_factor']).astype(int)
        
        return final_features, df['quantity']
    
    except Exception as e:
        logger.error(f"Lỗi khi chuẩn bị dữ liệu dự đoán tồn kho: {e}")
        logger.error(traceback.format_exc())
        return None, None

def predict_inventory_optimized(df):
    """Dự đoán nhu cầu tồn kho sử dụng mô hình ML (tối ưu cho dữ liệu lớn)"""
    import time
    import re
    import traceback
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    
    logger.info("Bắt đầu dự đoán tồn kho với phương pháp tối ưu cho dữ liệu lớn...")
    
    try:
        # 1. Chuẩn bị dữ liệu
        start_time = time.time()
        X, y = prepare_inventory_data_optimized(df)
        
        if X is None or y is None:
            logger.error("Chuẩn bị dữ liệu thất bại, không thể tiếp tục dự đoán")
            return df
        
        prep_time = time.time() - start_time
        logger.info(f"Thời gian chuẩn bị dữ liệu: {prep_time:.2f} giây")
        
        # 2. Chia dữ liệu train/test
        logger.info("Chia dữ liệu train/test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Huấn luyện các mô hình
        # a. Gradient Boosting (nhanh hơn XGBoost cho dữ liệu lớn)
        logger.info("Huấn luyện mô hình Gradient Boosting...")
        start_time = time.time()
        
        # Xây dựng mô hình với số lượng cây hợp lý cho dữ liệu lớn
        gb = GradientBoostingRegressor(
            n_estimators=150, 
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            subsample=0.8,  # Sử dụng 80% dữ liệu cho mỗi cây để tăng tốc
            verbose=1  # Hiển thị tiến trình
        )
        
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_mse = mean_squared_error(y_test, gb_pred)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        
        training_time = time.time() - start_time
        logger.info(f"Thời gian huấn luyện Gradient Boosting: {training_time:.2f} giây")
        logger.info(f"Gradient Boosting - MSE: {gb_mse:.2f}, MAE: {gb_mae:.2f}")
        
        # b. XGBoost nếu cần độ chính xác cao hơn
        logger.info("Huấn luyện mô hình XGBoost với cấu hình tối ưu...")
        start_time = time.time()
        
        # Sử dụng XGBoost với cấu hình phù hợp cho dữ liệu lớn
        xgb = XGBRegressor(
            n_estimators=200, 
            learning_rate=0.1,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',  # Phương pháp 'hist' nhanh hơn cho dữ liệu lớn
            random_state=42,
            n_jobs=-1  # Sử dụng tất cả CPU cores
        )
        
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_mae = mean_absolute_error(y_test, xgb_pred)
        
        training_time = time.time() - start_time
        logger.info(f"Thời gian huấn luyện XGBoost: {training_time:.2f} giây")
        logger.info(f"XGBoost - MSE: {xgb_mse:.2f}, MAE: {xgb_mae:.2f}")
        
        # 4. So sánh và chọn mô hình tốt nhất
        logger.info("So sánh và chọn mô hình tốt nhất...")
        if xgb_mse < gb_mse:
            logger.info("XGBoost có hiệu suất tốt hơn, sử dụng XGBoost để dự đoán")
            best_model = xgb
            best_model_name = "XGBoost"
        else:
            logger.info("Gradient Boosting có hiệu suất tốt hơn, sử dụng Gradient Boosting để dự đoán")
            best_model = gb
            best_model_name = "GradientBoosting"
        
        # 5. Phân tích đặc trưng quan trọng
        logger.info("Phân tích đặc trưng quan trọng...")
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_names = X.columns
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            logger.info(f"Top 15 đặc trưng quan trọng nhất ({best_model_name}):")
            print(feature_importance)
            
            # Lưu thông tin đặc trưng quan trọng
            feature_importance.to_csv(f"{RESULTS_PATH}/feature_importance.csv", index=False, encoding='utf-8-sig')
            upload_to_hdfs(f"{RESULTS_PATH}/feature_importance.csv", f"{HDFS_RESULTS_PATH}/feature_importance.csv")
        
        # 6. Dự đoán trên toàn bộ dữ liệu với mô hình tốt nhất
        logger.info("Dự đoán trên toàn bộ dữ liệu...")
        start_time = time.time()
        
        # Dự đoán từng batch để tiết kiệm bộ nhớ với dữ liệu lớn
        batch_size = 10000
        num_batches = int(np.ceil(X.shape[0] / batch_size))
        
        predictions = np.zeros(X.shape[0])
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X.shape[0])
            
            X_batch = X.iloc[start_idx:end_idx]
            predictions[start_idx:end_idx] = best_model.predict(X_batch)
            
            if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                logger.info(f"Đã dự đoán {end_idx}/{X.shape[0]} mẫu ({(end_idx/X.shape[0])*100:.1f}%)")
        
        # Làm tròn kết quả dự đoán
        df['predicted_inventory'] = np.round(predictions).astype(int)
        
        predict_time = time.time() - start_time
        logger.info(f"Thời gian dự đoán: {predict_time:.2f} giây")
        
        # 7. Thực hiện phân tích chi tiết cho các sản phẩm điều hòa
        logger.info("Phân tích chi tiết cho các sản phẩm điều hòa...")
        ac_df = df[df['sub_category'].str.contains('Conditioner', case=False, na=False)].copy()
        
        # Phân loại theo công suất và hãng sản xuất
        def extract_brand(name):
            common_brands = ['LG', 'Lloyd', 'Samsung', 'Hitachi', 'Daikin', 'Voltas', 'Carrier', 'Panasonic']
            for brand in common_brands:
                if brand.lower() in name.lower():
                    return brand
            return 'Other'
        
        ac_df['brand'] = ac_df['name'].apply(extract_brand)
        
        # Tính lượng tồn kho dự báo theo nhãn hiệu
        brand_inventory = ac_df.groupby('brand').agg({
            'quantity_sold': 'sum',
            'predicted_inventory': 'sum',
            'revenue': 'sum'
        }).reset_index()
        
        brand_inventory['inventory_ratio'] = brand_inventory['predicted_inventory'] / brand_inventory['quantity_sold']
        brand_inventory = brand_inventory.sort_values('revenue', ascending=False)
        
        logger.info("Dự báo tồn kho theo nhãn hiệu điều hòa:")
        print(brand_inventory)
        
        # 8. Lưu kết quả phân tích
        brand_inventory.to_csv(f"{RESULTS_PATH}/ac_brand_inventory.csv", index=False, encoding='utf-8-sig')
        upload_to_hdfs(f"{RESULTS_PATH}/ac_brand_inventory.csv", f"{HDFS_RESULTS_PATH}/ac_brand_inventory.csv")
        
        # 9. Lưu kết quả dự đoán tồn kho
        df[['name', 'main_category', 'sub_category', 'quantity_sold', 'predicted_inventory', 'revenue']].to_csv(
            f"{RESULTS_PATH}/inventory_forecast_optimized.csv", index=False, encoding='utf-8-sig'
        )
        upload_to_hdfs(
            f"{RESULTS_PATH}/inventory_forecast_optimized.csv", 
            f"{HDFS_RESULTS_PATH}/inventory_forecast_optimized.csv"
        )
        
        return df
    
    except Exception as e:
        logger.error(f"Lỗi khi dự đoán tồn kho: {e}")
        logger.error(traceback.format_exc())
        return df

# ============================================================
# PHẦN 8: HỆ THỐNG ĐỀ XUẤT SẢN PHẨM
# ============================================================
def recommend_products_for_stores(df):
    """Đề xuất sản phẩm cho từng cửa hàng dựa trên độ tương đồng"""
    
    # 1. Chuẩn bị ma trận đặc trưng
    def prepare_feature_matrix(df, weights=None):
        """Chuẩn bị dữ liệu cho hệ thống gợi ý"""
        logger.info("Chuẩn bị dữ liệu cho hệ thống gợi ý...")
        try:
            # Đặc trưng số
            features = ['ratings', 'no_of_ratings', 'discount_price', 'actual_price']
            feature_matrix = df[features].fillna(0)
            
            # Mã hóa sub_category
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            sub_category_encoded = encoder.fit_transform(df[['sub_category']])
            sub_category_df = pd.DataFrame(sub_category_encoded, columns=encoder.get_feature_names_out(['sub_category']))
            
            # Trích xuất đặc trưng từ name bằng TF-IDF
            tfidf = TfidfVectorizer(max_features=50, stop_words='english')
            name_features = tfidf.fit_transform(df['name']).toarray()
            name_df = pd.DataFrame(name_features, columns=[f"name_tfidf_{i}" for i in range(name_features.shape[1])])
            
            # Kết hợp tất cả đặc trưng
            feature_matrix = pd.concat([feature_matrix, sub_category_df, name_df], axis=1)
            
            # Lưu lại danh sách cột trước khi chuẩn hóa
            feature_columns = feature_matrix.columns
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
            feature_matrix = pd.DataFrame(feature_matrix, columns=feature_columns)
            
            # Áp dụng trọng số nếu có
            if weights is not None:
                for feature in features:
                    if feature in feature_matrix.columns:
                        feature_matrix[feature] = feature_matrix[feature] * weights.get(feature, 1.0)
            return feature_matrix
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu gợi ý: {e}")
            return None
    
    # 2. Thiết lập trọng số cho các đặc trưng
    weights = {
        'ratings': 3.0,
        'no_of_ratings': 2.0,
        'discount_price': 0.3,
        'actual_price': 0.3
    }
    
    # 3. Hàm đề xuất sản phẩm cho một cửa hàng cụ thể
    def recommend_products(store, df, n=3):
        """Gợi ý sản phẩm cho một cửa hàng cụ thể"""
        logger.info(f"Gợi ý sản phẩm cho cửa hàng {store}...")
        try:
            # Lấy danh sách sản phẩm trong cửa hàng
            store_products = df[df['main_category'] == store]
            store_indices = store_products.index.tolist()
            
            # Nếu cửa hàng không có sản phẩm, trả về rỗng
            if not store_indices:
                return []
            
            # Xác định các danh mục liên quan
            related_categories = {
                'appliances': ['tv, audio & cameras'],
                'car & motorbike': ['car & motorbike'],
                'tv, audio & cameras': ['appliances']
            }
            allowed_categories = related_categories.get(store, [store])
            
            # Lấy sản phẩm từ danh mục liên quan
            other_products = df[(df['main_category'] != store) & (df['main_category'].isin(allowed_categories))]
            other_indices = other_products.index.tolist()
            
            # Nếu không có sản phẩm liên quan, trả về rỗng
            if not other_indices:
                return []
            
            # Chuẩn bị ma trận đặc trưng
            feature_matrix = prepare_feature_matrix(df, weights=weights)
            if feature_matrix is None:
                return []
            
            # Tính độ tương đồng
            store_features = feature_matrix.loc[store_indices]
            other_features = feature_matrix.loc[other_indices]
            similarity_matrix = cosine_similarity(store_features, other_features)
            
            # Tính điểm trung bình và sắp xếp
            avg_similarity_scores = np.mean(similarity_matrix, axis=0)
            similarity_scores = [(idx, score) for idx, score in zip(other_indices, avg_similarity_scores)]
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Lấy top n sản phẩm
            top_n = similarity_scores[:n]
            recommended_products = [(df.loc[idx, 'name'], df.loc[idx, 'main_category'], score) for idx, score in top_n]
            return recommended_products
        except Exception as e:
            logger.error(f"Lỗi khi gợi ý sản phẩm cho cửa hàng {store}: {e}")
            return []
    
    # 4. Thực hiện đề xuất cho tất cả cửa hàng
    try:
        # Lấy danh sách cửa hàng
        stores = df['main_category'].unique()
        
        # Tạo danh sách lưu kết quả
        recommendation_table = []
        
        # Gợi ý cho từng cửa hàng
        for store in stores[:3]:  # Chỉ lấy 3 cửa hàng đầu để demo
            recommendations = recommend_products(store, df)
            if recommendations:
                for product_name, product_category, score in recommendations:
                    similarity_percent = score * 100
                    recommendation_table.append([store, product_name, product_category, similarity_percent])
            else:
                recommendation_table.append([store, "Không có", "-", 0.0])
        
        # Tạo DataFrame kết quả
        recommendation_df = pd.DataFrame(recommendation_table, 
                                         columns=["Cửa hàng", "Sản phẩm đề xuất", "Danh mục sản phẩm", "Điểm tương đồng"])
        
        # Tạo DataFrame hiển thị
        recommendation_display = recommendation_df.copy()
        recommendation_display['Điểm tương đồng'] = recommendation_display['Điểm tương đồng'].apply(
            lambda x: f"{x:.2f}%" if x > 0 else "-"
        )
        recommendation_display['Điểm tương đồng'] = recommendation_display['Điểm tương đồng'].apply(
            lambda x: f"{x} *" if x != "-" and float(x.strip("%")) > 90 else x
        )
        
        # Lưu kết quả
        recommendation_df.to_csv(f"{RESULTS_PATH}/recommendations.csv", index=False, encoding='utf-8-sig')
        upload_to_hdfs(f"{RESULTS_PATH}/recommendations.csv", f"{HDFS_RESULTS_PATH}/recommendations.csv")
        
        # Hiển thị kết quả
        logger.info("Kết quả đề xuất sản phẩm:")
        print(recommendation_display)
        
        return recommendation_df
    except Exception as e:
        logger.error(f"Lỗi khi đề xuất sản phẩm: {e}")
        exit(1)

# ============================================================
# PHẦN 9: HÀM MAIN - ĐIỀU KHIỂN LUỒNG CHẠY CHƯƠNG TRÌNH
# ============================================================
def main():
    """Hàm chính điều khiển toàn bộ luồng chạy của chương trình"""
    logger.info("===== BẮT ĐẦU CHƯƠNG TRÌNH PHÂN TÍCH DỮ LIỆU AMAZON PRODUCTS =====")
    
    # 1. Đọc và xử lý dữ liệu
    df = load_and_process_data()
    
    # 2. Phân tích cơ bản
    basic_analysis(df)
    
    # 3. Phân tích doanh thu theo cửa hàng
    revenue_by_store, top_products = analyze_store_revenue(df)
    
    # 4. Dự đoán tồn kho (sử dụng phiên bản tối ưu)
    df = predict_inventory_optimized(df)
    
    # 5. Hệ thống đề xuất sản phẩm
    recommend_products_for_stores(df)
    
    logger.info("===== KẾT THÚC CHƯƠNG TRÌNH PHÂN TÍCH DỮ LIỆU AMAZON PRODUCTS =====")

# Chạy chương trình
if __name__ == "__main__":
    main()

