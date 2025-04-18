import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# 프로젝트 루트 경로 설정 (notebooks/LHP에서 두 단계 상위 디렉토리)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# 결과 디렉토리 생성 (절대 경로 사용)
os.makedirs(os.path.join(PROJECT_ROOT, 'results/models'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'results/predictions'), exist_ok=True)

# 로깅 설정
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(train_path=None, test_path=None):
    """데이터 로드 함수"""
    logger.info("데이터 로드 시작")
    
    # 기본 경로 설정
    if train_path is None:
        train_path = os.path.join(PROJECT_ROOT, 'data/TRAIN_DATA.csv')
    if test_path is None:
        test_path = os.path.join(PROJECT_ROOT, 'data/TEST_DATA.csv')
    
    # cp949 인코딩으로 CSV 파일 읽기 (한글 인코딩 문제 해결)
    train_df = pd.read_csv(train_path, encoding='cp949')
    test_df = pd.read_csv(test_path, encoding='cp949')
    logger.info(f"학습 데이터 크기: {train_df.shape}, 테스트 데이터 크기: {test_df.shape}")
    return train_df, test_df

def create_time_features(train_df, test_df):
    """DATA_YM 변수를 사용하여 시간 관련 특성 생성"""
    logger.info("시간 관련 특성 생성 시작...")
    
    # 데이터프레임 복사
    train_df_result = train_df.copy()
    test_df_result = test_df.copy()
    
    # datetime으로 변환
    train_df_result['date'] = pd.to_datetime(train_df_result['DATA_YM'].astype(str), format="%Y%m")
    test_df_result['date'] = pd.to_datetime(test_df_result['DATA_YM'].astype(str), format="%Y%m")
    
    # 기본 시간 특성 생성 (year, month, quarter)
    for df in [train_df_result, test_df_result]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['year'].astype('category')
    
    # 시작 시점 계산 (train 기준) 및 개월 수(t) 계산
    date_start = train_df_result['date'].min()
    logger.info(f"기준 시작일: {date_start}")
    
    # train/test에 t 특성 추가
    for df in [train_df_result, test_df_result]:
        df['t'] = ((df['date'].dt.year - date_start.year) * 12 +
                    (df['date'].dt.month - date_start.month)).astype(int)
    
    # 월(month) 주기성 변환 - 1차 푸리에 변환
    for df in [train_df_result, test_df_result]:
        df['month_sin1'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos1'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 2~3차 푸리에 변환
    for k in [2, 3]:
        for df in [train_df_result, test_df_result]:
            df[f'sin{k}'] = np.sin(2 * np.pi * k * df['t'] / 12)
            df[f'cos{k}'] = np.cos(2 * np.pi * k * df['t'] / 12)
    
    # 분기(quarter) 주기성 변환
    for df in [train_df_result, test_df_result]:
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # 계절 변수 생성
    season_map = {
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    }
    
    strength_map = {'summer': 3, 'winter': 2, 'spring': 1, 'autumn': 1}
    
    for df in [train_df_result, test_df_result]:
        df['season'] = df['month'].map(season_map).astype('category')
        df['season_strength'] = df['season'].map(strength_map).astype(int)
    
    # DATA_YM 및 date 변수 제거
    train_df_result = train_df_result.drop(columns=['DATA_YM', 'date'])
    test_df_result = test_df_result.drop(columns=['DATA_YM', 'date'])
    
    logger.info("시간 관련 특성 생성 완료")
    return train_df_result, test_df_result

def create_station_feature(df):
    """'역' 또는 '역_숫자'로 끝나는 지역 식별"""
    df_result = df.copy()
    station_pattern = re.compile(r'역(_\d+)?$')
    
    if 'AREA_NM' in df_result.columns:
        df_result['IS_STATION'] = df_result['AREA_NM'].apply(
            lambda x: 1 if isinstance(x, str) and station_pattern.search(x) else 0
        )
        logger.info(f"역 관련 지역 수: {df_result['IS_STATION'].sum()}")
    else:
        df_result['IS_STATION'] = 0
    
    return df_result

def create_eup_feature(df):
    """'읍' 또는 '읍_숫자'로 끝나는 지역 식별"""
    df_result = df.copy()
    eup_pattern = re.compile(r'읍(_\d+)?$')
    
    if 'AREA_NM' in df_result.columns:
        df_result['IS_eup'] = df_result['AREA_NM'].apply(
            lambda x: 1 if isinstance(x, str) and eup_pattern.search(x) else 0
        )
        logger.info(f"읍 관련 지역 수: {df_result['IS_eup'].sum()}")
    else:
        df_result['IS_eup'] = 0
    
    return df_result

def apply_log_transformation(df, columns):
    """지정된 변수들에 로그 변환 적용"""
    df_transformed = df.copy()
    for col in columns:
        if col in df.columns:
            # 음수 값이 있는지 확인
            has_negative = (df[col] < 0).any()
            
            if has_negative:
                # 음수 값이 있는 경우 다른 처리 방법 적용
                min_val = df[col].min()
                if min_val < 0:
                    df_transformed[col] = np.log1p(df[col] - min_val + 1)
            else:
                # 음수 값이 없는 경우 일반적인 log1p 변환
                df_transformed[col] = np.log1p(df[col])
            
            logger.info(f"{col} 변수에 로그 변환 적용")
    
    return df_transformed

def identify_feature_types(df):
    """데이터프레임에서 범주형 변수와 수치형 변수를 식별"""
    # 제외할 변수 목록
    exclude_columns = ['AREA_ID', 'DIST_CD', 'FAC_TRAIN']
    
    # 범주형 변수 식별 (object 타입 또는 카디널리티가 낮은 변수)
    categorical_features = []
    numerical_features = []
    
    for col in df.columns:
        if col in exclude_columns:
            continue
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    logger.info(f"범주형 변수 수: {len(categorical_features)}")
    logger.info(f"수치형 변수 수: {len(numerical_features)}")
    
    return categorical_features, numerical_features

def one_hot_encode(df, categorical_columns, is_train=True, train_dummies=None):
    """범주형 변수에 원핫인코딩 적용"""
    df_encoded = df.copy()
    all_encoded_columns = []
    
    if is_train:
        # 학습 데이터 인코딩
        for col in categorical_columns:
            if col in df.columns:
                # 변수가 'DIST_NM' 또는 'AREA_NM'인 경우 접두사 설정
                prefix = 'DIST' if col == 'DIST_NM' else 'AREA' if col == 'AREA_NM' else col
                dummies = pd.get_dummies(df[col], prefix=prefix)
                
                # 원본 데이터프레임에서 범주형 변수 제거
                df_encoded = df_encoded.drop(columns=[col])
                
                # 원핫인코딩된 변수 추가
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # 인코딩된 컬럼 목록 저장
                all_encoded_columns.extend(dummies.columns.tolist())
        
        return df_encoded, all_encoded_columns
    else:
        # 테스트 데이터 인코딩 (학습 데이터와 같은 더미 변수 유지)
        for col in categorical_columns:
            if col in df.columns:
                prefix = 'DIST' if col == 'DIST_NM' else 'AREA' if col == 'AREA_NM' else col
                dummies = pd.get_dummies(df[col], prefix=prefix)
                
                # 원본 데이터프레임에서 범주형 변수 제거
                df_encoded = df_encoded.drop(columns=[col])
                
                # 원핫인코딩된 변수 추가
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
        
        # 학습 데이터에 있지만 테스트 데이터에 없는 컬럼 추가
        for col in train_dummies:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        # 테스트 데이터에만 있고 학습 데이터에 없는 컬럼 제거
        extra_cols = [col for col in df_encoded.columns if col not in train_dummies]
        if extra_cols:
            df_encoded = df_encoded.drop(columns=extra_cols)
        
        # 컬럼 순서 맞추기
        df_encoded = df_encoded[train_dummies]
        
        return df_encoded

def preprocess_data(train_df, test_df, target_col='TOTAL_ELEC'):
    """데이터 전처리 수행"""
    logger.info("데이터 전처리 시작...")
    
    # 시간 관련 특성 생성
    if 'DATA_YM' in train_df.columns and 'DATA_YM' in test_df.columns:
        train_df, test_df = create_time_features(train_df, test_df)
    
    # 제외할 변수 목록
    exclude_columns = ['AREA_ID', 'DIST_CD', 'FAC_TRAIN']
    
    # IS_STATION 및 IS_eup 변수 생성
    train_df = create_station_feature(train_df)
    test_df = create_station_feature(test_df)
    train_df = create_eup_feature(train_df)
    test_df = create_eup_feature(test_df)
    
    # 결측치 및 중복 처리
    train_df = train_df.dropna()
    train_df = train_df.drop_duplicates(keep='first')
    
    # 로그 변환 적용할 변수
    log_transformed_features = [
        'TOTAL_BIDG', 'FAC_NEIGH_1', 'FAC_NEIGH_2', 
        'FAC_RETAIL', 'FAC_STAY', 'FAC_LEISURE', 
        'TOTAL_GAS', 'CMRC_GAS'
    ]
    
    # 로그 변환 적용
    train_df = apply_log_transformation(train_df, log_transformed_features)
    test_df = apply_log_transformation(test_df, log_transformed_features)
    
    # 타겟 변수 로그 변환
    train_df[target_col] = np.log1p(train_df[target_col])
    
    # 제외할 변수 삭제
    train_df = train_df.drop(columns=exclude_columns, errors='ignore')
    test_df = test_df.drop(columns=exclude_columns, errors='ignore')
    
    # 범주형 변수와 수치형 변수 식별
    categorical_features, numerical_features = identify_feature_types(train_df)
    
    # 원핫인코딩 적용
    train_df_encoded, train_dummies = one_hot_encode(train_df, categorical_features, is_train=True)
    test_df_encoded = one_hot_encode(test_df, categorical_features, is_train=False, train_dummies=train_df_encoded.columns)
    
    # 테스트 데이터에 타겟 컬럼이 있는 경우 제거
    if target_col in test_df_encoded.columns:
        test_df_encoded = test_df_encoded.drop(columns=[target_col])
    
    # 타겟 변수 분리
    y_train = train_df_encoded[target_col]
    X_train = train_df_encoded.drop(columns=[target_col])
    X_test = test_df_encoded.copy()
    
    # 수치형 변수 스케일링 적용
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    for col in numerical_features:
        if col in X_train.columns:
            X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
            X_test_scaled[col] = scaler.transform(X_test[[col]])
    
    logger.info("데이터 전처리 완료")
    logger.info(f"최종 학습 데이터 특성 수: {X_train_scaled.shape[1]}")
    logger.info(f"최종 테스트 데이터 특성 수: {X_test_scaled.shape[1]}")
    
    # 전처리 관련 정보 저장
    preprocess_info = {
        'scaler': scaler,
        'feature_names': X_train.columns.tolist(),
        'log_transformed_features': log_transformed_features
    }
    
    return X_train_scaled, y_train, X_test_scaled, test_df, preprocess_info

def train_model(X_train, y_train):
    """XGBoost 모델 학습"""
    logger.info("XGBoost 모델 학습 시작")
    
    # 사용자가 지정한 파라미터 적용
    params = {
        'n_estimators': 100, 
        'max_depth': 9, 
        'learning_rate': 0.2, 
        'subsample': 0.8, 
        'colsample_bytree': 1.0, 
        'gamma': 0,
        'random_state': 42
    }# {'n_estimators': 100, 'max_depth': 9, 'learning_rate': 0.2, 'subsample': 0.8, 'colsample_bytree': 1.0, 'gamma': 0} 
    
    logger.info(f"사용할 파라미터: {params}")
    
    # 모델 학습
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # 학습 데이터에 대한 예측
    y_pred_train = model.predict(X_train)
    
    # 로그 스케일에서 RMSE 계산
    rmse_train_log = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    # 원래 스케일로 변환 후 RMSE 계산
    y_train_inv = np.expm1(y_train)
    y_pred_train_inv = np.expm1(y_pred_train)
    rmse_train_orig = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
    
    # R² 점수 계산
    r2_train = r2_score(y_train, y_pred_train)
    
    logger.info(f"학습 데이터 RMSE (로그 스케일): {rmse_train_log:.4f}")
    logger.info(f"학습 데이터 RMSE (원래 스케일): {rmse_train_orig:.4f}")
    logger.info(f"학습 데이터 R² 점수: {r2_train:.4f}")
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"상위 10개 중요 특성:\n{feature_importance.head(10)}")
    
    return model, feature_importance

def save_model_data(model, preprocess_info, model_name='XGBoost_optimized'):
    """모델 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(PROJECT_ROOT, f'results/models/{model_name}_{timestamp}.pkl')
    
    model_data = {
        'model': model,
        'scaler': preprocess_info['scaler'],
        'feature_names': preprocess_info['feature_names'],
        'timestamp': timestamp,
        'model_name': model_name
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"모델이 저장되었습니다: {model_path}")
    
    return model_path

def save_predictions_data(test_df, predictions, model_name='XGBoost_optimized'):
    """예측 결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(PROJECT_ROOT, f'results/predictions/{model_name}_predictions_{timestamp}.csv')
    
    # 예측 결과를 테스트 데이터프레임에 추가
    test_df['predicted_TOTAL_ELEC'] = predictions
    
    # 예측 결과 저장
    test_df.to_csv(output_path, index=False, encoding='cp949')
    logger.info(f"예측 결과가 저장되었습니다: {output_path}")
    
    return output_path

def main():
    """메인 함수"""
    logger.info("작업 시작")
    
    try:
        # 데이터 로드
        train_df, test_df = load_data()
        
        # 데이터 전처리
        X_train, y_train, X_test, test_df_orig, preprocess_info = preprocess_data(train_df, test_df)
        
        # 모델 학습
        model, feature_importance = train_model(X_train, y_train)
        
        # 테스트 데이터 예측
        predictions_log = model.predict(X_test)
        
        # 로그 스케일에서 원래 스케일로 변환
        predictions = np.expm1(predictions_log)
        
        # 모델 저장
        model_path = save_model_data(model, preprocess_info)
        
        # 예측 결과 저장
        predictions_path = save_predictions_data(test_df_orig, predictions)
        
        logger.info("모든 작업이 성공적으로 완료되었습니다.")
        logger.info(f"모델 저장 경로: {model_path}")
        logger.info(f"예측 결과 저장 경로: {predictions_path}")
        
    except Exception as e:
        logger.error(f"작업 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 