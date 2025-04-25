import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.preprocessing import StandardScaler
# XGBoost 대신 GradientBoostingRegressor 임포트
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import TargetEncoder
import shap # SHAP 라이브러리 추가

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
    """데이터 로드 함수 (날씨 데이터 컬럼 제외 버전)"""
    logger.info("데이터 로드 시작")
    
    # 기본 경로 설정 (ver2 데이터 사용)
    if train_path is None:
        train_path = os.path.join(PROJECT_ROOT, 'data/TRAIN_DATA_ver4.csv')
    if test_path is None:
        test_path = os.path.join(PROJECT_ROOT, 'data/TEST_DATA_ver4.csv')
    
    # cp949 인코딩으로 CSV 파일 읽기
    train_df = pd.read_csv(train_path, encoding='cp949')
    test_df = pd.read_csv(test_path, encoding='cp949')
    logger.info(f"초기 학습 데이터 크기: {train_df.shape}, 테스트 데이터 크기: {test_df.shape}")
    
    # 날씨 관련 컬럼 제거
    weather_cols_to_drop = ['불쾌지수_등급', '체감온도_등급']
    train_df = train_df.drop(columns=weather_cols_to_drop, errors='ignore')
    test_df = test_df.drop(columns=weather_cols_to_drop, errors='ignore')
    logger.info(f"날씨 관련 컬럼 제거 후 학습 데이터 크기: {train_df.shape}, 테스트 데이터 크기: {test_df.shape}")
            
    return train_df, test_df

def create_time_features(train_df, test_df):
    """DATA_YM 변수를 사용하여 시간 관련 특성 생성"""
    logger.info("시간 관련 특성 생성 시작...")
    
    train_df_result = train_df.copy()
    test_df_result = test_df.copy()
    
    train_df_result['date'] = pd.to_datetime(train_df_result['DATA_YM'].astype(str), format="%Y%m")
    test_df_result['date'] = pd.to_datetime(test_df_result['DATA_YM'].astype(str), format="%Y%m")
    
    for df in [train_df_result, test_df_result]:
        df['month'] = df['date'].dt.month.astype(int)
    
    for df in [train_df_result, test_df_result]:
        df['month_sin1'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos1'] = np.cos(2 * np.pi * df['month'] / 12)
    
    date_start = train_df_result['date'].min()
    logger.info(f"기준 시작일: {date_start}")
    
    for df in [train_df_result, test_df_result]:
        df['t'] = ((df['date'].dt.year - date_start.year) * 12 +
                    (df['date'].dt.month - date_start.month)).astype(int)
    
    for k in [2, 3]:
        for df in [train_df_result, test_df_result]:
            df[f'sin{k}'] = np.sin(2 * np.pi * k * df['t'] / 12)
            df[f'cos{k}'] = np.cos(2 * np.pi * k * df['t'] / 12)
    
    for df in [train_df_result, test_df_result]:
        df['quarter'] = df['date'].dt.quarter
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
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
    
    for df in [train_df_result, test_df_result]:
        df.drop(columns=['DATA_YM', 'date', 'month', 'quarter'], inplace=True)
    
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
            has_negative = (df[col] < 0).any()
            if has_negative:
                min_val = df[col].min()
                if min_val < 0:
                    df_transformed[col] = np.log1p(df[col] - min_val + 1)
            else:
                df_transformed[col] = np.log1p(df[col])
            logger.info(f"{col} 변수에 로그 변환 적용")
    return df_transformed

def identify_feature_types(df):
    """데이터프레임에서 범주형 변수와 수치형 변수를 식별"""
    exclude_columns = ['AREA_ID', 'DIST_CD', 'FAC_TRAIN']
    categorical_features = []
    numerical_features = []
    
    # 중요 범주형 변수들의 데이터 타입 확인 (날씨 제외)
    important_vars = ['season']
    for var in important_vars:
        if var in df.columns:
            logger.info(f"변수 {var}의 dtype: {df[var].dtype}")
    
    for col in df.columns:
        if col in exclude_columns:
            continue
        if df[col].dtype.name == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            categorical_features.append(col)
        else:
            numerical_features.append(col)
            
    logger.info(f"범주형 변수 수: {len(categorical_features)}")
    logger.info(f"수치형 변수 수: {len(numerical_features)}")
    
    return categorical_features, numerical_features

def target_encode(df_train, df_test, categorical_columns, target_col):
    """범주형 변수에 타겟인코딩 적용"""
    logger.info("타겟인코딩 적용 시작...")
    df_train_encoded = df_train.copy()
    df_test_encoded = df_test.copy()
    encoder = TargetEncoder(target_type='continuous')

    if target_col in df_train.columns:
        encoder.fit(df_train[categorical_columns], df_train[target_col])
        encoded_train_np = encoder.transform(df_train[categorical_columns])
        encoded_train_df = pd.DataFrame(
            encoded_train_np, index=df_train.index, columns=categorical_columns
        )
        df_train_encoded = df_train_encoded.drop(columns=categorical_columns)
        df_train_encoded = pd.concat([df_train_encoded, encoded_train_df], axis=1)

        encoded_test_np = encoder.transform(df_test[categorical_columns])
        encoded_test_df = pd.DataFrame(
            encoded_test_np, index=df_test.index, columns=categorical_columns
        )
        df_test_encoded = df_test_encoded.drop(columns=categorical_columns)
        df_test_encoded = pd.concat([df_test_encoded, encoded_test_df], axis=1)

        logger.info(f"타겟인코딩 적용 완료: {len(categorical_columns)}개 변수 처리")
    else:
        logger.warning(f"타겟 변수 {target_col}가 학습 데이터에 없어 타겟인코딩을 적용할 수 없습니다.")

    return df_train_encoded, df_test_encoded, encoder

def preprocess_data(train_df, test_df, target_col='TOTAL_ELEC'):
    """데이터 전처리 수행"""
    logger.info("데이터 전처리 시작...")
    
    excluded_areas = ['수원시청_1', '유천아파트앞']
    if 'AREA_NM' in train_df.columns:
        excluded_rows = train_df[train_df['AREA_NM'].isin(excluded_areas)]
        logger.info(f"제거할 행 개수: {len(excluded_rows)}")
        logger.info(f"제거할 지역 행 세부 정보:\n{excluded_rows['AREA_NM'].value_counts()}")
        train_df = train_df[~train_df['AREA_NM'].isin(excluded_areas)]
        logger.info(f"제거 후 train_df 크기: {train_df.shape}")
    
    if 'DATA_YM' in train_df.columns and 'DATA_YM' in test_df.columns:
        train_df, test_df = create_time_features(train_df, test_df)
    
    exclude_columns = ['AREA_ID', 'DIST_CD', 'FAC_TRAIN']
    
    train_df = create_station_feature(train_df)
    test_df = create_station_feature(test_df)
    train_df = create_eup_feature(train_df)
    test_df = create_eup_feature(test_df)
    
    train_df = train_df.dropna()
    train_df = train_df.drop_duplicates(keep='first')
    
    log_transformed_features = [
        'TOTAL_BIDG', 'FAC_NEIGH_1', 'FAC_NEIGH_2', 
        'FAC_RETAIL', 'FAC_STAY', 'FAC_LEISURE', 
        'TOTAL_GAS', 'CMRC_GAS'
    ]
    
    train_df = apply_log_transformation(train_df, log_transformed_features)
    test_df = apply_log_transformation(test_df, log_transformed_features)
    
    train_df[target_col] = np.log1p(train_df[target_col])
    
    train_df = train_df.drop(columns=exclude_columns, errors='ignore')
    test_df = test_df.drop(columns=exclude_columns, errors='ignore')
    
    categorical_features, numerical_features = identify_feature_types(train_df)
    
    logger.info(f"식별된 범주형 변수: {categorical_features}")
    logger.info(f"식별된 수치형 변수: {numerical_features}")
    
    # 타겟인코딩할 변수 (AREA_NM, DIST_NM, season - 날씨 제외)
    target_encode_features = []
    base_target_features = ['AREA_NM', 'DIST_NM', 'season'] # 날씨 컬럼 제외
    for feature in base_target_features:
        if feature in train_df.columns:
            target_encode_features.append(feature)
        else:
             logger.warning(f"타겟 인코딩 대상 컬럼 '{feature}'이(가) 데이터에 없습니다.")

    logger.info(f"타겟인코딩 적용할 변수: {target_encode_features}")
    
    if target_encode_features:
        train_df_encoded, test_df_encoded, target_encoder = target_encode(
            train_df, test_df, target_encode_features, target_col
        )
    else:
        train_df_encoded = train_df.copy()
        test_df_encoded = test_df.copy()
        target_encoder = None
        logger.info("타겟인코딩할 변수가 없습니다.")
    
    y_train = train_df_encoded[target_col]
    X_train = train_df_encoded.drop(columns=[target_col])
    
    if target_col in test_df_encoded.columns:
        X_test = test_df_encoded.drop(columns=[target_col])
    else:
        X_test = test_df_encoded.copy()
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    numerical_features_final = [col for col in numerical_features if col in X_train.columns]
    logger.info(f"스케일링 적용될 최종 수치형 변수: {numerical_features_final}")

    for col in numerical_features_final:
        if col in X_train_scaled.columns and col in X_test_scaled.columns:
            X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
            X_test_scaled[col] = scaler.transform(X_test[[col]])
        else:
            logger.warning(f"스케일링 중 컬럼 누락: {col}")

    logger.info("데이터 전처리 완료")
    logger.info(f"최종 학습 데이터 특성 수: {X_train_scaled.shape[1]}")
    logger.info(f"최종 테스트 데이터 특성 수: {X_test_scaled.shape[1]}")
    
    preprocess_info = {
        'scaler': scaler,
        'target_encoder': target_encoder,
        'feature_names': X_train.columns.tolist(),
        'log_transformed_features': log_transformed_features,
        'categorical_features': categorical_features,
        'target_encode_features': target_encode_features
    }
    
    return X_train_scaled, y_train, X_test_scaled, test_df, preprocess_info

def train_model(X_train, y_train, categorical_features=None):
    """GBM 모델 학습 및 k-fold 교차 검증""" # 함수 설명 변경
    logger.info("GBM 모델 학습 및 k-fold 교차 검증 시작") # 로그 메시지 변경
    
    # 지정된 하이퍼파라미터 적용
    params = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.2,
        'subsample': 1.0,
        'max_features': 0.7, 
        'min_samples_split': 5,
        'random_state': 42
    }
    
    logger.info(f"사용할 파라미터: {params}")
    
    # 모델 정의 (GradientBoostingRegressor 사용)
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train) # GBM은 eval_metric 파라미터 사용 안 함
    
    # 학습 데이터에 대한 예측
    y_pred_train = model.predict(X_train)
    rmse_train_log = np.sqrt(mean_squared_error(y_train, y_pred_train))
    y_train_inv = np.expm1(y_train)
    y_pred_train_inv = np.expm1(y_pred_train)
    rmse_train_orig = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
    r2_train = r2_score(y_train, y_pred_train)
    
    logger.info(f"학습 데이터 RMSE (로그 스케일): {rmse_train_log:.4f}")
    logger.info(f"학습 데이터 RMSE (원래 스케일): {rmse_train_orig:.4f}")
    logger.info(f"학습 데이터 R² 점수: {r2_train:.4f}")
    
    logger.info("5-fold 교차 검증 시작...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_train_rmse = []
    fold_train_r2 = []
    fold_val_rmse = []
    fold_val_r2 = []
    
    # K-Fold 내 타겟 인코딩 로직은 동일하게 유지
    target_encode_cols_in_train = [col for col in categorical_features if col in X_train.columns]
    if target_encode_cols_in_train:
        logger.info("K-fold 검증에서 각 fold별로 타겟 인코딩 적용")
    
    X_train_df = X_train.copy()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_fold_train, X_fold_val = X_train_df.iloc[train_idx].copy(), X_train_df.iloc[val_idx].copy()
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if target_encode_cols_in_train:
            encoder = TargetEncoder(target_type='continuous')
            target_cols_current_fold = [col for col in target_encode_cols_in_train if col in X_fold_train.columns]

            if target_cols_current_fold:
                encoder.fit(X_fold_train[target_cols_current_fold], y_fold_train)
                
                encoded_train_np = encoder.transform(X_fold_train[target_cols_current_fold])
                encoded_val_np = encoder.transform(X_fold_val[target_cols_current_fold])
                
                encoded_train_df = pd.DataFrame(
                    encoded_train_np, index=X_fold_train.index, columns=target_cols_current_fold
                )
                encoded_val_df = pd.DataFrame(
                    encoded_val_np, index=X_fold_val.index, columns=target_cols_current_fold
                )
                
                X_fold_train = X_fold_train.drop(columns=target_cols_current_fold)
                X_fold_train = pd.concat([X_fold_train, encoded_train_df], axis=1)
                
                X_fold_val = X_fold_val.drop(columns=target_cols_current_fold)
                X_fold_val = pd.concat([X_fold_val, encoded_val_df], axis=1)
        
        # Fold별 모델 학습 (GBM 사용)
        fold_model = GradientBoostingRegressor(**params)
        fold_model.fit(X_fold_train, y_fold_train)
        
        # 예측 및 성능 계산 (동일)
        y_fold_train_pred = fold_model.predict(X_fold_train)
        y_fold_train_orig = np.expm1(y_fold_train)
        y_fold_train_pred_orig = np.expm1(y_fold_train_pred)
        
        y_fold_val_pred = fold_model.predict(X_fold_val)
        y_fold_val_orig = np.expm1(y_fold_val)
        y_fold_val_pred_orig = np.expm1(y_fold_val_pred)
        
        train_rmse = np.sqrt(mean_squared_error(y_fold_train_orig, y_fold_train_pred_orig))
        train_r2 = r2_score(y_fold_train, y_fold_train_pred)
        
        val_rmse = np.sqrt(mean_squared_error(y_fold_val_orig, y_fold_val_pred_orig))
        val_r2 = r2_score(y_fold_val, y_fold_val_pred)
        
        fold_train_rmse.append(train_rmse)
        fold_train_r2.append(train_r2)
        fold_val_rmse.append(val_rmse)
        fold_val_r2.append(val_r2)
        
        logger.info(f"Fold {fold}:")
        logger.info(f"  훈련 데이터 - RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
        logger.info(f"  검증 데이터 - RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    
    mean_train_rmse = np.mean(fold_train_rmse)
    mean_train_r2 = np.mean(fold_train_r2)
    mean_val_rmse = np.mean(fold_val_rmse)
    mean_val_r2 = np.mean(fold_val_r2)
    
    logger.info("5-fold 교차 검증 평균 성능:")
    logger.info(f"  훈련 데이터 - RMSE: {mean_train_rmse:.4f}, R²: {mean_train_r2:.4f}")
    logger.info(f"  검증 데이터 - RMSE: {mean_val_rmse:.4f}, R²: {mean_val_r2:.4f}")
    
    # 특성 중요도 계산 (GBM은 feature_importances_ 속성 제공)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"상위 25개 중요 특성 (기본):\n{feature_importance.head(25)}")
    
    # SHAP 중요도 계산 (SHAP은 scikit-learn 트리 모델 지원)
    logger.info("SHAP 중요도 계산 시작...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': X_train.columns,
        'shap_importance': shap_sum
    }).sort_values('shap_importance', ascending=False)
    
    logger.info(f"상위 25개 중요 특성 (SHAP):\n{shap_importance.head(25)}")
    
    return model, feature_importance

def save_model_data(model, preprocess_info, model_name='GBM_no_weather'): # 모델 이름 변경
    """모델 저장 (GBM, 날씨 제외 버전)"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(PROJECT_ROOT, f'results/models/{model_name}_{timestamp}.pkl')
    
    model_data = {
        'model': model,
        'scaler': preprocess_info['scaler'],
        'target_encoder': preprocess_info['target_encoder'],
        'feature_names': preprocess_info['feature_names'],
        'timestamp': timestamp,
        'model_name': model_name
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"모델이 저장되었습니다: {model_path}")
    
    return model_path

def save_predictions_data(test_df, predictions, model_name='GBM_no_weather'): # 모델 이름 변경
    """예측 결과 저장 (GBM, 날씨 제외 버전)"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(PROJECT_ROOT, f'results/predictions/{model_name}_predictions_{timestamp}.csv')
    
    test_df['predicted_TOTAL_ELEC'] = predictions
    test_df.to_csv(output_path, index=False, encoding='cp949')
    logger.info(f"예측 결과가 저장되었습니다: {output_path}")
    
    return output_path

def main():
    """메인 함수"""
    logger.info("작업 시작")
    
    try:
        # 데이터 로드 (날씨 컬럼 제외)
        train_df, test_df = load_data()
        
        # 데이터 전처리 (날씨 컬럼 없이 진행)
        X_train, y_train, X_test, test_df_orig, preprocess_info = preprocess_data(train_df, test_df)
        
        # 타겟 인코딩 대상 컬럼 목록 (날씨 제외)
        target_encoded_features = preprocess_info.get('target_encode_features', None) 
        
        # 모델 학습 (GBM 모델 사용)
        model, feature_importance = train_model(
            X_train, y_train,
            categorical_features=target_encoded_features
        )
        
        predictions_log = model.predict(X_test)
        predictions = np.expm1(predictions_log)
        
        # 모델 저장 (이름 변경됨)
        model_path = save_model_data(model, preprocess_info)
        
        # 예측 결과 저장 (이름 변경됨)
        predictions_path = save_predictions_data(test_df_orig, predictions)
        
        logger.info("모든 작업이 성공적으로 완료되었습니다.")
        logger.info(f"모델 저장 경로: {model_path}")
        logger.info(f"예측 결과 저장 경로: {predictions_path}")
        
    except Exception as e:
        logger.error(f"작업 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 