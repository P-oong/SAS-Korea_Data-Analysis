import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import TargetEncoder
import sys
import time
from tqdm import tqdm

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
    
    # 기본 경로 설정 (ver4 데이터 사용)
    if train_path is None:
        train_path = os.path.join(PROJECT_ROOT, 'data/TRAIN_DATA_ver2.csv')
    if test_path is None:
        test_path = os.path.join(PROJECT_ROOT, 'data/TEST_DATA_ver2.csv')
    
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
    
    # 타겟값이 0인 행 제거
    if target_col in train_df.columns:
        zero_target_rows = train_df[train_df[target_col] == 0]
        logger.info(f"타겟값이 0인 행 개수: {len(zero_target_rows)}")
        train_df = train_df[train_df[target_col] > 0]
        logger.info(f"타겟값 0 제거 후 train_df 크기: {train_df.shape}")
    
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
    
    # tqdm으로 로그 변환 진행상황 표시
    for col in tqdm(log_transformed_features, desc="로그 변환 적용중"):
        if col in train_df.columns:
            has_negative = (train_df[col] < 0).any()
            if has_negative:
                min_val = train_df[col].min()
                if min_val < 0:
                    train_df[col] = np.log1p(train_df[col] - min_val + 1)
            else:
                train_df[col] = np.log1p(train_df[col])
            
            # 테스트 데이터도 같은 방식으로 변환
            if col in test_df.columns:
                has_negative = (test_df[col] < 0).any()
                if has_negative:
                    min_val = test_df[col].min()
                    if min_val < 0:
                        test_df[col] = np.log1p(test_df[col] - min_val + 1)
                else:
                    test_df[col] = np.log1p(test_df[col])
                
            logger.info(f"{col} 변수에 로그 변환 적용")
    
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

    # tqdm으로 스케일링 진행상황 표시
    for col in tqdm(numerical_features_final, desc="스케일링 적용중"):
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

def train_model_with_gridsearch(X_train, y_train, cv=5, n_jobs=-1):
    """RandomForest 모델을 그리드서치로 최적화하여 학습 (tqdm 적용)"""
    logger.info("RandomForest 모델 그리드서치 최적화 시작")
    
    # 그리드서치를 위한 파라미터 그리드 설정 (오버피팅 방지를 위한 파라미터 포함)
    custom_param_grid = {
        'n_estimators': [180, 200, 220],
        'max_depth': [18, 20, 22],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 6],
        'max_features': [0.7, 0.9],
        'bootstrap': [True]
    }
    
    # 파라미터 조합 생성
    import itertools
    param_names = list(custom_param_grid.keys())
    param_values = list(itertools.product(*[custom_param_grid[param] for param in param_names]))
    total_combinations = len(param_values)
    
    logger.info(f"총 {total_combinations}개의 파라미터 조합을 평가합니다.")
    
    # 교차 검증을 위한 KFold 설정
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 모든 파라미터 조합 결과 저장 변수
    all_params = []
    mean_train_scores = []
    mean_test_scores = []
    std_train_scores = []
    std_test_scores = []
    
    # 파라미터 조합별 그리드서치
    for i, values in enumerate(tqdm(param_values, desc="파라미터 조합 평가중")):
        param_dict = {param_names[j]: values[j] for j in range(len(param_names))}
        
        # 현재 파라미터 조합에 대한 모델 생성
        try:
            model = RandomForestRegressor(**param_dict, random_state=42, n_jobs=n_jobs)
            
            # 교차 검증 점수 저장 변수
            train_scores = []
            test_scores = []
            
            # K-fold 교차 검증
            fold_indices = list(kf.split(X_train))
            for fold_idx, (train_idx, val_idx) in enumerate(tqdm(fold_indices, total=cv, desc=f"교차 검증 {i+1}/{total_combinations}", leave=False)):
                # NumPy 배열로 변환 후 인덱싱
                X_fold_train = X_train.values[train_idx]
                X_fold_val = X_train.values[val_idx]
                y_fold_train = y_train.values[train_idx]
                y_fold_val = y_train.values[val_idx]
                
                # 모델 학습
                model.fit(X_fold_train, y_fold_train)
                
                # 훈련셋 예측 및 평가
                y_train_pred = model.predict(X_fold_train)
                y_train_true_orig = np.expm1(y_fold_train)
                y_train_pred_orig = np.expm1(y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_train_true_orig, y_train_pred_orig))
                train_scores.append(-train_rmse)  # 음수로 저장 (높을수록 좋게)
                
                # 검증셋 예측 및 평가
                y_val_pred = model.predict(X_fold_val)
                y_val_true_orig = np.expm1(y_fold_val)
                y_val_pred_orig = np.expm1(y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val_true_orig, y_val_pred_orig))
                test_scores.append(-val_rmse)  # 음수로 저장 (높을수록 좋게)
            
            # 평균 및 표준편차 계산
            mean_train_score = np.mean(train_scores)
            mean_test_score = np.mean(test_scores)
            std_train_score = np.std(train_scores)
            std_test_score = np.std(test_scores)
            
            # 결과 저장
            all_params.append(param_dict)
            mean_train_scores.append(mean_train_score)
            mean_test_scores.append(mean_test_score)
            std_train_scores.append(std_train_score)
            std_test_scores.append(std_test_score)
            
            logger.info(f"파라미터 {i+1}/{total_combinations}: 훈련 RMSE = {-mean_train_score:.4f} (±{std_train_score:.4f}), 검증 RMSE = {-mean_test_score:.4f} (±{std_test_score:.4f})")
            logger.info(f"  파라미터: {param_dict}")
        
        except Exception as e:
            logger.error(f"파라미터 조합 {param_dict} 평가 중 오류 발생: {str(e)}")
            continue
    
    if not all_params:
        logger.error("모든 파라미터 조합에서 오류가 발생했습니다.")
        # 기본 모델 반환
        default_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=20, 
            min_samples_split=5, 
            min_samples_leaf=4, 
            max_features=0.7, 
            bootstrap=True, 
            random_state=42,
            n_jobs=n_jobs
        )
        default_model.fit(X_train, y_train)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': default_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return default_model, feature_importance, default_model.get_params()
    
    # 결과를 GridSearchCV와 유사한 형식으로 저장
    cv_results = {
        'params': all_params,
        'mean_train_score': np.array(mean_train_scores),
        'mean_test_score': np.array(mean_test_scores),
        'std_train_score': np.array(std_train_scores),
        'std_test_score': np.array(std_test_scores)
    }
    
    # 가장 좋은 파라미터 조합 찾기 (검증 점수 기준)
    best_idx = np.argmax(mean_test_scores)
    best_params = all_params[best_idx]
    best_score = mean_test_scores[best_idx]
    
    logger.info(f"최적 파라미터 (검증 RMSE 기준): {best_params}")
    logger.info(f"최적 검증 RMSE: {-best_score:.4f}")
    
    # 최적 모델 재학습 (전체 데이터)
    best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=n_jobs)
    best_model.fit(X_train, y_train)
    
    # 훈련-검증 점수 차이와 검증 점수를 모두 고려한 균형 있는 모델 선택
    # 점수는 음수로 저장되어 있으므로 양수로 변환 (낮을수록 좋음)
    train_scores = -np.array(mean_train_scores)
    test_scores = -np.array(mean_test_scores)
    
    # 훈련-검증 점수 차이 계산
    score_ratios = train_scores / test_scores
    
    # 균형 점수 계산: 검증 RMSE와 훈련-검증 점수 차이의 조합
    # 검증 RMSE는 낮을수록 좋고, 훈련-검증 점수 차이도 낮을수록 좋음
    # 여기서는 검증 RMSE에 더 많은 가중치를 둠 (0.7 vs 0.3)
    balance_scores = 0.7 * test_scores + 0.3 * score_ratios
    
    # 균형 점수가 가장 낮은 상위 5개 모델 확인
    top_indices = np.argsort(balance_scores)[:min(5, len(balance_scores))]
    
    logger.info("\n균형 잡힌 상위 5개 파라미터 조합 (검증 RMSE와 훈련-검증 점수 차이 고려):")
    for i, idx in enumerate(top_indices):
        params = all_params[idx]
        test_rmse = test_scores[idx]
        train_rmse = train_scores[idx]
        ratio = score_ratios[idx]
        balance = balance_scores[idx]
        
        logger.info(f"{i+1}. 검증 RMSE: {test_rmse:.4f}, 훈련 RMSE: {train_rmse:.4f}, 비율: {ratio:.4f}, 균형 점수: {balance:.4f}")
        logger.info(f"   파라미터: {params}")
        
        # 첫 번째(가장 균형 잡힌) 모델을 선택
        if i == 0:
            balanced_best_params = params
            
    # 균형 잡힌 최적 모델로 재학습
    logger.info(f"\n균형 잡힌 최적 파라미터: {balanced_best_params}")
    balanced_model = RandomForestRegressor(**balanced_best_params, random_state=42, n_jobs=n_jobs)
    balanced_model.fit(X_train, y_train)
    
    # 균형 잡힌 모델과 원래 모델의 성능 비교
    y_pred_orig = best_model.predict(X_train)
    y_pred_balanced = balanced_model.predict(X_train)
    
    rmse_orig = np.sqrt(mean_squared_error(y_train, y_pred_orig))
    rmse_balanced = np.sqrt(mean_squared_error(y_train, y_pred_balanced))
    
    logger.info(f"\n원래 최적 모델 (검증 RMSE 기준) 훈련 RMSE: {rmse_orig:.4f}")
    logger.info(f"균형 잡힌 최적 모델 훈련 RMSE: {rmse_balanced:.4f}")
    
    # 교차 검증으로 두 모델 비교
    orig_cv_scores = []
    balanced_cv_scores = []
    
    logger.info("\n두 모델의 5-fold 교차 검증 비교:")
    
    fold_indices = list(kf.split(X_train))
    for fold, (train_idx, val_idx) in enumerate(tqdm(fold_indices, total=cv, desc="최종 모델 비교 중"), 1):
        # NumPy 배열로 변환 후 인덱싱
        X_fold_train = X_train.values[train_idx]
        X_fold_val = X_train.values[val_idx]
        y_fold_train = y_train.values[train_idx]
        y_fold_val = y_train.values[val_idx]
        
        # 원래 모델
        orig_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=n_jobs)
        orig_model.fit(X_fold_train, y_fold_train)
        y_orig_pred = orig_model.predict(X_fold_val)
        
        # 균형 잡힌 모델
        bal_model = RandomForestRegressor(**balanced_best_params, random_state=42, n_jobs=n_jobs)
        bal_model.fit(X_fold_train, y_fold_train)
        y_bal_pred = bal_model.predict(X_fold_val)
        
        # 원래 스케일로 변환하여 RMSE 계산
        y_val_orig = np.expm1(y_fold_val)
        y_orig_pred_orig = np.expm1(y_orig_pred)
        y_bal_pred_orig = np.expm1(y_bal_pred)
        
        orig_rmse = np.sqrt(mean_squared_error(y_val_orig, y_orig_pred_orig))
        bal_rmse = np.sqrt(mean_squared_error(y_val_orig, y_bal_pred_orig))
        
        orig_cv_scores.append(orig_rmse)
        balanced_cv_scores.append(bal_rmse)
        
        logger.info(f"Fold {fold} - 원래 모델 RMSE: {orig_rmse:.4f}, 균형 모델 RMSE: {bal_rmse:.4f}")
    
    logger.info(f"\n교차 검증 평균 - 원래 모델: {np.mean(orig_cv_scores):.4f} ± {np.std(orig_cv_scores):.4f}")
    logger.info(f"교차 검증 평균 - 균형 모델: {np.mean(balanced_cv_scores):.4f} ± {np.std(balanced_cv_scores):.4f}")
    
    # 두 모델 중 더 나은 모델 선택
    if np.mean(balanced_cv_scores) <= np.mean(orig_cv_scores) * 1.03:  # 균형 모델이 3% 이내로 나쁘다면 균형 모델 선택
        logger.info("\n최종 선택: 균형 잡힌 모델 (훈련-검증 균형 고려)")
        final_model = balanced_model
        final_params = balanced_best_params
    else:
        logger.info("\n최종 선택: 원래 최적 모델 (검증 RMSE가 훨씬 더 좋음)")
        final_model = best_model
        final_params = best_params
    
    # 특성 중요도 계산
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"상위 25개 중요 특성:\n{feature_importance.head(25)}")
    
    # 최적 모델 반환
    return final_model, feature_importance, final_params

def evaluate_model(model, X_train, y_train, X_test=None, y_test=None):
    """모델 성능 평가"""
    logger.info("모델 성능 평가")
    
    # 훈련 데이터 성능 평가
    y_pred_train = model.predict(X_train)
    rmse_train_log = np.sqrt(mean_squared_error(y_train, y_pred_train))
    y_train_inv = np.expm1(y_train)
    y_pred_train_inv = np.expm1(y_pred_train)
    rmse_train_orig = np.sqrt(mean_squared_error(y_train_inv, y_pred_train_inv))
    r2_train = r2_score(y_train, y_pred_train)
    
    logger.info(f"훈련 데이터 RMSE (로그 스케일): {rmse_train_log:.4f}")
    logger.info(f"훈련 데이터 RMSE (원래 스케일): {rmse_train_orig:.4f}")
    logger.info(f"훈련 데이터 R² 점수: {r2_train:.4f}")
    
    # 테스트 데이터가 있는 경우 테스트 성능도 평가
    if X_test is not None and y_test is not None:
        y_pred_test = model.predict(X_test)
        rmse_test_log = np.sqrt(mean_squared_error(y_test, y_pred_test))
        y_test_inv = np.expm1(y_test)
        y_pred_test_inv = np.expm1(y_pred_test)
        rmse_test_orig = np.sqrt(mean_squared_error(y_test_inv, y_pred_test_inv))
        r2_test = r2_score(y_test, y_pred_test)
        
        logger.info(f"테스트 데이터 RMSE (로그 스케일): {rmse_test_log:.4f}")
        logger.info(f"테스트 데이터 RMSE (원래 스케일): {rmse_test_orig:.4f}")
        logger.info(f"테스트 데이터 R² 점수: {r2_test:.4f}")
        
        # 오버피팅 정도 평가
        rmse_diff = rmse_train_log - rmse_test_log
        rmse_ratio = rmse_train_log / rmse_test_log if rmse_test_log > 0 else float('inf')
        
        logger.info(f"훈련-테스트 RMSE 차이 (로그): {rmse_diff:.4f}")
        logger.info(f"훈련-테스트 RMSE 비율 (로그): {rmse_ratio:.4f}")
        
        if rmse_ratio < 0.7 or rmse_ratio > 1.3:
            logger.warning("주의: 훈련-테스트 성능 차이가 크며, 모델이 과적합 또는 과소적합 될 수 있습니다.")
        else:
            logger.info("모델이 훈련 및 테스트 데이터에서 균형있게 수행됩니다.")
    
    # K-fold 교차 검증 수행
    logger.info("5-fold 교차 검증 수행 중...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    
    fold_indices = list(kf.split(X_train))
    for train_idx, val_idx in tqdm(fold_indices, total=5, desc="교차 검증 진행중"):
        # NumPy 배열로 변환 후 인덱싱
        X_fold_train = X_train.values[train_idx]
        X_fold_val = X_train.values[val_idx]
        y_fold_train = y_train.values[train_idx]
        y_fold_val = y_train.values[val_idx]
        
        # 모델 학습
        model_fold = RandomForestRegressor(**model.get_params())
        model_fold.fit(X_fold_train, y_fold_train)
        
        # 검증 세트 예측
        y_fold_val_pred = model_fold.predict(X_fold_val)
        
        # 원래 스케일로 변환하여 RMSE 계산
        y_fold_val_orig = np.expm1(y_fold_val)
        y_fold_val_pred_orig = np.expm1(y_fold_val_pred)
        fold_rmse = np.sqrt(mean_squared_error(y_fold_val_orig, y_fold_val_pred_orig))
        cv_scores.append(fold_rmse)
    
    logger.info(f"교차 검증 RMSE (원래 스케일): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    return {
        'rmse_train_log': rmse_train_log,
        'rmse_train_orig': rmse_train_orig,
        'r2_train': r2_train,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores)
    }

def save_model_data(model, preprocess_info, best_params, feature_importance=None, model_name='RandomForest_no_weather'):
    """모델 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(PROJECT_ROOT, f'results/models/{model_name}_{timestamp}.pkl')
    
    model_data = {
        'model': model,
        'scaler': preprocess_info['scaler'],
        'target_encoder': preprocess_info['target_encoder'],
        'feature_names': preprocess_info['feature_names'],
        'best_params': best_params,
        'feature_importance': feature_importance,
        'timestamp': timestamp,
        'model_name': model_name
    }
    
    joblib.dump(model_data, model_path)
    logger.info(f"모델이 저장되었습니다: {model_path}")
    
    return model_path

def save_predictions_data(test_df, predictions, model_name='RandomForest_no_weather'):
    """예측 결과 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(PROJECT_ROOT, f'results/predictions/{model_name}_predictions_{timestamp}.csv')
    
    test_df_with_pred = test_df.copy()
    test_df_with_pred['predicted_TOTAL_ELEC'] = predictions
    test_df_with_pred.to_csv(output_path, index=False, encoding='cp949')
    logger.info(f"예측 결과가 저장되었습니다: {output_path}")
    
    return output_path

def main():
    """메인 함수"""
    logger.info("RandomForest 모델 (날씨 데이터 제외) 작업 시작")
    
    try:
        # 데이터 로드 (날씨 컬럼 제외)
        train_df, test_df = load_data()
        
        # 데이터 전처리 (날씨 컬럼 없이 진행)
        X_train, y_train, X_test, test_df_orig, preprocess_info = preprocess_data(train_df, test_df)
        
        # 그리드서치를 이용한 RandomForest 모델 최적화
        best_model, feature_importance, best_params = train_model_with_gridsearch(
            X_train, y_train, cv=5, n_jobs=-1
        )
        
        # 모델 성능 평가
        evaluation_results = evaluate_model(best_model, X_train, y_train)
        
        # 테스트 데이터 예측
        predictions_log = best_model.predict(X_test)
        predictions = np.expm1(predictions_log)
        
        # 모델 저장
        model_path = save_model_data(
            best_model, preprocess_info, best_params, 
            feature_importance=feature_importance
        )
        
        # 예측 결과 저장
        predictions_path = save_predictions_data(test_df_orig, predictions)
        
        logger.info("모든 작업이 성공적으로 완료되었습니다.")
        logger.info(f"모델 저장 경로: {model_path}")
        logger.info(f"예측 결과 저장 경로: {predictions_path}")
        
    except Exception as e:
        logger.error(f"작업 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 