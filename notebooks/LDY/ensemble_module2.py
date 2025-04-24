# ensemble_module2.py
# 사용법:
# 1. 아래 ★ 설정 섹션 ★ 의 경로 및 모델 이름을 사용자 환경과 파일에 맞게 수정.
# 2. 사용할 메타 모델 타입을 META_MODEL_TYPE 에서 선택.
# 3. 필요한 모든 예측값/실제값 CSV 파일을 PREDICTION_DIR 폴더에 준비.
#    (로그 스케일, 헤더X, 인덱스X, 파일명 규칙: {모델명}_val_preds.csv 등)
# 4. 터미널에서 python ensemble_module2.py 를 실행.

import pandas as pd
import numpy as np
import os
import sys # 오류 발생 시 종료용

# === 필요한 모델 라이브러리 Import ===
from sklearn.linear_model import RidgeCV, LinearRegression, LassoCV, ElasticNetCV # 메타 모델 후보
from sklearn.ensemble import GradientBoostingRegressor # 메타 모델 후보
import lightgbm as lgb # 메타 모델 후보
from sklearn.metrics import mean_squared_error, r2_score # 성능 확인용

# ======================================================================
#                           ★ 설정 섹션 ★
# ======================================================================
# 이 스크립트 사용자는 이 섹션의 값들을 확인하고 수정해야 합니다.

# 1. 예측 CSV 파일들이 저장된 폴더 경로 (필수 수정)
PREDICTION_DIR = "/Users/idoyeong/계명대학교/sas공모전/ensemble_predictions" # ★★★ 팀과 공유할 실제 경로로 수정 ★★★

# 2. 앙상블에 포함할 기본 모델 이름 리스트 (필수 수정)
BASE_MODEL_NAMES = ['lgbm', 'mlp', 'ridge_poly', 'rf'] # ★★★ 사용할 모델 이름 확인/수정 ★★★

# 3. 사용할 메타 모델 타입 선택 (필수 수정)
#    옵션: 'ridgecv', 'linear', 'lassocv', 'elasticnetcv', 'gbr', 'lgbm'
META_MODEL_TYPE = 'ridgecv' # ★★★ 원하는 메타 모델 타입으로 변경 ★★★
# META_MODEL_TYPE = 'lgbm' # 예시: LightGBM 사용 시

# 4. (선택 사항) 홀드아웃 테스트 성능 계산용 실제값 파일 경로
Y_TEST_FILE_PATH = os.path.join(PREDICTION_DIR, "y_test.csv") # ★★★ 경로 확인, 없으면 None ★★★
# Y_TEST_FILE_PATH = None # 성능 계산 안 할 경우

# 5. 최종 제출 파일 이름 (선택 사항)
SUBMISSION_FILENAME = "./submission_final_ensemble.csv" # ★★★ 원하는 파일명으로 수정 ★★★

# 6. 메타 모델 파라미터 설정 (선택 사항, 해당 모델 사용 시 적용됨)
#    - RidgeCV, LassoCV, ElasticNetCV 용 Alpha 후보
ALPHAS_TO_TEST = np.logspace(-3, 2, 6)
#    - RidgeCV, LassoCV, ElasticNetCV 용 내부 교차검증 폴드
CV_FOLDS = 5
#    - ElasticNetCV 용 L1 비율 후보 (Lasso(1) ~ Ridge(0))
ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
#    - GBR, LGBM 용 파라미터 (기본값 사용, 필요시 수정)
GBR_PARAMS = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42}
LGBM_PARAMS = {
    'n_estimators': 3000,
    'learning_rate': 0.08745934114377983,
    'num_leaves': 117,
    'max_depth': 10, 
    'subsample': 0.8277787647225167,
    'colsample_bytree': 0.7819184951611192,
    'reg_alpha': 0.05835834054800604,
    'reg_lambda': 0.0038123006520339904,
    'min_child_samples': 7,
    'objective': 'regression_l1', # MAE 목적 함수
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1 # 학습 로그 최소화
    # 'metric' 파라미터는 fit 시 지정 가능 (e.g., for early stopping)
}
# ======================================================================
#                      ★★★ 설정 섹션 끝 ★★★
#         (아래 코드는 일반적으로 수정할 필요가 없습니다)
# ======================================================================

def check_files_exist(directory, model_names, y_test_path):
    """필요한 파일 존재 여부 확인"""
    missing_files = []
    required_files = ["y_val.csv"] # 메타 모델 학습용 타겟 파일 (필수)
    if y_test_path and os.path.exists(os.path.dirname(y_test_path)): # 경로가 유효하고 None이 아니면
        required_files.append(os.path.basename(y_test_path))

    for name in model_names:
        required_files.extend([f"{name}_val_preds.csv", f"{name}_test_preds.csv"])

    print(f"  - 확인 대상 디렉토리: {directory}")
    print(f"  - 확인할 필수 파일: y_val.csv, " + ", ".join([f"{name}_*.csv" for name in model_names]) + (f", {os.path.basename(y_test_path)}" if y_test_path else ""))

    all_exist = True
    if not os.path.isdir(directory):
        print(f"\n★★★ 오류: 지정된 디렉토리를 찾을 수 없습니다: {directory} ★★★")
        return False

    for filename in required_files:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
            all_exist = False

    if not all_exist:
        print("\n★★★ 오류: 다음 필수 파일들을 찾을 수 없습니다! ★★★")
        for f in missing_files:
            print(f"  - {f}")
        print(f"'{directory}' 경로와 BASE_MODEL_NAMES 리스트를 확인하세요.")
        return False
    else:
        print("  - 모든 필수 파일 존재 확인 완료.")
        return True

def main():
    """메인 스태킹 앙상블 실행 함수"""
    print("="*60)
    print(f"      스태킹 앙상블 스크립트 시작 (메타 모델: {META_MODEL_TYPE.upper()})") # 선택된 모델 표시
    print("="*60)
    print(f"설정:")
    print(f"  - 예측 파일 디렉토리: {PREDICTION_DIR}")
    print(f"  - 기본 모델 리스트: {BASE_MODEL_NAMES}")
    print(f"  - 메타 모델 타입: {META_MODEL_TYPE}") # 설정 표시
    print(f"  - 테스트 성능 계산용 파일: {Y_TEST_FILE_PATH if Y_TEST_FILE_PATH else '미지정'}")
    print(f"  - 최종 제출 파일명: {SUBMISSION_FILENAME}")
    print("-"*60)

    # --- 단계 0: 필수 파일 존재 여부 확인 ---
    print("\n--- 단계 0: 필수 파일 존재 여부 확인 ---")
    if not check_files_exist(PREDICTION_DIR, BASE_MODEL_NAMES, Y_TEST_FILE_PATH):
        sys.exit(1) # 파일 없으면 종료

    # --- 단계 1: 파일 로드 ---
    print("\n--- 단계 1: 파일 로드 ---")
    pred_val = {}
    pred_test = {}
    y_val_log = None
    y_test_log = None

    try:
        print(f"  - 검증 예측값 로드 ({len(BASE_MODEL_NAMES)}개 모델)...")
        for name in BASE_MODEL_NAMES:
            path = os.path.join(PREDICTION_DIR, f"{name}_val_preds.csv")
            pred_val[name] = pd.read_csv(path, header=None)[0]

        print(f"  - 테스트 예측값 로드 ({len(BASE_MODEL_NAMES)}개 모델)...")
        for name in BASE_MODEL_NAMES:
            path = os.path.join(PREDICTION_DIR, f"{name}_test_preds.csv")
            pred_test[name] = pd.read_csv(path, header=None)[0]

        y_val_path = os.path.join(PREDICTION_DIR, "y_val.csv")
        print(f"  - 검증 실제 타겟값 로드 ({y_val_path})...")
        y_val_log = pd.read_csv(y_val_path, header=None)[0]

        if Y_TEST_FILE_PATH and os.path.exists(Y_TEST_FILE_PATH):
            print(f"  - 테스트 실제 타겟값 로드 ({Y_TEST_FILE_PATH})...")
            y_test_log = pd.read_csv(Y_TEST_FILE_PATH, header=None)[0]
        elif Y_TEST_FILE_PATH:
             print(f"  - 정보: 테스트 실제 타겟값 파일({Y_TEST_FILE_PATH})을 찾을 수 없어 성능 계산 생략.")
        else:
             print(f"  - 정보: 테스트 실제 타겟값 파일 경로 미지정으로 성능 계산 생략.")

        print("  - 파일 로드 완료.")
    except FileNotFoundError as e: # 구체적인 오류 처리
        print(f"★★★ 오류: 파일 로드 실패! '{e.filename}' 파일을 찾을 수 없습니다. ★★★")
        sys.exit(1)
    except Exception as e:
        print(f"★★★ 오류: 파일 로드 중 예외 발생: {e} ★★★")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 단계 2: 메타 모델 입력 데이터 준비 ---
    print("\n--- 단계 2: 메타 모델 입력 데이터 준비 ---")
    try:
        X_meta_train = pd.DataFrame(pred_val)
        y_meta_train = y_val_log
        X_meta_test = pd.DataFrame(pred_test)

        # 데이터 정합성 검사 (NaN 값 확인 추가)
        if X_meta_train.isnull().sum().sum() > 0 or y_meta_train.isnull().sum() > 0:
            print("★★★ 경고: 메타 학습 데이터 또는 타겟에 NaN 값이 포함되어 있습니다! ★★★")
            # 필요시 처리 로직 추가 (예: sys.exit(1) 또는 fillna)
        if X_meta_test.isnull().sum().sum() > 0:
            print("★★★ 경고: 메타 테스트 데이터에 NaN 값이 포함되어 있습니다! ★★★")

        if not (X_meta_train.shape[0] == y_meta_train.shape[0]): raise ValueError("행 수 불일치 (메타 학습 데이터/타겟)")
        if y_test_log is not None and not (X_meta_test.shape[0] == y_test_log.shape[0]):
             print(f"  - 경고: 테스트 예측({X_meta_test.shape[0]})과 y_test({y_test_log.shape[0]}) 행 수 불일치!")
        if not (X_meta_train.shape[1] == X_meta_test.shape[1] == len(BASE_MODEL_NAMES)): raise ValueError("열 수 불일치 (메타 학습/테스트 또는 모델 수)")

        print(f"  - 메타 학습 X: {X_meta_train.shape}, 메타 학습 y: {y_meta_train.shape}")
        print(f"  - 메타 테스트 X: {X_meta_test.shape}")
    except Exception as e:
         print(f"★★★ 오류: 메타 데이터 준비 중 예외 발생: {e} ★★★")
         sys.exit(1)

    # --- 단계 3: 메타 모델 학습 및 예측 ---
    print(f"\n--- 단계 3: 메타 모델({META_MODEL_TYPE.upper()}) 학습 및 예측 ---")
    try:
        meta_model = None
        model_type_lower = META_MODEL_TYPE.lower()

        # 메타 모델 인스턴스화
        if model_type_lower == 'ridgecv':
            meta_model = RidgeCV(alphas=ALPHAS_TO_TEST, cv=CV_FOLDS, scoring='neg_root_mean_squared_error')
        elif model_type_lower == 'linear':
            meta_model = LinearRegression()
        elif model_type_lower == 'lassocv':
            meta_model = LassoCV(alphas=ALPHAS_TO_TEST, cv=CV_FOLDS, random_state=42, max_iter=10000) # max_iter 증가 고려
        elif model_type_lower == 'elasticnetcv':
            meta_model = ElasticNetCV(alphas=ALPHAS_TO_TEST, l1_ratio=ELASTICNET_L1_RATIOS, cv=CV_FOLDS, random_state=42, max_iter=10000) # max_iter 증가 고려
        elif model_type_lower == 'gbr':
            meta_model = GradientBoostingRegressor(**GBR_PARAMS)
        elif model_type_lower == 'lgbm':
            if lgb is None:
                raise ImportError("LightGBM 모델을 사용하려면 'lightgbm' 라이브러리를 설치해야 합니다. (pip install lightgbm)")
            meta_model = lgb.LGBMRegressor(**LGBM_PARAMS)
        else:
            raise ValueError(f"지원하지 않는 메타 모델 타입입니다: {META_MODEL_TYPE}. "
                             f"옵션: 'ridgecv', 'linear', 'lassocv', 'elasticnetcv', 'gbr', 'lgbm'")

        # 메타 모델 학습
        print(f"  - {model_type_lower.upper()} 모델 학습 시작...")
        meta_model.fit(X_meta_train, y_meta_train)
        print(f"  - 메타 모델 학습 완료.")

        # 학습된 모델 정보 출력 (모델 타입에 따라 다름)
        if hasattr(meta_model, 'alpha_'): # RidgeCV, LassoCV, ElasticNetCV
            print(f"    최적 Alpha: {meta_model.alpha_:.4f}")
        if hasattr(meta_model, 'l1_ratio_') and model_type_lower == 'elasticnetcv': # ElasticNetCV
            print(f"    최적 L1 Ratio: {meta_model.l1_ratio_:.4f}")
        if hasattr(meta_model, 'coef_'): # Linear, Ridge, Lasso, ElasticNet
             print(f"    학습된 계수: {dict(zip(X_meta_train.columns, np.round(meta_model.coef_, 4)))}")
        if hasattr(meta_model, 'feature_importances_'): # GBR, LGBM
             print(f"    피처 중요도: {dict(zip(X_meta_train.columns, np.round(meta_model.feature_importances_, 4)))}")

        # 메타 모델 예측
        y_pred_log_ensemble = meta_model.predict(X_meta_test)
        print("  - 최종 앙상블 예측값 생성 완료 (로그 스케일).")

    except ValueError as e:
        print(f"★★★ 오류: {e} ★★★")
        sys.exit(1)
    except ImportError as e:
        print(f"★★★ 오류: {e} ★★★")
        sys.exit(1)
    except Exception as e:
         print(f"★★★ 오류: 메타 모델 학습 또는 예측 중 예외 발생: {e} ★★★")
         import traceback
         traceback.print_exc()
         sys.exit(1)

    # --- 단계 4: 최종 예측값 변환 ---
    print("\n--- 단계 4: 최종 예측값 변환 (원본 스케일) ---")
    y_pred_final = np.expm1(y_pred_log_ensemble)
    y_pred_final[y_pred_final < 0] = 0 # 음수 처리
    print("  - 역변환 및 음수 처리 완료.")

    # --- 단계 5: 성능 평가 (y_test_log 가 있을 경우) ---
    if y_test_log is not None:
        print("\n--- 단계 5: 최종 성능 평가 (테스트셋 기준) ---")
        try:
            y_test_original = np.expm1(y_test_log)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_final))
            r2 = r2_score(y_test_original, y_pred_final)
            adj_r2 = np.nan
            n = len(y_test_original)
            # k = X_meta_test.shape[1] # 원래 k는 피처 수
            # Adjusted R2 계산 시 k는 모델 파라미터 수로 보는 것이 더 정확할 수 있으나,
            # 여기서는 단순하게 기본 모델(피처) 수로 유지합니다.
            k = X_meta_train.shape[1]
            if (n - k - 1) > 0:
                adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

            print(f"  - RMSE (원본): {rmse:.4f}")
            print(f"  - R2 (원본):   {r2:.4f}")
            if not np.isnan(adj_r2): print(f"  - Adj R2 (원본): {adj_r2:.4f}")
        except Exception as e:
             print(f"  - 오류: 성능 평가 중 예외 발생: {e}")
    else:
        print("\n--- 단계 5: 최종 성능 평가 ---")
        print("  - y_test.csv 파일이 지정되지 않아 성능 계산을 생략합니다.")


    # --- 단계 6: 제출 파일 생성 ---
    print("\n--- 단계 6: 제출 파일 생성 ---")
    try:
        submission_df = pd.DataFrame({'y_pred': y_pred_final}) # 가장 기본적인 형태
        submission_df.to_csv(SUBMISSION_FILENAME, index=False)
        print(f"  - 제출 파일 생성 완료: {SUBMISSION_FILENAME}")
        print(f"  - 생성된 파일 미리보기 (상위 5개):\n{submission_df.head()}")
    except Exception as e:
        print(f"★★★ 오류: 제출 파일 생성 실패: {e} ★★★")
        sys.exit(1)

    print("\n"+"="*60)
    print(f"      스태킹 앙상블({META_MODEL_TYPE.upper()}) 스크립트 성공적으로 완료됨")
    print("="*60)

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 main 함수 호출
    main()