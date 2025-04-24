# ensemble_weighted_average.py
# 사용법:
# 1. 아래 ★ 설정 섹션 ★ 의 경로 및 모델 이름을 사용자 환경과 파일에 맞게 수정.
# 2. 필요한 모든 예측값/실제값 CSV 파일을 PREDICTION_DIR 폴더에 준비.
#    (로그 스케일 가정, 헤더X, 인덱스X, 파일명 규칙: {모델명}_val_preds.csv 등)
# 3. 터미널에서 python ensemble_weighted_average.py 를 실행.

import pandas as pd
import numpy as np
import os
import sys # 오류 발생 시 종료용
from sklearn.metrics import mean_squared_error, r2_score # 성능 확인용

# ======================================================================
#                           ★ 설정 섹션 ★
# ======================================================================
# 이 스크립트 사용자는 이 섹션의 값들을 확인하고 수정해야 합니다.

# 1. 예측/실제값 CSV 파일들이 저장된 폴더 경로 (필수 수정)
PREDICTION_DIR = "/Users/idoyeong/계명대학교/sas공모전/ensemble_predictions" # ★★★ 팀과 공유할 실제 경로로 수정 ★★★

# 2. 앙상블에 포함할 기본 모델 이름 리스트 (필수 수정)
BASE_MODEL_NAMES = ['lgbm', 'mlp', 'ridge_poly', 'rf'] # ★★★ 사용할 모델 이름 확인/수정 ★★★

# 3. (선택 사항) 홀드아웃 테스트 성능 계산용 실제값 파일 경로
#    이 파일의 값들도 로그 스케일이어야 함 (원본 스케일 RMSE 계산 시 내부적으로 변환)
Y_TEST_FILE_PATH = os.path.join(PREDICTION_DIR, "y_test.csv") # ★★★ 경로 확인, 없으면 None ★★★
# Y_TEST_FILE_PATH = None # 성능 계산 안 할 경우

# 4. 최종 제출 파일 이름 (선택 사항)
SUBMISSION_FILENAME = "./submission_weighted_average.csv" # ★★★ 원하는 파일명으로 수정 ★★★

# 5. 가중치 계산 시 RMSE 역수에 더할 작은 값 (0으로 나누기 방지)
EPSILON = 1e-9 # 0에 가까운 RMSE 처리용

# ======================================================================
#                      ★★★ 설정 섹션 끝 ★★★
#         (아래 코드는 일반적으로 수정할 필요가 없습니다)
# ======================================================================

def check_files_exist(directory, model_names, y_test_path):
    """필요한 파일 존재 여부 확인"""
    missing_files = []
    # 가중치 계산을 위해 y_val.csv는 필수
    required_files = ["y_val.csv"]
    if y_test_path and os.path.exists(os.path.dirname(y_test_path)): # 경로가 유효하고 None이 아니면
        # y_test.csv 파일 이름은 설정에서 가져옴
        required_files.append(os.path.basename(y_test_path))

    # 각 모델별 검증 예측값(RMSE 계산용), 테스트 예측값(최종 예측용) 필요
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

def rmse(y_true, y_pred):
    """RMSE 계산 함수"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    """메인 가중 평균 앙상블 실행 함수"""
    print("="*60)
    print("      RMSE 기반 가중 평균 앙상블 스크립트 시작")
    print("="*60)
    print(f"설정:")
    print(f"  - 예측/실제값 파일 디렉토리: {PREDICTION_DIR}")
    print(f"  - 기본 모델 리스트: {BASE_MODEL_NAMES}")
    print(f"  - 테스트 성능 계산용 파일: {Y_TEST_FILE_PATH if Y_TEST_FILE_PATH else '미지정'}")
    print(f"  - 최종 제출 파일명: {SUBMISSION_FILENAME}")
    print(f"  - RMSE 역수 계산 시 Epsilon: {EPSILON}")
    print("-"*60)
    print("주의: 모든 입력 CSV 파일(예측값, 실제값)은 로그 스케일 값을 가정합니다.")
    print("-"*60)

    # --- 단계 0: 필수 파일 존재 여부 확인 ---
    print("\n--- 단계 0: 필수 파일 존재 여부 확인 ---")
    # y_test.csv 파일 경로를 정확히 전달
    if not check_files_exist(PREDICTION_DIR, BASE_MODEL_NAMES, Y_TEST_FILE_PATH):
        sys.exit(1) # 파일 없으면 종료

    # --- 단계 1: 파일 로드 (로그 스케일 가정) ---
    print("\n--- 단계 1: 파일 로드 (로그 스케일) ---")
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

        # 데이터 길이 검증
        val_len = len(y_val_log)
        for name in BASE_MODEL_NAMES:
            if len(pred_val[name]) != val_len:
                raise ValueError(f"검증 데이터 길이 불일치: y_val({val_len}) vs {name}_val_preds({len(pred_val[name])})")
        if Y_TEST_FILE_PATH and y_test_log is not None:
            test_len = len(y_test_log)
            for name in BASE_MODEL_NAMES:
                 if len(pred_test[name]) != test_len:
                     print(f"  - 경고: 테스트 데이터 길이 불일치: y_test({test_len}) vs {name}_test_preds({len(pred_test[name])})")
        else:
            # y_test가 없을 경우, test 예측값들의 길이가 모두 같은지만 확인
            if len(BASE_MODEL_NAMES) > 0:
                first_test_len = len(pred_test[BASE_MODEL_NAMES[0]])
                for name in BASE_MODEL_NAMES[1:]:
                    if len(pred_test[name]) != first_test_len:
                        raise ValueError(f"테스트 예측값 간 길이 불일치: {BASE_MODEL_NAMES[0]}({first_test_len}) vs {name}({len(pred_test[name])})")


        print("  - 파일 로드 완료.")
    except FileNotFoundError as e: # 구체적인 오류 처리
        print(f"★★★ 오류: 파일 로드 실패! '{e.filename}' 파일을 찾을 수 없습니다. ★★★")
        sys.exit(1)
    except ValueError as e:
        print(f"★★★ 오류: 데이터 로드 중 오류 발생: {e} ★★★")
        sys.exit(1)
    except Exception as e:
        print(f"★★★ 오류: 파일 로드 중 예외 발생: {e} ★★★")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 단계 2: 모델별 검증 RMSE 계산 및 가중치 결정 ---
    print("\n--- 단계 2: 모델별 검증 RMSE 계산 및 가중치 결정 ---")
    model_rmses = {}
    model_weights = {}

    try:
        print("  - 각 모델의 검증셋 RMSE 계산 (로그 스케일 기준)...")
        for name in BASE_MODEL_NAMES:
            # 로그 스케일에서 RMSE 계산
            current_rmse = rmse(y_val_log, pred_val[name])
            model_rmses[name] = current_rmse
            print(f"    - {name}: RMSE = {current_rmse:.6f}")

        # RMSE 값의 유효성 확인 (NaN 또는 무한대)
        if any(np.isnan(r) or np.isinf(r) for r in model_rmses.values()):
             print("★★★ 오류: 계산된 RMSE 값 중에 NaN 또는 무한대가 포함되어 있습니다! ★★★")
             print(f"  - 계산된 RMSEs: {model_rmses}")
             sys.exit(1)

        # 가중치 계산 (Inverse RMSE, normalized)
        print("\n  - RMSE 기반 가중치 계산 (낮을수록 높은 가중치)...")
        raw_weights = {name: 1.0 / (rmse_val + EPSILON) for name, rmse_val in model_rmses.items()}
        total_raw_weight = sum(raw_weights.values())

        if total_raw_weight <= 0: # 모든 RMSE가 무한대에 가깝거나 다른 문제 발생 시
             print("★★★ 오류: 가중치 합이 0 이하입니다. RMSE 값을 확인하세요. ★★★")
             print(f"  - 계산된 RMSEs: {model_rmses}")
             print(f"  - Raw Weights: {raw_weights}")
             sys.exit(1)

        # 정규화된 가중치 계산
        model_weights = {name: w / total_raw_weight for name, w in raw_weights.items()}

        print("  - 계산된 최종 가중치:")
        for name, weight in model_weights.items():
            print(f"    - {name}: Weight = {weight:.6f} (Based on RMSE: {model_rmses[name]:.6f})")

    except Exception as e:
        print(f"★★★ 오류: 가중치 계산 중 예외 발생: {e} ★★★")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 단계 3: 가중 평균 앙상블 예측 수행 (로그 스케일) ---
    print("\n--- 단계 3: 가중 평균 앙상블 예측 수행 (로그 스케일) ---")
    try:
        # 테스트 예측값들을 DataFrame으로 결합
        X_test_preds = pd.DataFrame(pred_test)

        # 가중 평균 계산
        y_pred_log_ensemble = np.zeros(X_test_preds.shape[0])
        print("  - 각 모델의 테스트 예측값에 가중치 적용 및 합산...")
        for name in BASE_MODEL_NAMES:
            weight = model_weights[name]
            y_pred_log_ensemble += X_test_preds[name] * weight
            print(f"    - {name} (Weight: {weight:.4f}) 기여분 추가됨")

        print("  - 최종 가중 평균 앙상블 예측값 생성 완료 (로그 스케일).")
    except Exception as e:
         print(f"★★★ 오류: 앙상블 예측 생성 중 예외 발생: {e} ★★★")
         import traceback
         traceback.print_exc()
         sys.exit(1)


    # --- 단계 4: 최종 예측값 변환 (원본 스케일) ---
    print("\n--- 단계 4: 최종 예측값 변환 (원본 스케일) ---")
    try:
        y_pred_final = np.expm1(y_pred_log_ensemble)
        # np.expm1 계산 결과가 아주 작은 음수가 될 수 있음 (부동소수점 오류)
        y_pred_final[y_pred_final < 0] = 0 # 음수 값 0으로 처리
        print("  - 역변환(np.expm1) 및 음수 처리 완료.")
    except Exception as e:
        print(f"★★★ 오류: 최종 예측값 변환 중 예외 발생: {e} ★★★")
        sys.exit(1)


    # --- 단계 5: 최종 성능 평가 (y_test_log 가 있을 경우, 원본 스케일 기준) ---
    if y_test_log is not None:
        print("\n--- 단계 5: 최종 성능 평가 (테스트셋, 원본 스케일 기준) ---")
        try:
            # y_test도 원본 스케일로 변환 필요
            y_test_original = np.expm1(y_test_log)
            y_test_original[y_test_original < 0] = 0 # 일관성을 위해 음수 처리 (원본에 음수가 없어야 함)

            final_rmse = rmse(y_test_original, y_pred_final)
            final_r2 = r2_score(y_test_original, y_pred_final)
            adj_r2 = np.nan
            n = len(y_test_original)
            # 가중 평균은 명시적인 피처 수가 없으므로, k=1 (단순 평균 모델) 또는 k=모델 수 등 해석 여지 있음
            # 여기서는 스태킹과 유사하게 기본 모델 수를 k로 사용
            k = len(BASE_MODEL_NAMES)
            if (n - k - 1) > 0:
                adj_r2 = 1 - ((1 - final_r2) * (n - 1) / (n - k - 1))

            print(f"  - 최종 앙상블 RMSE (원본): {final_rmse:.4f}")
            print(f"  - 최종 앙상블 R2 (원본):   {final_r2:.4f}")
            if not np.isnan(adj_r2): print(f"  - 최종 앙상블 Adj R2 (원본): {adj_r2:.4f}")

            print("\n  - 참고: 개별 모델의 테스트셋 성능 (로그 스케일 RMSE):")
            for name in BASE_MODEL_NAMES:
                # 개별 모델의 로그 스케일 RMSE 계산 (참고용)
                try:
                    individual_test_rmse_log = rmse(y_test_log, pred_test[name])
                    print(f"    - {name} (Log RMSE): {individual_test_rmse_log:.6f}")
                except Exception as e_indiv:
                    print(f"    - {name}: 테스트셋 성능 계산 오류 - {e_indiv}")

        except Exception as e:
             print(f"  - 오류: 성능 평가 중 예외 발생: {e}")
    else:
        print("\n--- 단계 5: 최종 성능 평가 ---")
        print("  - y_test.csv 파일이 지정되지 않아 성능 계산을 생략합니다.")


    # --- 단계 6: 제출 파일 생성 ---
    print("\n--- 단계 6: 제출 파일 생성 ---")
    try:
        submission_df = pd.DataFrame({'y_pred': y_pred_final})
        submission_df.to_csv(SUBMISSION_FILENAME, index=False, header=True) # 헤더 추가 권장
        print(f"  - 제출 파일 생성 완료: {SUBMISSION_FILENAME}")
        print(f"  - 생성된 파일 미리보기 (상위 5개):\n{submission_df.head()}")
    except Exception as e:
        print(f"★★★ 오류: 제출 파일 생성 실패: {e} ★★★")
        sys.exit(1)

    print("\n"+"="*60)
    print(f"      RMSE 기반 가중 평균 앙상블 스크립트 성공적으로 완료됨")
    print("="*60)

if __name__ == "__main__":
    # 스크립트가 직접 실행될 때 main 함수 호출
    main()