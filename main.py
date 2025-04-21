from sklearn.ensemble import RandomForestRegressor
import os
from modules import (
    DataLoader,
    Preprocessor,
    ModelTrainer,
    ModelOptimizer,
    setup_logging,
    save_model,
    save_predictions,
    inverse_transform_predictions
)

# LightGBM 경고 메시지 억제
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    # 로깅 설정
    logger = setup_logging()
    
    try:
        # results 디렉토리 생성
        os.makedirs('results/models', exist_ok=True)
        os.makedirs('results/predictions', exist_ok=True)
        
        # 데이터 로더 초기화
        data_loader = DataLoader()
        
        # 데이터 로드
        train_df, test_df = data_loader.load_data(
            train_path='data/TRAIN_DATA.csv',
            test_path='data/TEST_DATA.csv'
        )
        
        # 모델 이름 설정 (사용할 모델 선택) # RF 추가예정
        model_name = 'HBM' # 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'GBM', 'HBM' 중 선택
        
        # 전처리기 초기화 (모델 타입 정보 전달)
        preprocessor = Preprocessor(model_type=model_name)
        
        # 데이터 전처리
        X_train_scaled, y_train, X_test_scaled = preprocessor.preprocess_data(
            train_df=train_df,
            test_df=test_df
        )
        
        # ====== 기존 모델 설정 방식 (주석 처리) ======
        """
        # XGBoost 모델 설정
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # LightGBM 모델 설정
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # CatBoost 모델 설정
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_seed=42,
            verbose=0
        )
        
        # 모델 트레이너 초기화 (5-fold 교차검증 설정)
        model_trainer = ModelTrainer(
            model=model,
            n_splits=5,
            random_state=42,
            cat_features=preprocessor.catboost_cat_features if model_name == 'CatBoost' else None
        )
        
        # 모델 학습 (교차검증 포함)
        trained_model = model_trainer.train(X_train_scaled, y_train)
        """
        # ====== 그리드 서치를 통한 모델 최적화 방식 ======
        
        # 모델 최적화 객체 초기화
        model_optimizer = ModelOptimizer(
            model_type=model_name,
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=2
        )
        
        # 그리드 서치를 통한 최적 모델 학습
        logger.info("그리드 서치를 통한 최적 모델 학습 시작")
        best_model = model_optimizer.optimize(
            X_train_scaled, 
            y_train, 
            cat_features=preprocessor.catboost_cat_features if model_name == 'CatBoost' else None
        )
        
        # 최적 파라미터 출력
        best_params = model_optimizer.get_best_params()
        logger.info(f"최적 하이퍼파라미터: {best_params}")
        
        # 모델 최적화 상세 결과 출력
        model_optimizer.print_optimization_results()
        
        # 특성 중요도 확인 (가능한 경우)
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = model_optimizer.get_feature_importance(preprocessor.feature_names)
            if feature_importance is not None:
                logger.info(f"상위 10개 중요 특성:\n{feature_importance.head(10)}")
        
        # 모델 저장
        model_path = save_model(
            model=best_model,
            scaler=preprocessor.scaler,
            feature_names=preprocessor.feature_names,
            model_name=f"{model_name}_optimized",
            column_transformer=preprocessor.column_transformer
        )
        
        logger.info(f"최적화된 모델이 저장되었습니다: {model_path}")
        
        # 테스트 데이터 예측
        predictions = model_optimizer.predict(X_test_scaled)
        
        # 예측값을 원래 스케일로 변환
        predictions = inverse_transform_predictions(predictions)
        
        # 예측 결과 저장
        predictions_path = save_predictions(
            test_df=test_df,
            predictions=predictions,
            model_name=f"{model_name}_optimized"
        )
        
        logger.info("모든 작업이 성공적으로 완료되었습니다.")
        logger.info(f"모델 저장 경로: {model_path}")
        logger.info(f"예측 결과 저장 경로: {predictions_path}")
        
    except Exception as e:
        logger.error(f"작업 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()