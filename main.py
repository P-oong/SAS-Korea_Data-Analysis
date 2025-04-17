from sklearn.ensemble import RandomForestRegressor
import os
from modules import (
    DataLoader,
    Preprocessor,
    ModelTrainer,
    setup_logging,
    save_model,
    save_predictions,
    inverse_transform_predictions
)

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
        
        # 전처리기 초기화
        preprocessor = Preprocessor()
        
        # 데이터 전처리
        X_train_scaled, y_train, X_test_scaled = preprocessor.preprocess_data(
            train_df=train_df,
            test_df=test_df
        )
        
        # 모델 이름 설정
        model_name = 'RandomForest_baseline'
        
        # 모델 인스턴스 생성
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # # XGBoost 모델 설정
        # from xgboost import XGBRegressor
        # model_name = 'XGBoost'
        # model = XGBRegressor(
        #     n_estimators=100,
        #     max_depth=6,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42
        # )
        
        # #LightGBM 모델 설정
        # from lightgbm import LGBMRegressor
        # model_name = 'LightGBM'
        # model = LGBMRegressor(
        #     n_estimators=100,
        #     max_depth=6,
        #     learning_rate=0.1,
        #     num_leaves=31,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     random_state=42
        # )
        
        # # CatBoost 모델 설정
        # from catboost import CatBoostRegressor
        # model_name = 'CatBoost'
        # model = CatBoostRegressor(
        #     iterations=100,
        #     depth=6,
        #     learning_rate=0.1,
        #     random_seed=42,
        #     verbose=0
        # )
        
        # #Gradient Boosting Machine 모델 설정
        # from sklearn.ensemble import GradientBoostingRegressor
        # model_name = 'GBM'
        # model = GradientBoostingRegressor(
        #     n_estimators=100,
        #     max_depth=6,
        #     learning_rate=0.1,
        #     subsample=0.8,
        #     random_state=42
        # )
        
        # # Histogram-based Gradient Boosting 모델 설정
        # from sklearn.ensemble import HistGradientBoostingRegressor
        # model_name = 'HBM'
        # model = HistGradientBoostingRegressor(
        #     max_iter=100,
        #     max_depth=6,
        #     learning_rate=0.1,
        #     random_state=42
        # )
        
        # 모델 트레이너 초기화 (5-fold 교차검증 설정)
        model_trainer = ModelTrainer(
            model=model,
            n_splits=5,
            random_state=42
        )
        
        # 모델 학습 (교차검증 포함)
        trained_model = model_trainer.train(X_train_scaled, y_train)
        
        # 모델 저장
        model_path = save_model(
            model=trained_model,
            scaler=preprocessor.scaler,
            feature_names=preprocessor.feature_names,
            model_name=model_name,
            column_transformer=preprocessor.column_transformer
        )
        
        logger.info(f"모델이 저장되었습니다: {model_path}")
        
        # 테스트 데이터 예측
        predictions = model_trainer.predict(X_test_scaled)
        
        # 예측값을 원래 스케일로 변환
        predictions = inverse_transform_predictions(predictions)
        
        # 예측 결과 저장
        predictions_path = save_predictions(
            test_df=test_df,
            predictions=predictions,
            model_name=model_name
        )
        
        logger.info("모든 작업이 성공적으로 완료되었습니다.")
        logger.info(f"모델 저장 경로: {model_path}")
        logger.info(f"예측 결과 저장 경로: {predictions_path}")
        
    except Exception as e:
        logger.error(f"작업 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()