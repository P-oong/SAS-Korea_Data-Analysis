import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    모델 학습 및 예측을 담당하는 클래스
    """
    
    def __init__(self, model=None, n_splits=5, random_state=42):
        """
        ModelTrainer 클래스 초기화
        
        Parameters:
        -----------
        model : object, optional
            사용할 모델 객체. 기본값은 None
        n_splits : int, optional
            교차 검증을 위한 분할 수. 기본값은 5
        random_state : int, optional
            랜덤 시드. 기본값은 42
        """
        self.model = model
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv_scores = []
        
    def train(self, X_train, y_train):
        """
        모델 학습 및 교차 검증 수행
        
        Parameters:
        -----------
        X_train : array-like
            학습 데이터 특성
        y_train : array-like
            학습 데이터 타겟
            
        Returns:
        --------
        object
            학습된 모델 객체
        """
        logger.info("모델 학습 시작...")
        
        try:
            # 교차 검증 설정
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            
            # 각 폴드별 학습 및 검증
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # 모델 학습
                self.model.fit(X_fold_train, y_fold_train)
                
                # 검증 세트 예측
                y_pred = self.model.predict(X_fold_val)
                
                # 예측값을 원본 스케일로 변환
                y_pred_original = np.expm1(y_pred)
                y_val_original = np.expm1(y_fold_val)
                
                # 원본 스케일에서 RMSE 계산
                rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
                self.cv_scores.append(rmse)
                
                logger.info(f"Fold {fold} RMSE: {rmse:.4f}")
            
            # 전체 데이터로 다시 학습
            self.model.fit(X_train, y_train)
            
            # 평균 RMSE 출력
            logger.info(f"평균 RMSE: {np.mean(self.cv_scores):.4f} (±{np.std(self.cv_scores):.4f})")
            
            return self.model
            
        except Exception as e:
            logger.error(f"모델 학습 중 오류 발생: {str(e)}")
            raise
            
    def predict(self, X_test):
        """
        테스트 데이터에 대한 예측 수행
        
        Parameters:
        -----------
        X_test : array-like
            테스트 데이터 특성
            
        Returns:
        --------
        array
            예측값
        """
        logger.info("테스트 데이터 예측 시작...")
        logger.info(f"예측에 사용되는 특성 수: {X_test.shape[1]}")
        logger.info("'AREA_ID'와 'DIST_CD' 등 변수는 예측에서 제외됨")
        
        try:
            predictions = self.model.predict(X_test)
            logger.info("예측 완료")
            
            return predictions
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise 