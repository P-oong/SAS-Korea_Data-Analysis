import os
import joblib
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

def setup_logging():
    """
    로깅 설정을 초기화하는 함수
    
    Returns:
    --------
    logging.Logger
        설정된 로거 객체
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def save_model(model, scaler, feature_names, model_name, column_transformer=None, model_dir='results/models'):
    """
    학습된 모델 저장
    
    Parameters:
    -----------
    model : object
        학습된 모델 객체
    scaler : object
        사용된 스케일러 객체
    feature_names : list
        특성 이름 리스트
    model_name : str
        모델의 이름 (예: 'RandomForest', 'XGBoost' 등)
    column_transformer : object, optional
        사용된 ColumnTransformer 객체. 기본값은 None
    model_dir : str, optional
        모델을 저장할 디렉토리. 기본값은 'results/models'
        
    Returns:
    --------
    str
        저장된 모델 파일 경로
    """
    try:
        # 모델 저장 디렉토리 생성
        os.makedirs(model_dir, exist_ok=True)
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'{model_name}_model_{timestamp}.pkl')
        
        # 모델과 관련 객체들을 딕셔너리로 저장
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'column_transformer': column_transformer,
            'timestamp': timestamp,
            'model_name': model_name
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"모델이 저장되었습니다: {model_path}")
        
        return model_path
        
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {str(e)}")
        raise

def save_predictions(test_df, predictions, model_name, output_dir='results/predictions'):
    """
    예측 결과를 CSV 파일로 저장
    
    Parameters:
    -----------
    test_df : pandas.DataFrame
        테스트 데이터프레임
    predictions : array-like
        예측값
    model_name : str
        모델의 이름 (예: 'RandomForest', 'XGBoost' 등)
    output_dir : str, optional
        예측 결과를 저장할 디렉토리. 기본값은 'results/predictions'
        
    Returns:
    --------
    str
        저장된 예측 결과 파일 경로
    """
    try:
        # 예측 결과 저장 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_dir, f'{model_name}_predictions_{timestamp}.csv')
        
        # 예측 결과를 테스트 데이터프레임에 추가
        test_df['predicted_TOTAL_ELEC'] = predictions
        
        # 예측 결과 저장
        test_df.to_csv(output_path, index=False, encoding='cp949')
        logger.info(f"예측 결과가 저장되었습니다: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"예측 결과 저장 중 오류 발생: {str(e)}")
        raise

def inverse_transform_predictions(predictions):
    """
    로그 변환된 예측값을 원래 스케일로 변환
    
    Parameters:
    -----------
    predictions : array-like
        로그 변환된 예측값
        
    Returns:
    --------
    array
        원래 스케일로 변환된 예측값
    """
    return np.expm1(predictions) 