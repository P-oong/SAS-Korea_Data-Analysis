import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    데이터 로딩을 담당하는 클래스
    """
    
    def __init__(self):
        """
        DataLoader 클래스 초기화
        """
        self.train_df = None
        self.test_df = None
        
    def load_data(self, train_path, test_path):
        """
        학습 데이터와 테스트 데이터를 로드
        
        Parameters:
        -----------
        train_path : str
            학습 데이터 파일 경로
        test_path : str
            테스트 데이터 파일 경로
            
        Returns:
        --------
        tuple
            (train_df, test_df) 형태의 튜플
        """
        try:
            logger.info("데이터 로딩 시작...")
            
            # 학습 데이터 로드
            self.train_df = pd.read_csv(train_path, encoding='cp949')
            logger.info(f"학습 데이터 로드 완료: {self.train_df.shape}")
            
            # 테스트 데이터 로드
            self.test_df = pd.read_csv(test_path, encoding='cp949')
            logger.info(f"테스트 데이터 로드 완료: {self.test_df.shape}")
            
            return self.train_df, self.test_df
            
        except Exception as e:
            logger.error(f"데이터 로딩 중 오류 발생: {str(e)}")
            raise 