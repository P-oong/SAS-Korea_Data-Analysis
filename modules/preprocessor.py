import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    데이터 전처리를 담당하는 클래스
    """
    
    def __init__(self, model_type=None):
        """
        Preprocessor 클래스 초기화
        
        Parameters:
        -----------
        model_type : str, optional
            사용할 모델 타입 (예: 'CatBoost'). 기본값은 None
        """
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.column_transformer = None
        self.categorical_features = None
        self.numerical_features = None
        self.model_type = model_type
        self.log_transformed_features = ['TOTAL_BIDG', 'FAC_NEIGH_1', 'FAC_NEIGH_2', 
                                       'FAC_RETAIL', 'FAC_STAY', 'FAC_LEISURE', 
                                       'TOTAL_GAS', 'CMRC_GAS']
        # 유지할 원핫인코딩 변수명 리스트
        self.selected_encoded_features = [
            'DIST_강남구', 'DIST_안산시 단원구', 'DIST_중구', 'DIST_서초구', 'DIST_고양시 일산동구', 'DIST_부산진구', 'AREA_광화문역_2', 'AREA_장산역_2', 'AREA_판교역', 'DIST_안양시 동안구', 'AREA_가평터미널',
            'AREA_도리섬상점가상권', 'DIST_거창군', 'AREA_교육청', 'AREA_석동', 'AREA_롯데백화점 본점', 'AREA_양림동', 'DIST_옥천군', 'AREA_옥천읍', 'AREA_역삼역_4', 'AREA_센텀시티역_2', 'DIST_창녕군',
            'AREA_인천시청_2', 'AREA_천안역', 'AREA_구즉동', 'AREA_가산디지털단지역_1', 'AREA_충북혁신도시', 'AREA_명동', 'AREA_영통역_2', 'AREA_진천읍', 'AREA_온천장역', 'DIST_해운대구', 'AREA_신촌역_2',
            'AREA_종로3가역_2', 'AREA_하당동_1', 'DIST_동대문구', 'DIST_영등포구', 'AREA_광천동', 'DIST_사하구', 'DIST_예산군', 'AREA_쌍촌동1', 'AREA_신흥역_1', 'DIST_성남시 분당구', 'AREA_하남', 'DIST_청도군',
            'DIST_가평군', 'AREA_신림역_4', 'AREA_평택역_1', 'DIST_진도군', 'AREA_진도읍', 'DIST_전주시 완산구', 'AREA_논현역_3', 'DIST_마포구', 'AREA_교대역_4', 'AREA_국회의사당역_2', 'AREA_기장읍',
            'AREA_마산시외터미널', 'AREA_역삼역_3', 'AREA_범어역', 'AREA_죽전카페거리', 'AREA_고속터미널역', 'AREA_상무지구3', 'AREA_영산면', 'AREA_안동시청_2', 'AREA_종로3가역_1', 'DIST_청주시 서원구',
            'AREA_동대구역', 'AREA_연산9동_1', 'AREA_관양사거리', 'AREA_신당동', 'AREA_진영읍', 'DIST_금정구', 'AREA_진해역', 'AREA_정발산역_3', 'AREA_가락시장', 'AREA_다대포항역', 'DIST_통영시', 'DIST_순창군',
            'AREA_수내역', 'AREA_수원역', 'DIST_포항시 남구', 'DIST_평택시', 'AREA_평택역_3', 'AREA_간성시장 상권', 'AREA_신당역_2', 'AREA_안산참치골목', 'AREA_중마동_4', 'DIST_영덕군', 'DIST_강북구', 'DIST_순천시',
            'AREA_강경시외버스터미널', 'AREA_방이동 먹자골목', 'DIST_울진군', 'AREA_강남구청역_1', 'AREA_지곡동_1', 'AREA_분평동', 'AREA_부평역_2', 'AREA_호수공원 예천동 상권', 'DIST_성북구', 'DIST_의성군',
            'AREA_의성역', 'DIST_광산구', 'AREA_부평해물탕거리', 'AREA_화정역_2', 'AREA_사당역_2', 'AREA_경주역', 'AREA_시청역_2', 'AREA_중앙시장', 'DIST_남양주시', 'AREA_한티역', 'AREA_별망중학교앞', 'DIST_논산시',
            'DIST_서대문구', 'DIST_완주군', 'AREA_혜화역 대학로_2', 'AREA_도안상권', 'AREA_공주대학교', 'AREA_소사벌지구', 'DIST_영도구', 'DIST_남원시', 'DIST_해남군', 'AREA_양덕2동', 'AREA_지산1동',
            'DIST_영천시', 'AREA_영천역', 'AREA_진월동', 'AREA_내외동', 'DIST_계룡시', 'AREA_계룡엄사', 'DIST_천안시 서북구', 'DIST_경산시', 'DIST_고성군', 'AREA_대명리조트앞', 'DIST_공주시', 'DIST_속초시',
            'AREA_성당못역_2', 'DIST_나주시', 'AREA_노원동', 'AREA_용인사거리_2', 'DIST_익산시', 'AREA_연신내역_2', 'DIST_김해시', 'DIST_군위군', 'AREA_군위군법원', 'AREA_안암역', 'DIST_북구',
            'AREA_경산시외버스터미널', 'AREA_가산디지털단지역_2', 'DIST_평창군', 'DIST_경주시', 'AREA_양재역_2', 'DIST_군포시', 'DIST_춘천시', 'DIST_기장군', 'DIST_강서구', 'AREA_신정네거리역', 'AREA_강서구청',
            'DIST_단양군', 'AREA_가능역2-3번출구', 'AREA_주안2동', 'DIST_남동구', 'AREA_수원시청_1', 'AREA_양평역_1', 'DIST_담양군', 'DIST_무안군', 'DIST_제주시', 'DIST_창원시 마산합포구', 'AREA_경동시장_2',
            'AREA_동대신동역', 'AREA_칠성로', 'DIST_창원시 성산구', 'DIST_양평군', 'AREA_벌리동', 'DIST_이천시', 'DIST_동작구', 'AREA_고성동', 'DIST_관악구', 'AREA_구로중앙유통단지', 'DIST_인제군', 'DIST_유성구',
            'DIST_성남시 중원구', 'DIST_남구', 'DIST_오산시', 'DIST_종로구', 'DIST_구미시', 'DIST_사천시', 'DIST_군산시', 'AREA_송도1동', 'DIST_원주시', 'DIST_포천시', 'AREA_신매역', 'DIST_강동구', 'DIST_수성구',
            'AREA_중앙동_2', 'DIST_장흥군', 'AREA_장흥군청', 'AREA_중앙동', 'DIST_연수구', 'AREA_서창동', 'AREA_향남신도시', 'DIST_전주시 덕진구', 'AREA_남부시장', 'AREA_선부광장', 'AREA_신흥역_2', 'AREA_성결대',
            'DIST_충주시', 'AREA_경성대부경대역_1', 'DIST_청주시 청원구','AREA_중화산2동', 'AREA_흥국상가', 'AREA_동부시장', 'DIST_동두천시', 'DIST_영주시', 'AREA_순천시청', 'DIST_김천시',
            'AREA_오산시청', 'DIST_수원시 영통구', 'DIST_청양군', 'AREA_청양시외버스터미널', 'DIST_부천시', 'AREA_엠파크타워', 'DIST_양천구','AREA_디지털미디어시티', 'AREA_NM_유천아파트앞'
        ]
        # 추가 특성 변수 목록
        self.additional_features = ['IS_STATION', 'IS_eup']
        # 학습 데이터의 원핫인코딩 결과를 저장
        self.train_encoded_features = None
        # 각 범주형 변수 그룹
        self.category_groups = ['DIST', 'AREA']
        # CatBoost용 범주형 피처 인덱스 저장
        self.catboost_cat_features = None
        # 시간 특성 관련 변수
        self.date_start = None
        
    def create_time_features(self, train_df, test_df):
        """
        DATA_YM 변수를 사용하여 시간 관련 특성 생성
        
        Parameters:
        -----------
        train_df : pandas.DataFrame
            학습 데이터프레임
        test_df : pandas.DataFrame
            테스트 데이터프레임
            
        Returns:
        --------
        tuple
            (train_df, test_df) 형태의 시간 특성이 추가된 데이터프레임 튜플
        """
        logger.info("시간 관련 특성 생성 시작...")
        
        # DATA_YM 변수가 없으면 처리 중단
        if 'DATA_YM' not in train_df.columns or 'DATA_YM' not in test_df.columns:
            logger.warning("DATA_YM 변수가 없어 시간 특성을 생성할 수 없습니다")
            return train_df, test_df
        
        # 데이터프레임 복사
        train_df_result = train_df.copy()
        test_df_result = test_df.copy()
        
        # 1. datetime으로 변환
        try:
            train_df_result['date'] = pd.to_datetime(train_df_result['DATA_YM'].astype(str), format="%Y%m")
            test_df_result['date'] = pd.to_datetime(test_df_result['DATA_YM'].astype(str), format="%Y%m")
            logger.info("DATA_YM을 datetime으로 변환 완료")
        except Exception as e:
            logger.error(f"DATA_YM 변수 변환 중 오류: {str(e)}")
            return train_df, test_df
        
        # 2. 기본 시간 특성 생성 (year, month, quarter)
        train_df_result['year'] = train_df_result['date'].dt.year
        train_df_result['month'] = train_df_result['date'].dt.month
        train_df_result['quarter'] = train_df_result['date'].dt.quarter
        
        test_df_result['year'] = test_df_result['date'].dt.year
        test_df_result['month'] = test_df_result['date'].dt.month
        test_df_result['quarter'] = test_df_result['date'].dt.quarter
        
        logger.info("기본 시간 특성(year, month, quarter) 생성 완료")
        
        # 3. 연도를 범주형으로 변환
        train_df_result['year'] = train_df_result['year'].astype('category')
        test_df_result['year'] = test_df_result['year'].astype('category')
        logger.info("연도를 범주형 변수로 변환 완료")
        
        # 4. 시작 시점 계산 (train 기준) 및 개월 수(t) 계산
        self.date_start = train_df_result['date'].min()
        logger.info(f"기준 시작일: {self.date_start}")
        
        # train에 t 특성 추가
        train_df_result['t'] = ((train_df_result['date'].dt.year - self.date_start.year) * 12 +
                              (train_df_result['date'].dt.month - self.date_start.month)).astype(int)
        
        # test에도 동일한 start 기준으로 t 특성 추가
        test_df_result['t'] = ((test_df_result['date'].dt.year - self.date_start.year) * 12 +
                             (test_df_result['date'].dt.month - self.date_start.month)).astype(int)
        
        logger.info("시작 시점부터의 개월 수(t) 계산 완료")
        
        # 5. 월(month) 주기성 변환
        # 1차 푸리에 변환 (sin/cos)
        train_df_result['month_sin1'] = np.sin(2 * np.pi * train_df_result['month'] / 12)
        train_df_result['month_cos1'] = np.cos(2 * np.pi * train_df_result['month'] / 12)
        
        test_df_result['month_sin1'] = np.sin(2 * np.pi * test_df_result['month'] / 12)
        test_df_result['month_cos1'] = np.cos(2 * np.pi * test_df_result['month'] / 12)
        
        # 2~3차 푸리에 변환
        for k in [2, 3]:
            train_df_result[f'sin{k}'] = np.sin(2 * np.pi * k * train_df_result['t'] / 12)
            train_df_result[f'cos{k}'] = np.cos(2 * np.pi * k * train_df_result['t'] / 12)
            
            test_df_result[f'sin{k}'] = np.sin(2 * np.pi * k * test_df_result['t'] / 12)
            test_df_result[f'cos{k}'] = np.cos(2 * np.pi * k * test_df_result['t'] / 12)
        
        logger.info("월 주기성 변환(1~3차 푸리에) 완료")
        
        # 6. 분기(quarter) 주기성 변환
        train_df_result['quarter_sin'] = np.sin(2 * np.pi * train_df_result['quarter'] / 4)
        train_df_result['quarter_cos'] = np.cos(2 * np.pi * train_df_result['quarter'] / 4)
        
        test_df_result['quarter_sin'] = np.sin(2 * np.pi * test_df_result['quarter'] / 4)
        test_df_result['quarter_cos'] = np.cos(2 * np.pi * test_df_result['quarter'] / 4)
        
        logger.info("분기 주기성 변환 완료")
        
        # 7. 계절 변수 생성
        # 계절 매핑
        season_map = {
            12: 'winter', 1: 'winter', 2: 'winter',
            3: 'spring', 4: 'spring', 5: 'spring',
            6: 'summer', 7: 'summer', 8: 'summer',
            9: 'autumn', 10: 'autumn', 11: 'autumn'
        }
        
        train_df_result['season'] = train_df_result['month'].map(season_map).astype('category')
        test_df_result['season'] = test_df_result['month'].map(season_map).astype('category')
        
        # 계절 강도 매핑
        strength_map = {'summer': 3, 'winter': 2, 'spring': 1, 'autumn': 1}
        train_df_result['season_strength'] = train_df_result['season'].map(strength_map).astype(int)
        test_df_result['season_strength'] = test_df_result['season'].map(strength_map).astype(int)
        
        logger.info("계절 변수 및 계절 강도 변수 생성 완료")
        
        # 8. DATA_YM 변수 제거 (date 변수는 유지, 나중에 제거)
        train_df_result = train_df_result.drop(columns=['DATA_YM'])
        test_df_result = test_df_result.drop(columns=['DATA_YM'])
        
        logger.info("DATA_YM 변수 제거 완료")
        
        # 9. date 변수 제거
        train_df_result = train_df_result.drop(columns=['date'])
        test_df_result = test_df_result.drop(columns=['date'])
        
        logger.info("date 변수 제거 완료")
        logger.info("시간 관련 특성 생성 완료")
        
        return train_df_result, test_df_result
        
    def create_station_feature(self, df):
        """
        AREA_NM 변수에서 '역' 또는 '역_숫자'로 끝나는 값을 식별하여 IS_STATION 변수 생성
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
            
        Returns:
        --------
        pandas.DataFrame
            IS_STATION 변수가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        # '역' 또는 '역_숫자'로 끝나는 패턴 정의
        station_pattern = re.compile(r'역(_\d+)?$')
        
        # AREA_NM 변수가 있는 경우에만 처리
        if 'AREA_NM' in df_result.columns:
            # 새로운 변수 생성: 역 관련 지역인지 여부 (1: 역 관련, 0: 역 관련 아님)
            df_result['IS_STATION'] = df_result['AREA_NM'].apply(
                lambda x: 1 if isinstance(x, str) and station_pattern.search(x) else 0
            )
            logger.info("IS_STATION 변수 생성 완료")
            logger.info(f"역 관련 지역 수: {df_result['IS_STATION'].sum()}")
        else:
            logger.warning("AREA_NM 변수가 없어 IS_STATION 변수를 생성할 수 없습니다")
            df_result['IS_STATION'] = 0
        
        return df_result
        
    def create_eup_feature(self, df):
        """
        AREA_NM 변수에서 '읍' 또는 '읍_숫자'로 끝나는 값을 식별하여 IS_eup 변수 생성
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
            
        Returns:
        --------
        pandas.DataFrame
            IS_eup 변수가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        # '읍' 또는 '읍_숫자'로 끝나는 패턴 정의
        eup_pattern = re.compile(r'읍(_\d+)?$')
        
        # AREA_NM 변수가 있는 경우에만 처리
        if 'AREA_NM' in df_result.columns:
            # 새로운 변수 생성: 읍 관련 지역인지 여부 (1: 읍 관련, 0: 읍 관련 아님)
            df_result['IS_eup'] = df_result['AREA_NM'].apply(
                lambda x: 1 if isinstance(x, str) and eup_pattern.search(x) else 0
            )
            logger.info("IS_eup 변수 생성 완료")
            logger.info(f"읍 관련 지역 수: {df_result['IS_eup'].sum()}")
        else:
            logger.warning("AREA_NM 변수가 없어 IS_eup 변수를 생성할 수 없습니다")
            df_result['IS_eup'] = 0
        
        return df_result
        
    def identify_feature_types(self, df):
        """
        데이터프레임에서 범주형 변수와 수치형 변수를 식별
        
        Parameters:
        -----------
        df : pandas.DataFrame
            데이터프레임
            
        Returns:
        --------
        tuple
            (categorical_features, numerical_features) 형태의 튜플
        """
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
                
        logger.info(f"범주형 변수: {categorical_features}")
        logger.info(f"수치형 변수: {numerical_features}")
        
        return categorical_features, numerical_features
    
    def create_other_variables(self, df_encoded, category_group, all_encoded_cols, selected_features):
        """
        특정 범주 그룹(DIST, AREA 등)에 대한 'other' 변수 생성
        
        Parameters:
        -----------
        df_encoded : pandas.DataFrame
            원핫인코딩이 수행된 데이터프레임
        category_group : str
            범주 그룹 (예: 'DIST', 'AREA')
        all_encoded_cols : list
            해당 범주 그룹의 모든 인코딩된 컬럼 목록
        selected_features : list
            유지할 인코딩 변수 목록
            
        Returns:
        --------
        pandas.DataFrame
            'other' 변수가 추가된 데이터프레임
        """
        # 해당 그룹의 컬럼 중 선택된 변수
        group_selected = [col for col in selected_features if col.startswith(f"{category_group}_")]
        
        # 해당 그룹의 모든 컬럼
        group_all = [col for col in all_encoded_cols if col.startswith(f"{category_group}_")]
        
        # 선택되지 않은 변수들 목록
        not_selected = [col for col in group_all if col not in group_selected]
        
        # 'other' 변수 이름
        other_var_name = f"{category_group}_other"
        
        # 'other' 변수 초기화 (기본값 0)
        df_encoded[other_var_name] = 0
        
        # 선택되지 않은 변수들 중 하나라도 1인 경우 other 변수에 1 설정
        if not_selected:
            # 선택되지 않은 변수들의 합이 0보다 크면 other에 1 설정
            df_encoded.loc[df_encoded[not_selected].sum(axis=1) > 0, other_var_name] = 1
            
        logger.info(f"'{other_var_name}' 변수 생성 (전체 변수 수: {len(group_all)}, 선택된 변수 수: {len(group_selected)}, 기타로 묶인 변수 수: {len(not_selected)})")
        
        return df_encoded
    
    def selective_one_hot_encoding(self, df, categorical_columns, is_train=True):
        """
        모든 범주형 변수에 대해 원핫인코딩을 수행하고 선택된 변수만 유지
        CatBoost 모델인 경우 원핫인코딩을 건너뛰고 범주형 변수를 그대로 유지
        
        Parameters:
        -----------
        df : pandas.DataFrame
            원핫인코딩을 수행할 데이터프레임
        categorical_columns : list
            원핫인코딩을 수행할 범주형 변수 목록
        is_train : bool, optional
            학습 데이터인지 여부. 기본값은 True
            
        Returns:
        --------
        tuple
            (원핫인코딩이 적용된 데이터프레임, 유지된 인코딩 변수 목록)
        """
        # CatBoost 모델인 경우 원핫인코딩 건너뛰기
        if self.model_type == 'CatBoost':
            logger.info("CatBoost 모델 사용 중: 범주형 변수 원핫인코딩 건너뛰기")
            return df, categorical_columns
            
        df_encoded = df.copy()
        all_encoded_columns = []
        encoded_by_group = {group: [] for group in self.category_groups}
        
        # 범주형 변수의 원핫인코딩 수행
        for col in categorical_columns:
            # 변수가 'DIST_NM' 또는 'AREA_NM'인 경우 접두사 설정
            prefix = 'DIST' if col == 'DIST_NM' else 'AREA' if col == 'AREA_NM' else col
            
            # 원핫인코딩 수행
            dummies = pd.get_dummies(df[col], prefix=prefix)
            logger.info(f"{col} 변수의 원핫인코딩 수행: {dummies.columns.tolist()}")
            
            # 원본 데이터프레임에서 범주형 변수 제거
            df_encoded = df_encoded.drop(columns=[col])
            
            # 원핫인코딩된 변수 추가
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            # 인코딩된 컬럼 목록 저장
            all_encoded_columns.extend(dummies.columns.tolist())
            
            # 그룹별로 인코딩된 컬럼 분류
            for group in self.category_groups:
                if prefix == group:
                    encoded_by_group[group].extend(dummies.columns.tolist())
        
        # 선택된 변수 중 실제로 데이터프레임에 존재하는 변수만 필터링
        all_columns = df_encoded.columns.tolist()
        selected_features = [col for col in self.selected_encoded_features if col in all_columns]
        
        # 각 그룹별로 'other' 변수 생성
        for group in self.category_groups:
            # 'other' 변수 생성
            # 'other' 변수 초기화 (기본값 0)
            other_var_name = f"{group}_other"
            df_encoded[other_var_name] = 0
            
            # 'other' 변수 생성
            df_encoded = self.create_other_variables(
                df_encoded, 
                group, 
                encoded_by_group[group], 
                selected_features
            )
            
            # 'other' 변수도 선택된 변수 목록에 추가
            if other_var_name not in selected_features:
                selected_features.append(other_var_name)
        
        # 학습 데이터인 경우 인코딩된 컬럼 목록 저장
        if is_train:
            logger.info("학습 데이터의 원핫인코딩 결과 저장")
            self.train_encoded_features = selected_features
        # 테스트 데이터인 경우 학습 데이터와 같은 컬럼 구조로 맞추기
        elif self.train_encoded_features is not None:
            logger.info("테스트 데이터를 학습 데이터의 컬럼 구조에 맞춤")
            # 학습 데이터에 있지만 테스트 데이터에 없는 컬럼 추가
            for col in self.train_encoded_features:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
                    logger.info(f"테스트 데이터에 없는 컬럼 추가: {col}")
            
            # 테스트 데이터에만 있는 컬럼 제거
            test_only_cols = [col for col in df_encoded.columns if col in all_encoded_columns 
                             and col not in self.train_encoded_features]
            if test_only_cols:
                df_encoded = df_encoded.drop(columns=test_only_cols)
                logger.info(f"테스트 데이터에만 있는 컬럼 제거: {test_only_cols}")
        
        # 유지할 컬럼 목록 생성 (원본 수치형 변수 + 선택된 인코딩 변수)
        keep_columns = [col for col in df_encoded.columns if col not in all_encoded_columns or col in selected_features]
        
        # 유지할 컬럼만 남기고 나머지 삭제
        df_encoded = df_encoded[keep_columns]
        
        logger.info(f"선택된 인코딩 변수 수: {len(selected_features)}")
        logger.info(f"최종 변수 수: {len(df_encoded.columns)}")
        
        return df_encoded, selected_features
        
    def apply_log_transformation(self, df, columns):
        """
        지정된 변수들에 로그 변환 적용
        
        Parameters:
        -----------
        df : pandas.DataFrame
            변환할 데이터프레임
        columns : list
            로그 변환할 컬럼 목록
            
        Returns:
        --------
        pandas.DataFrame
            로그 변환이 적용된 데이터프레임
        """
        df_transformed = df.copy()
        for col in columns:
            if col in df.columns:
                # 음수 값이 있는지 확인
                has_negative = (df[col] < 0).any()
                
                if has_negative:
                    logger.warning(f"{col} 변수에 음수 값이 있어 변환 방식을 조정합니다.")
                    # 음수 값이 있는 경우 다른 처리 방법 적용
                    min_val = df[col].min()
                    if min_val < 0:
                        df_transformed[col] = np.log1p(df[col] - min_val + 1)
                else:
                    # 음수 값이 없는 경우 일반적인 log1p 변환
                    df_transformed[col] = np.log1p(df[col])
                
                logger.info(f"{col} 변수에 로그 변환 적용")
        
        return df_transformed
        
    def remove_missing_target(self, df, target_column='TOTAL_ELEC'):
        """
        타겟값이 결측인 행을 삭제
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        target_column : str, optional
            타겟 컬럼명. 기본값은 'TOTAL_ELEC'
            
        Returns:
        --------
        pandas.DataFrame
            타겟값 결측이 제거된 데이터프레임
        """
        initial_rows = len(df)
        df_cleaned = df.dropna(subset=[target_column])
        removed_rows = initial_rows - len(df_cleaned)
        
        if removed_rows > 0:
            logger.info(f"타겟값 결측 행 {removed_rows}개 삭제됨")
            
        return df_cleaned
        
    def remove_duplicates(self, df, keep='first'):
        """
        중복 행을 제거
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        keep : str, optional
            중복 제거 시 남겨둘 행 ('first' 또는 'last'). 기본값은 'first'
            
        Returns:
        --------
        pandas.DataFrame
            중복이 제거된 데이터프레임
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates(keep=keep)
        removed_rows = initial_rows - len(df_cleaned)
        
        if removed_rows > 0:
            logger.info(f"중복 행 {removed_rows}개 삭제됨")
            
        return df_cleaned
        
    def remove_missing_values(self, df):
        """
        데이터프레임에서 결측치가 있는 행을 제거
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
            
        Returns:
        --------
        pandas.DataFrame
            결측치가 제거된 데이터프레임
        """
        initial_rows = len(df)
        df_cleaned = df.dropna()
        removed_rows = initial_rows - len(df_cleaned)
        
        if removed_rows > 0:
            logger.info(f"결측치가 있는 행 {removed_rows}개 삭제됨")
            
        return df_cleaned
        
    def preprocess_data(self, train_df, test_df, target_col='TOTAL_ELEC'):
        """
        데이터 전처리 수행
        
        Parameters:
        -----------
        train_df : pandas.DataFrame
            학습 데이터프레임
        test_df : pandas.DataFrame
            테스트 데이터프레임
        target_col : str, optional
            타겟 변수 컬럼명. 기본값은 'TOTAL_ELEC'
            
        Returns:
        --------
        tuple
            (X_train_scaled, y_train, X_test_scaled) 형태의 튜플
        """
        logger.info("데이터 전처리 시작...")
        
        try:
            # 시간 관련 특성 생성
            if 'DATA_YM' in train_df.columns and 'DATA_YM' in test_df.columns:
                train_df, test_df = self.create_time_features(train_df, test_df)
                logger.info("시간 관련 특성 추가 완료")
            
            # 제외할 변수 목록
            exclude_columns = ['AREA_ID', 'DIST_CD', 'FAC_TRAIN']
            logger.info(f"제외할 변수: {exclude_columns}")
            
            # IS_STATION 변수 생성
            train_df = self.create_station_feature(train_df)
            test_df = self.create_station_feature(test_df)
            
            # IS_eup 변수 생성 
            train_df = self.create_eup_feature(train_df)
            test_df = self.create_eup_feature(test_df)
            
            # 모든 결측치 제거 (입력 X의 결측치도 제거)
            train_df = self.remove_missing_values(train_df)
            
            # 타겟값 결측 처리 (위에서 이미 모든 결측치를 제거했지만, 타겟만 결측인 경우를 위해 유지)
            train_df = self.remove_missing_target(train_df, target_col)
            
            # 중복 행 제거
            train_df = self.remove_duplicates(train_df)
            
            # 주요 독립변수 로그 변환
            logger.info(f"로그 변환 적용 변수: {self.log_transformed_features}")
            train_df = self.apply_log_transformation(train_df, self.log_transformed_features)
            test_df = self.apply_log_transformation(test_df, self.log_transformed_features)
            
            # 타겟 변수 로그 변환
            train_df[target_col] = np.log1p(train_df[target_col])
            
            # 제외할 변수 삭제
            train_df = train_df.drop(columns=exclude_columns, errors='ignore')
            test_df = test_df.drop(columns=exclude_columns, errors='ignore')
            
            # 범주형 변수와 수치형 변수 식별
            self.categorical_features, self.numerical_features = self.identify_feature_types(train_df)
            
            # 선택적 원핫인코딩 적용 (학습 데이터)
            train_df_encoded, selected_features = self.selective_one_hot_encoding(
                train_df, self.categorical_features, is_train=True
            )
            
            # 선택적 원핫인코딩 적용 (테스트 데이터)
            test_df_encoded, _ = self.selective_one_hot_encoding(
                test_df, self.categorical_features, is_train=False
            )
            
            # 테스트 데이터에 타겟 컬럼이 있는 경우 제거
            if target_col in test_df_encoded.columns:
                logger.info(f"테스트 데이터에서 타겟 컬럼 {target_col} 제거")
                test_df_encoded = test_df_encoded.drop(columns=[target_col])
            
            # 타겟 변수 분리
            y_train = train_df_encoded[target_col]
            X_train = train_df_encoded.drop(columns=[target_col])
            X_test = test_df_encoded.copy()
            
            # CatBoost 모델인 경우 범주형 변수 인덱스 저장
            if self.model_type == 'CatBoost':
                cat_features_idx = []
                for i, col in enumerate(X_train.columns):
                    if col in self.categorical_features:
                        cat_features_idx.append(i)
                self.catboost_cat_features = cat_features_idx
                logger.info(f"CatBoost 모델용 범주형 변수 인덱스: {cat_features_idx}")
            
            # 학습 데이터와 테스트 데이터의 컬럼 일치 확인
            missing_cols = set(X_train.columns) - set(X_test.columns)
            extra_cols = set(X_test.columns) - set(X_train.columns)
            
            if missing_cols:
                logger.warning(f"테스트 데이터에 없는 학습 데이터 컬럼: {missing_cols}")
                # 테스트 데이터에 빠진 컬럼 추가 (0으로 채움)
                for col in missing_cols:
                    X_test[col] = 0
            
            if extra_cols:
                logger.warning(f"학습 데이터에 없는 테스트 데이터 컬럼: {extra_cols}")
                # 학습 데이터에 없는 컬럼 제거
                X_test = X_test.drop(columns=list(extra_cols))
            
            # 컬럼 순서 맞추기
            X_test = X_test[X_train.columns]
            
            logger.info(f"학습 데이터 특성 수: {X_train.shape[1]}")
            logger.info(f"테스트 데이터 특성 수: {X_test.shape[1]}")
            
            # 특성 이름 저장
            self.feature_names = X_train.columns.tolist()
            
            # 범주형 변수 업데이트 (원핫인코딩 이후)
            self.categorical_features = selected_features
            
            # 수치형 변수 스케일링 적용
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            for col in self.numerical_features:
                if col in X_train.columns:
                    X_train_scaled[col] = scaler.fit_transform(X_train[[col]])
                    X_test_scaled[col] = scaler.transform(X_test[[col]])
            
            # 결과 데이터 저장
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            
            # CatBoost 모델인 경우 스케일링된 값 대신 DataFrame 반환
            if self.model_type == 'CatBoost':
                self.X_train_scaled = X_train_scaled
                self.X_test_scaled = X_test_scaled
                logger.info("CatBoost 모델 사용 중: DataFrame 형태로 데이터 유지")
            else:
                self.X_train_scaled = X_train_scaled.values
                self.X_test_scaled = X_test_scaled.values
            
            logger.info("데이터 전처리 완료")
            logger.info(f"최종 학습 데이터 특성 수: {X_train_scaled.shape[1]}")
            logger.info(f"최종 테스트 데이터 특성 수: {X_test_scaled.shape[1]}")
            
            return self.X_train_scaled, self.y_train, self.X_test_scaled
            
        except Exception as e:
            logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
            raise 