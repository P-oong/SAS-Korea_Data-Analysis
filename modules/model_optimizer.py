import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import logging
import time
import itertools
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

logger = logging.getLogger(__name__)

def rmse_scorer(y_true, y_pred):
    """
    로그변환된 값을 역변환 후 RMSE 스코어 계산 함수
    
    Parameters:
    -----------
    y_true : array-like
        실제 타겟값 (로그 변환된 상태)
    y_pred : array-like
        예측값 (로그 변환된 상태)
        
    Returns:
    --------
    float
        RMSE 점수
    """
    # 로그 스케일에서 원래 스케일로 역변환
    y_true_inv = np.expm1(y_true)
    y_pred_inv = np.expm1(y_pred)
    
    # 원래 스케일에서 RMSE 계산
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    
    # sklearn의 점수는 높을수록 좋으므로 음수 반환
    return -rmse

class ModelOptimizer:
    """
    그리드 서치를 통한 모델 최적화 클래스
    """
    
    def __init__(self, model_type=None, cv=5, random_state=42, n_jobs=-1, verbose=2):
        """
        ModelOptimizer 클래스 초기화
        
        Parameters:
        -----------
        model_type : str, optional
            사용할 모델 타입 (예: 'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest')
        cv : int, optional
            교차 검증 폴드 수. 기본값은 5
        random_state : int, optional
            랜덤 시드. 기본값은 42
        n_jobs : int, optional
            병렬 작업 수. 기본값은 -1 (모든 코어 사용)
        verbose : int, optional
            출력 상세도. 기본값은 2
        """
        self.model_type = model_type
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.best_model = None
        self.best_params = None
        self.grid_search = None
        self.param_grid = self._get_param_grid()
        self.base_model = self._get_base_model()
        self.cv_results_ = {}  # GridSearchCV와 유사한 결과 저장 구조
        
    def _get_base_model(self):
        """
        선택한 모델 타입에 따른 기본 모델 인스턴스 반환
        
        Returns:
        --------
        object
            선택한 모델 타입의 기본 인스턴스
        """
        if self.model_type == 'XGBoost':
            # scikit-learn 인터페이스 호환성을 위해 추가 파라미터 설정
            return XGBRegressor(
                random_state=self.random_state,
                use_label_encoder=False,  # 경고 메시지 방지
                enable_categorical=False  # 호환성 문제 해결
            )
        elif self.model_type == 'LightGBM':
            return LGBMRegressor(
                random_state=self.random_state,
                verbose=-1,  # 경고 메시지 출력 억제
                min_child_samples=20,  # 기본값 설정: 작은 리프 노드 방지
                min_split_gain=0.1,    # 기본값 설정: 최소 분할 이득
                boost_from_average=True  # 평균에서 부스팅 시작
            )
        elif self.model_type == 'CatBoost':
            return CatBoostRegressor(random_seed=self.random_state, verbose=0)
        elif self.model_type == 'RandomForest':
            return RandomForestRegressor(random_state=self.random_state)
        elif self.model_type == 'GBM':
            return GradientBoostingRegressor(random_state=self.random_state)
        elif self.model_type == 'HBM':
            return HistGradientBoostingRegressor(random_state=self.random_state)
        else:
            logger.warning(f"지원하지 않는 모델 타입: {self.model_type}. 기본값으로 RandomForest를 사용합니다.")
            return RandomForestRegressor(random_state=self.random_state)
    
    def _get_param_grid(self):
        """
        선택한 모델 타입에 따른 하이퍼파라미터 그리드 반환
        
        Returns:
        --------
        dict
            하이퍼파라미터 그리드
        """
        if self.model_type == 'XGBoost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.3]
            }
        elif self.model_type == 'LightGBM':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [20, 31, 50],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
                #'min_child_samples': [20, 50, 100],  # 작은 리프 노드 방지
                #'min_child_weight': [0.001, 0.01, 0.1],  # 너무 작은 가중치 방지
                #'min_split_gain': [0.1, 0.5, 1.0],  # 최소 분할 이득 설정
                #'reg_alpha': [0.0, 0.1, 0.5],  # L1 정규화
                #'reg_lambda': [0.0, 0.1, 0.5]   # L2 정규화
            }
        elif self.model_type == 'CatBoost':
            return {
                'iterations': [50, 100, 200],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2],
                'l2_leaf_reg': [1, 3, 5],
                'subsample': [0.6, 0.8, 1.0],           # 데이터 샘플링 비율 추가
                'rsm': [0.6, 0.8, 1.0]                 # 특성 샘플링 비율 추가 (colsample_bytree와 유사)
            }
        elif self.model_type == 'RandomForest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],        # None 추가 - 제한 없는 트리 깊이
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.7],  # 각 분할에서 고려할 특성 수
                'bootstrap': [True, False]             # 부트스트랩 샘플링 여부
            }
        elif self.model_type == 'GBM':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.7]   # 각 분할에서 고려할 특성 수 추가
            }
        elif self.model_type == 'HBM':
            return {
                'max_iter': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_leaf': [1, 5, 10],
                'l2_regularization': [0, 0.1, 0.5],    # L2 정규화 강도 추가
                'max_bins': [128, 255, 512]            # 히스토그램 빈의 최대 수 추가
            }
        else:
            logger.warning(f"지원하지 않는 모델 타입: {self.model_type}. 기본 파라미터 그리드를 사용합니다.")
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15]
            }
            
    def _manual_grid_search(self, X_train, y_train, fit_params=None):
        """
        GridSearchCV 대신 수동으로 그리드 서치를 구현
        
        Parameters:
        -----------
        X_train : array-like
            학습 데이터 특성
        y_train : array-like
            학습 데이터 타겟
        fit_params : dict, optional
            모델 학습에 전달할 추가 파라미터
            
        Returns:
        --------
        tuple
            (최적 모델, 최적 파라미터)
        """
        if fit_params is None:
            fit_params = {}
            
        # 파라미터 조합 생성
        param_combinations = []
        param_names = list(self.param_grid.keys())
        param_values = list(itertools.product(*[self.param_grid[param] for param in param_names]))
        
        for values in param_values:
            param_dict = {param_names[i]: values[i] for i in range(len(param_names))}
            param_combinations.append(param_dict)
        
        logger.info(f"총 {len(param_combinations)}개의 파라미터 조합에 대해 그리드 서치 수행")
        
        best_score = float('-inf')
        best_params = None
        best_model = None
        
        # 교차 검증 설정
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # 결과 저장용 리스트
        mean_train_scores = []
        mean_test_scores = []
        all_params = []
        
        # 각 파라미터 조합에 대해 교차 검증 수행
        for i, params in enumerate(param_combinations):
            if self.verbose >= 1:
                logger.info(f"파라미터 조합 {i+1}/{len(param_combinations)}: {params}")
            
            # 모델 인스턴스 생성 및 파라미터 설정
            model = self._get_base_model()
            model.set_params(**params)
            
            train_scores = []
            test_scores = []
            
            # K-폴드 교차 검증
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
                if isinstance(X_train, np.ndarray):
                    X_fold_train = X_train[train_idx]
                    X_fold_val = X_train[val_idx]
                else:  # DataFrame인 경우
                    X_fold_train = X_train.iloc[train_idx]
                    X_fold_val = X_train.iloc[val_idx]
                
                y_fold_train = y_train.iloc[train_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # 모델 학습
                model.fit(X_fold_train, y_fold_train, **fit_params)
                
                # 훈련 세트 평가
                y_train_pred = model.predict(X_fold_train)
                # 로그 스케일에서 원래 스케일로 역변환 후 RMSE 계산
                y_fold_train_inv = np.expm1(y_fold_train)
                y_train_pred_inv = np.expm1(y_train_pred)
                train_rmse = np.sqrt(mean_squared_error(y_fold_train_inv, y_train_pred_inv))
                train_scores.append(train_rmse)
                
                # 검증 세트 평가
                y_val_pred = model.predict(X_fold_val)
                # 로그 스케일에서 원래 스케일로 역변환 후 RMSE 계산
                y_fold_val_inv = np.expm1(y_fold_val)
                y_val_pred_inv = np.expm1(y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_fold_val_inv, y_val_pred_inv))
                test_scores.append(val_rmse)
                
                if self.verbose >= 2:
                    logger.info(f"  Fold {fold} - 훈련 RMSE: {train_rmse:.4f}, 검증 RMSE: {val_rmse:.4f}")
            
            # 평균 RMSE 계산
            mean_train_rmse = np.mean(train_scores)
            mean_test_rmse = np.mean(test_scores)
            
            # 표준편차 계산
            std_train_rmse = np.std(train_scores)
            std_test_rmse = np.std(test_scores)
            
            # 결과 저장
            mean_train_scores.append(-mean_train_rmse)  # 음수로 저장 (높을수록 좋은 점수 형태로)
            mean_test_scores.append(-mean_test_rmse)    # 음수로 저장
            all_params.append(params)
            
            # 표준편차도 저장
            if 'std_train_score' not in self.cv_results_:
                self.cv_results_['std_train_score'] = []
                self.cv_results_['std_test_score'] = []
            
            self.cv_results_['std_train_score'].append(std_train_rmse)
            self.cv_results_['std_test_score'].append(std_test_rmse)
            
            if self.verbose >= 1:
                logger.info(f"  평균 훈련 RMSE: {mean_train_rmse:.4f} (±{std_train_rmse:.4f}), 평균 검증 RMSE: {mean_test_rmse:.4f} (±{std_test_rmse:.4f})")
            
            # 최고 점수 갱신
            if -mean_test_rmse > best_score:
                best_score = -mean_test_rmse
                best_params = params
                
                # 최고 성능 모델 재학습 (전체 데이터로)
                best_model = self._get_base_model()
                best_model.set_params(**params)
                best_model.fit(X_train, y_train, **fit_params)
        
        # 결과 저장 (GridSearchCV와 유사한 형태로)
        self.cv_results_ = {
            'mean_train_score': np.array(mean_train_scores),
            'mean_test_score': np.array(mean_test_scores),
            'params': all_params
        }
        self.best_score_ = best_score
        self.best_index_ = np.argmax(mean_test_scores)
        self.best_params_ = best_params
        
        return best_model, best_params
    
    def optimize(self, X_train, y_train, cat_features=None):
        """
        그리드 서치를 통한 모델 최적화 수행
        
        Parameters:
        -----------
        X_train : array-like
            학습 데이터 특성
        y_train : array-like
            학습 데이터 타겟
        cat_features : list, optional
            CatBoost 모델 사용 시 범주형 특성의 인덱스 리스트
            
        Returns:
        --------
        object
            최적화된 모델 객체
        """
        logger.info(f"{self.model_type} 모델 최적화 시작...")
        start_time = time.time()
        
        # 특별한 경우 처리 (CatBoost)
        fit_params = {}
        if self.model_type == 'CatBoost' and cat_features is not None:
            fit_params = {'cat_features': cat_features}
        
        try:
            # XGBoost 모델은 호환성 문제로 수동 그리드 서치 사용
            if self.model_type == 'XGBoost':
                logger.info(f"XGBoost 모델에 대해 수동 그리드 서치 수행 (호환성 문제 해결)")
                logger.info(f"총 {self.cv} 폴드로 교차검증 수행 중...")
                self.best_model, self.best_params = self._manual_grid_search(X_train, y_train, fit_params)
            else:
                # 일반 모델은 GridSearchCV 사용
                # RMSE 스코어 생성
                rmse = make_scorer(rmse_scorer)
                
                # 그리드 서치 설정
                self.grid_search = GridSearchCV(
                    estimator=self.base_model,
                    param_grid=self.param_grid,
                    scoring=rmse,
                    cv=self.cv,
                    return_train_score=True,
                    verbose=self.verbose,
                    n_jobs=self.n_jobs,
                    refit=True
                )
                
                # 그리드 서치 수행
                logger.info(f"총 {self.cv} 폴드로 교차검증 수행 중...")
                self.grid_search.fit(X_train, y_train, **fit_params)
                
                # 최적 모델 및 파라미터 저장
                self.best_model = self.grid_search.best_estimator_
                self.best_params = self.grid_search.best_params_
                self.cv_results_ = self.grid_search.cv_results_
                self.best_score_ = self.grid_search.best_score_
                self.best_index_ = self.grid_search.best_index_
                
                # 표준편차 정보가 없으면 변환하여 저장
                if 'std_train_score' not in self.cv_results_:
                    # GridSearchCV는 표준편차 정보를 다른 이름으로 저장하므로 변환
                    if 'std_train_score' in self.grid_search.cv_results_:
                        self.cv_results_['std_train_score'] = self.grid_search.cv_results_['std_train_score']
                    if 'std_test_score' in self.grid_search.cv_results_:
                        self.cv_results_['std_test_score'] = self.grid_search.cv_results_['std_test_score']
            
            # 결과 출력
            elapsed_time = time.time() - start_time
            logger.info(f"모델 최적화 완료 (소요 시간: {elapsed_time:.2f}초)")
            logger.info(f"최적 파라미터: {self.best_params}")
            logger.info(f"최적 RMSE 점수 (원래 스케일): {-self.best_score_:.4f}")
            
            # 과적합 모니터링
            self._check_overfitting()
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"모델 최적화 중 오류 발생: {str(e)}")
            raise
    
    def _check_overfitting(self):
        """
        학습 데이터와 검증 데이터의 점수를 비교하여 과적합 여부 확인
        """
        if not hasattr(self, 'cv_results_'):
            logger.warning("그리드 서치 결과가 아직 없습니다.")
            return
        
        # 결과 추출
        results = self.cv_results_
        
        # 최적 모델의 학습/검증 점수
        if self.model_type == 'XGBoost':
            train_score = -results['mean_train_score'][self.best_index_]
            test_score = -results['mean_test_score'][self.best_index_]
        else:
            train_score = -results['mean_train_score'][self.best_index_]
            test_score = -results['mean_test_score'][self.best_index_]
        
        # 점수 차이 계산
        score_diff = train_score - test_score
        score_ratio = train_score / test_score if test_score > 0 else float('inf')
        
        logger.info(f"최적 모델의 학습 RMSE (원래 스케일): {train_score:.4f}")
        logger.info(f"최적 모델의 검증 RMSE (원래 스케일): {test_score:.4f}")
        logger.info(f"RMSE 차이: {score_diff:.4f}")
        logger.info(f"RMSE 비율: {score_ratio:.4f}")
        
        # 과적합 진단
        if score_ratio > 1.3:  # 30% 이상 차이나면 주의
            logger.warning("주의: 학습 데이터와 검증 데이터의 성능 차이가 큽니다. 과적합 가능성이 있습니다.")
        else:
            logger.info("과적합 없이 모델이 잘 학습되었습니다.")
    
    def get_best_model(self):
        """
        최적화된 최고 모델 반환
        
        Returns:
        --------
        object
            최적화된 모델 객체
        """
        if self.best_model is None:
            logger.warning("아직 최적화된 모델이 없습니다. optimize 메서드를 먼저 호출하세요.")
        return self.best_model
    
    def get_best_params(self):
        """
        최적 하이퍼파라미터 반환
        
        Returns:
        --------
        dict
            최적 하이퍼파라미터
        """
        if self.best_params is None:
            logger.warning("아직 최적화된 파라미터가 없습니다. optimize 메서드를 먼저 호출하세요.")
        return self.best_params
    
    def get_feature_importance(self, feature_names=None):
        """
        최적 모델의 특성 중요도 반환
        
        Parameters:
        -----------
        feature_names : list, optional
            특성 이름 목록
            
        Returns:
        --------
        pandas.DataFrame
            특성 중요도 데이터프레임
        """
        if self.best_model is None:
            logger.warning("아직 최적화된 모델이 없습니다. optimize 메서드를 먼저 호출하세요.")
            return None
        
        # 모델 타입에 따라 특성 중요도 추출 방식 다름
        importances = None
        
        try:
            if self.model_type in ['XGBoost', 'LightGBM', 'RandomForest', 'GBM']:
                importances = self.best_model.feature_importances_
            elif self.model_type == 'CatBoost':
                importances = self.best_model.feature_importances_
            
            if importances is not None and feature_names is not None:
                # 데이터프레임으로 변환
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # 중요도에 따라 정렬
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                return importance_df
            else:
                logger.warning("특성 중요도를 추출할 수 없습니다.")
                return None
        except Exception as e:
            logger.error(f"특성 중요도 추출 중 오류 발생: {str(e)}")
            return None
    
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
            로그 변환된 예측값 (원래 스케일로 변환하려면 inverse_transform_predictions 함수 사용)
        """
        if self.best_model is None:
            logger.warning("아직 최적화된 모델이 없습니다. optimize 메서드를 먼저 호출하세요.")
            return None
        
        try:
            predictions = self.best_model.predict(X_test)
            logger.info("예측 완료 (로그 스케일)")
            return predictions
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            raise
            
    def print_optimization_results(self):
        """
        모델 최적화 결과를 상세하게 출력하는 메서드
        """
        if self.best_model is None or self.best_params is None:
            logger.warning("아직 최적화된 모델이 없습니다. optimize 메서드를 먼저 호출하세요.")
            return
        
        # 최적 파라미터 및 성능 점수 출력
        logger.info("=" * 50)
        logger.info("모델 최적화 상세 결과")
        logger.info("=" * 50)
        logger.info(f"모델 유형: {self.model_type}")
        logger.info(f"최적 하이퍼파라미터: {self.best_params}")
        logger.info(f"최적 RMSE 점수 (원래 스케일): {-self.best_score_:.4f}")
        
        # 과적합 여부 확인
        if hasattr(self, 'cv_results_'):
            train_score = -self.cv_results_['mean_train_score'][self.best_index_]
            test_score = -self.cv_results_['mean_test_score'][self.best_index_]
            
            # 표준편차 정보 출력 (있는 경우)
            if 'std_train_score' in self.cv_results_ and 'std_test_score' in self.cv_results_:
                train_std = self.cv_results_['std_train_score'][self.best_index_]
                test_std = self.cv_results_['std_test_score'][self.best_index_]
                logger.info(f"학습 데이터 RMSE (원래 스케일): {train_score:.4f} (±{train_std:.4f})")
                logger.info(f"검증 데이터 RMSE (원래 스케일): {test_score:.4f} (±{test_std:.4f})")
            else:
                logger.info(f"학습 데이터 RMSE (원래 스케일): {train_score:.4f}")
                logger.info(f"검증 데이터 RMSE (원래 스케일): {test_score:.4f}")
            
            score_diff = train_score - test_score
            score_ratio = train_score / test_score if test_score > 0 else float('inf')
            
            logger.info(f"RMSE 차이: {score_diff:.4f}")
            logger.info(f"RMSE 비율: {score_ratio:.4f}")
            
            if score_ratio > 1.3:
                logger.warning("주의: 과적합 가능성이 있습니다 (학습/검증 비율 > 1.3)")
            else:
                logger.info("과적합 없이 모델이 잘 학습되었습니다.")
        
        # 상위 5개 파라미터 조합 성능 출력 (있는 경우)
        if hasattr(self, 'cv_results_') and 'mean_test_score' in self.cv_results_:
            logger.info("\n상위 5개 파라미터 조합 성능:")
            
            # 점수에 따라 정렬된 인덱스 (음수이므로 오름차순)
            sorted_indices = np.argsort(self.cv_results_['mean_test_score'])[::-1][:5]
            
            for i, idx in enumerate(sorted_indices):
                params = self.cv_results_['params'][idx]
                test_score = -self.cv_results_['mean_test_score'][idx]
                
                if 'mean_train_score' in self.cv_results_:
                    train_score = -self.cv_results_['mean_train_score'][idx]
                    
                    # 표준편차 출력 (있는 경우)
                    if 'std_train_score' in self.cv_results_ and 'std_test_score' in self.cv_results_:
                        train_std = self.cv_results_['std_train_score'][idx]
                        test_std = self.cv_results_['std_test_score'][idx]
                        logger.info(f"{i+1}. RMSE (원래 스케일): {test_score:.4f} (±{test_std:.4f}) (학습: {train_score:.4f} ±{train_std:.4f}) - 파라미터: {params}")
                    else:
                        logger.info(f"{i+1}. RMSE (원래 스케일): {test_score:.4f} (학습: {train_score:.4f}) - 파라미터: {params}")
                else:
                    if 'std_test_score' in self.cv_results_:
                        test_std = self.cv_results_['std_test_score'][idx]
                        logger.info(f"{i+1}. RMSE (원래 스케일): {test_score:.4f} (±{test_std:.4f}) - 파라미터: {params}")
                    else:
                        logger.info(f"{i+1}. RMSE (원래 스케일): {test_score:.4f} - 파라미터: {params}")
        
        logger.info("=" * 50) 