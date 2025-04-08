# 2025 한국데이터정보과학회 & SAS KOREA 데이터 분석 경진대회

## 🔍 대회 목적

- 기관에서 조사한 상권별 전기 사용량 데이터를 활용하여 각 상권의 평균 전기 총 사용량을 예측하는 모델을 개발하는 것이 목적입니다.

## 📅 대회 일정

| 일정          | 세부 내용                   |
| ----------- | ----------------------- |
| 데이터 제공      | 2025.03.28              |
| 분석 및 모델링 기간 | 2025.03.28 - 2025.04.30 |
| 예선 심사 발표    | 2025.05.09              |
| 본선 발표 평가       | 2025.05.16              |


## 📂 제공 Data 정보

### 📌 데이터 구성

- **TRAIN_DATA.csv** : 모델 학습용 데이터
- **TEST_DATA.csv** : 예측 결과 제출용 데이터

### 📌 변수 설명

| 변수명            | 설명                   | 타입        | 형태(예시)         |
| -------------- | -------------------- | --------- | -------------- |
| DATA_YM       | 데이터 작성 월             | numeric   | YYYYMM(202303) |
| AREA_ID       | 상권 코드                | numeric   | 9626           |
| AREA_NM       | 상권 명칭                | character | 중앙로역_4        |
| DIST_CD       | 상권 위치 지역 코드          | numeric   | 27110          |
| DIST_NM       | 상권 위치 지역 명칭          | character | 중구             |
| TOTAL_BIDG    | 상권 내 총 건물 수          | numeric   | 538            |
| FAC_NEIGH_1  | 상권 내 1종 생활편의시설 수     | numeric   | 266            |
| FAC_NEIGH_2  | 상권 내 2종 생활편의시설 수     | numeric   | 88             |
| FAC_CULT_MTG | 상권 내 문화 및 집회시설 수     | numeric   | 6              |
| FAC_RELG      | 상권 내 종교시설 수          | numeric   | 3              |
| FAC_RETAIL    | 상권 내 판매시설 수          | numeric   | 2              |
| FAC_MEDI      | 상권 내 의료시설 수          | numeric   | 3              |
| FAC_YOSE      | 상권 내 아동 및 노인복지시설 수   | numeric   | 0              |
| FAC_TRAIN     | 상권 내 수련원시설 수         | numeric   | 0              |
| FAC_SPORT     | 상권 내 운동시설 수          | numeric   | 0              |
| FAC_STAY      | 상권 내 숙박시설 수          | numeric   | 16             |
| ENERGY_USAGE  | 상권 내 모든 건물의 가스 총 사용량 | numeric   | 517520         |
| POWER_USAGE   | 상권 내 모든 건물의 전기 총 사용량 | numeric   | 1174.57        |

### 📌 평가지표

- **RMSE** (Root Mean Square Error) 사용

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

## 📌 TASK

1. **데이터 전처리**
   - 결측치 처리 및 이상치 탐지
   - 데이터 타입 변환 및 인코딩
   - 시계열 데이터 처리 (DATA_YM 기준)
   - 지역 코드 및 상권 코드 정규화

2. **기본 EDA (탐색적 데이터 분석)**
   - 상권별 전기 사용량 분포 분석
   - 시설 수와 전기 사용량의 상관관계 분석
   - 지역별 전기 사용량 패턴 분석
   - 시계열별 전기 사용량 추이 분석

3. **피처 엔지니어링**
   - 시설 수 관련 파생 변수 생성 (총 시설 수, 시설 밀도 등)
   - 지역 관련 파생 변수 생성 (지역 크기, 인구밀도 등)
   - 시계열 관련 파생 변수 생성 (계절성, 추세 등)
   - 상호작용 변수 생성 (시설 수와 건물 수의 비율 등)
   - 외부 데이터 추가

4. **모델 개발**
   - 기본 모델 구현 (선형 회귀, 랜덤 포레스트 등)
   - 앙상블 모델 구현 (XGBoost, LightGBM 등)
   - 딥러닝 모델 구현 (시계열 예측 모델)
   - 모델 하이퍼파라미터 튜닝

5. **모델 평가 및 검증**
   - K-Fold 교차 검증 수행
   - 시계열 기반 검증 (시간 순서 고려)
   - 지역별 성능 평가
   - RMSE 기반 모델 성능 비교

6. **최종 모델 선택 및 최적화**
   - 최적 모델 선정
   - 모델 앙상블 전략 수립
   - 최종 예측 결과 생성
   - 예측 결과 검증 및 리포팅

7. **SAS Viya 연동**
   - SAS Viya 환경 설정
   - 모델 SAS Viya 마이그레이션
   - SAS Viya 기반 예측 파이프라인 구축
   - 성능 비교 및 최적화

## 📌 분석 및 모델링 개요

(모델링 및 분석 방법에 대한 세부 내용 추가 예정)

## 📌 결과물

(결과 분석 및 성능 요약 추가 예정)

## 📁 프로젝트 구조 및 역할 분담

```
project_root/
├── data/                  # 데이터 파일 (train/test)
├── modules/               # 모델 구성 모듈
│   ├── preprocessing.py   # 전처리 모듈
│   ├── train.py           # 학습 모듈
│   ├── predict.py         # 예측 모듈
│   └── utils.py           # 공통 기능 함수
│
├── notebooks/             # EDA 및 실행전 노팁
├── results/               # 계산결과 보관 결과물
├── .gitignore            # Git 제외 파일 설정
├── pyproject.toml        # Poetry 프로젝트 설정
└── main.py                # 최종 시작 진입점
```

### 👥 역할 분담

<table width="100%">
    <tbody>
        <tr>
            <td align="center" width="20%">
                <a href="https://github.com/P-oong">
                    <img src="https://github.com/P-oong.png" width="120px;" alt="이현풍 GitHub"/>
                </a>
            </td>
            <td align="center" width="20%">
                <a href="https://github.com/2degree0">
                    <img src="https://github.com/2degree0.png" width="120px;" alt="이도영 GitHub"/>
                </a>
            </td>
            <td align="center" width="20%">
                <a href="https://github.com/dlllll0">
                    <img src="https://github.com/dlllll0.png" width="120px;" alt="이주희 GitHub"/>
                </a>
            </td>
            <td align="center" width="20%">
                <a href="https://github.com/irongmin">
                    <img src="https://github.com/irongmin.png" width="120px;" alt="정철민 GitHub"/>
                </a>
            </td>
            <td align="center" width="20%">
                <a href="https://github.com/YEOUL0520">
                    <img src="https://github.com/YEOUL0520.png" width="120px;" alt="류효정 GitHub"/>
                </a>
            </td>
        </tr>
        <tr>
            <td align="center"><b>이현풍</b></td>
            <td align="center"><b>이도영</b></td>
            <td align="center"><b>이주희</b></td>
            <td align="center"><b>정철민</b></td>
            <td align="center"><b>류효정</b></td>
        </tr>
        <tr>
            <td align="center"><a href="https://github.com/P-oong"><b>hyeonpoong-lee</b></a></td>
            <td align="center"><a href="https://github.com/2degree0"><b>doyeong-lee</b></a></td>
            <td align="center"><a href="https://github.com/dlllll0"><b>juhee-lee</b></a></td>
            <td align="center"><a href="https://github.com/irongmin"><b>chulmin-jeong</b></a></td>
            <td align="center"><a href="https://github.com/YEOUL0520"><b>hyojeong-ryu</b></a></td>
        </tr>
        <tr>
            <td align="left" valign="top">
                <ul>
                    <li>PM 및 프로젝트 구조 설계</li>
                    <li>데이터 전처리 파이프라인 구축</li>
                    <li>기본 모델 개발 및 튜닝</li>
                    <li>모델 평가 및 검증</li>
                </ul>
            </td>
            <td align="left" valign="top">
                <ul>
                    <li>EDA 및 데이터 분석</li>
                    <li>피처 엔지니어링</li>
                    <li>앙상블 모델 개발</li>
                    <li>성능 최적화</li>
                </ul>
            </td>
            <td align="left" valign="top">
                <ul>
                    <li>시계열 데이터 처리</li>
                    <li>딥러닝 모델 개발</li>
                    <li>모델 하이퍼파라미터 튜닝</li>
                    <li>결과 시각화</li>
                </ul>
            </td>
            <td align="left" valign="top">
                <ul>
                    <li>SAS Viya 연동</li>
                    <li>모델 배포 자동화</li>
                    <li>성능 모니터링</li>
                    <li>문서화 및 보고서 작성</li>
                </ul>
            </td>
            <td align="left" valign="top">
                <ul>
                    <li>외부 데이터 수집 및 통합</li>
                    <li>데이터 품질 관리</li>
                    <li>모델 검증 및 테스트</li>
                    <li>최종 결과 분석</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>
<br>
</br>

---

## 🚀 모듈 사용 가이드

### 1. 환경 설정
```bash
# Poetry 설치
curl -sSL https://install.python-poetry.org | python3 -

# 프로젝트 의존성 설치 및 가상환경 활성화
poetry install
poetry shell
```

### 2. 실행 방법
```bash
# 전체 파이프라인 실행
python main.py

# 개별 모듈 실행
python modules/preprocessing.py  # 데이터 전처리
python modules/train.py         # 모델 학습
python modules/predict.py       # 예측 실행
```

### 3. SAS Viya 연동
추후 SAS Viya 연동 방법 추가 예정

---

✨ **진행 상황에 따라 내용을 추가해 주세요.** ✨

