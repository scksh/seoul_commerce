# Seoul Commerce

서울 열린데이터광장의 상권 데이터를 수집·전처리하고, 상권별 매출·고객·경쟁 환경과 Explainable Boosting Machine(EBM) 분석 결과를 지도 기반 Streamlit 대시보드로 제공하는 프로젝트입니다.

> 추정매출과 모델 결과는 후보 상권을 비교하기 위한 참고 정보이며, 실제 수익성이나 미래 성과를 보장하지 않습니다.

## 빠른 실행

### 요구 사항

- Python 3.12 이상
- [uv](https://docs.astral.sh/uv/)

```powershell
uv sync
uv run streamlit run app.py
```

브라우저에서 안내되는 로컬 주소로 접속합니다. 저장소에 포함된 `data/mart` 분석 마트를 사용하므로 대시보드 조회만 할 때는 서울시 API 키가 필요하지 않습니다.

AI 분석 인사이트 기능은 선택 사항입니다. 사용하려면 `.env.example`을 복사한 뒤 OpenAI API 키를 입력합니다.

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

버튼을 누를 때만 현재 선택한 상권·업종의 리포트 지표가 OpenAI Responses API(`gpt-4o-mini`)로 전송됩니다. 요청에는 `store: false`가 적용되며 결과는 현재 브라우저 세션에만 유지됩니다.

## 주요 기능

- 자치구·행정동·업종 조건별 유망 상권 탐색 및 지도 시각화
- 매출 성장, 고객 구성, 경쟁 밀도, 배후 환경 비교
- EBM 기반 예측 성능과 전역·개별 변수 기여도 제공
- 선택 상권 리포트의 AI 요약(선택 기능)

## 데이터 수집 및 분석 재실행

서울 열린데이터광장에서 인증키를 발급받아 `.env`에 설정합니다.

```dotenv
SEOUL_OPEN_DATA_API_KEY=your_seoul_open_data_api_key
```

수집 가능한 데이터셋 이름은 `trade_area`, `stores`, `sales`, `floating_population`, `resident_population`, `working_population`, `facilities`입니다. 아래처럼 데이터셋별로 수집하며, 분기형 데이터는 필요하면 `--quarter`를 지정합니다.

```powershell
uv run seoul-commerce trade_area
uv run seoul-commerce sales --quarter 20251
```

원천 데이터 전체를 준비한 뒤 분석 산출물을 순서대로 생성합니다.

```powershell
uv run seoul-commerce-preprocess
uv run seoul-commerce-train-ebm
uv run seoul-commerce-build-marts
```

- 원천 CSV: `data/raw`
- 전처리 결과: `data/analysis/english`
- 모델·평가 파일: `outputs/models`
- 대시보드 마트: `data/mart`
- 기준 분기와 모델 설정: `config/analysis_pipeline.yml`

한국어 컬럼 사본이 필요하면 다음 명령을 사용합니다.

```powershell
uv run seoul-commerce-korean
uv run seoul-commerce-korean-marts
```

테스트는 다음과 같이 실행합니다.

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m unittest discover -s tests/dashboard -v
```

## 데이터 출처

원천 데이터는 서울특별시 또는 서울신용보증재단 저작물이며, 서울신용보증재단이 제공하는 **서울시 상권분석서비스** 데이터입니다.

| 데이터 | ID | 갱신 주기 | 출처 |
| --- | --- | --- | --- |
| 영역-상권 | OA-15560 | 비정기 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15560/A/1/datasetView.do) |
| 점포-상권 | OA-15577 | 연간 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15577/A/1/datasetView.do) |
| 추정매출-상권 | OA-15572 | 분기 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15572/A/1/datasetView.do) |
| 길단위인구-상권 | OA-15568 | 월간 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15568/A/1/datasetView.do) |
| 상주인구-상권 | OA-15584 | 반기 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15584/A/1/datasetView.do) |
| 직장인구-상권 | OA-15569 | 반기 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15569/A/1/datasetView.do) |
| 집객시설-상권 | OA-15580 | 연간 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15580/A/1/datasetView.do) |

데이터별 스키마와 수집 상태는 `config/data_contracts.yml`에서 관리합니다. 원천 데이터의 갱신 시점과 제공 기준이 바뀔 수 있으므로 재수집 전 각 출처 페이지를 확인하세요.

## 라이선스

### 프로젝트 코드

프로젝트에서 작성한 소스 코드는 [MIT License](LICENSE)로 배포합니다. 주요 직접 의존성은 모두 상업적 이용과 수정·재배포가 가능한 허용형 오픈소스 라이선스입니다.

| 라이선스 | 주요 라이브러리 |
| --- | --- |
| MIT | Folium, InterpretML, Plotly, PyYAML, streamlit-folium, xmltodict |
| BSD 3-Clause/BSD | ipykernel, Joblib, Jupyter, pandas, python-dotenv, scikit-learn, seaborn, statsmodels |
| Apache 2.0 | Requests, Streamlit |
| PSF 기반 Matplotlib License | Matplotlib |
| MIT/MPL 2.0 | tqdm |

정확한 설치 버전은 `uv.lock`, 전체 직접 의존성은 `pyproject.toml`을 기준으로 합니다. 각 라이브러리 자체에는 해당 라이브러리의 라이선스가 별도로 적용됩니다.

### 데이터

원천 데이터와 이를 가공한 산출물에는 코드의 MIT License가 적용되지 않습니다. 각 출처 페이지에 표시된 **공공누리 제1유형(출처표시, 상업적 이용 및 변경 가능)** 조건을 따릅니다.

> 본 프로젝트는 서울특별시 및 서울신용보증재단이 공공누리 제1유형으로 개방한 ‘서울시 상권분석서비스’ 데이터를 이용하였으며, 해당 데이터는 [서울 열린데이터광장](https://data.seoul.go.kr/)에서 확인할 수 있습니다.

재배포 시에도 위 출처와 각 데이터셋 이름을 함께 표시해야 합니다. 자세한 조건은 [서울 열린데이터광장 저작권 안내](https://data.seoul.go.kr/etc/openInfo.do)와 개별 데이터셋 페이지를 우선 확인하세요.
