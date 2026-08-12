# Seoul Commerce

## OpenAPI 인증키 설정

서울 열린데이터광장에서 인증키를 발급받은 뒤 프로젝트 루트의 `.env`에 입력합니다.

```dotenv
SEOUL_OPEN_DATA_API_KEY=발급받은_인증키
```

`.env`는 Git에서 제외되며 실제 키를 커밋하지 않습니다. `.env.example`에는 변수명만 유지합니다.

## 한국어 컬럼 CSV 복사본 만들기

원본 CSV는 `data/raw`에 그대로 두고, 기존 영문 변환용
`config/column_mapping.yml`은 변경하지 않습니다. 한국어 복사본 전용
`config/column_mapping_ko.yml`의
`원본 컬럼명: 한국어 컬럼명` 매핑을 적용한 복사본을 `data/korean`에 생성합니다.

전체 원본 CSV를 변환합니다.

```powershell
uv run seoul-commerce-korean
```

일부 데이터셋만 변환할 수도 있습니다.

```powershell
uv run seoul-commerce-korean sales stores
```

다른 출력 위치가 필요하면 `--output-dir`을 지정합니다.

```powershell
uv run seoul-commerce-korean trade_area --output-dir data/sample_korean
```

등록되지 않은 원본 컬럼이 발견되면 잘못 해석하지 않도록 변환을 중단하고 해당
컬럼명을 출력합니다. 이때 `config/column_mapping_ko.yml`에 매핑을 먼저 추가합니다.

## 분석 마트와 EBM 모델

분석 설정은 `config/analysis_pipeline.yml`에서 관리합니다. 기본 기준분기는
`20261`, 추세 범위는 최근 8분기이며 전처리된 영문 분석 CSV를 입력으로 사용합니다.

운영 실행 순서는 전처리, 모델 학습, 마트 생성입니다.

```powershell
uv run seoul-commerce-preprocess
uv run seoul-commerce-train-ebm
uv run seoul-commerce-build-marts
```

모델 학습 중에는 후보·검증 폴드별 시작과 완료, 30초 간격 경과시간, 성능,
홀드아웃 평가와 전체 재학습 상태가 출력됩니다. 자동화 환경에서 출력을 생략하려면
`uv run seoul-commerce-train-ebm --quiet`을 사용합니다.

모델 학습 명령은 기준분기 이전의 최근 3개 분기를 롤링 검증점으로 사용해 EBM
후보 A/B/C를 비교합니다. 선택된 후보는 기준분기를 한 번만 평가한 후 전체 가용
데이터로 다시 학습됩니다. 모델은 `outputs/models`, 모델 성능·전역/지역 설명과
기술 마트는 `data/mart`에 저장됩니다.

기준분기나 출력 경로를 바꾸는 예시는 다음과 같습니다.

```powershell
uv run seoul-commerce-train-ebm --reference-quarter 20261 --model-dir outputs/models
uv run seoul-commerce-build-marts --reference-quarter 20261 --output-dir data/mart
```

`seoul-commerce-build-marts`는 모델을 다시 학습하지 않습니다. 같은 기준분기의
`ebm_evaluation_<분기>.joblib`이 있으면 저장된 예측과 설명 결과를 요약 마트에
결합하고, 없으면 기술 분석 마트와 빈 모델 결과 CSV를 생성합니다.

전체 테스트는 다음 명령으로 실행합니다.

```powershell
uv run python -m unittest discover -s tests -v
```
