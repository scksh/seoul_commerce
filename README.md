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
