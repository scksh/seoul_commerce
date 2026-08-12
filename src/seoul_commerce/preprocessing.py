"""서울시 상권 분석 데이터 전처리 모듈.

Author:
    SeungHwan Kim

Created:
    2026-08-12

Description:
    서울 상권 원본 7종을 결합하고 파생값을 계산하여
    영문 분석 CSV와 한국어 참고 CSV로 저장한다.
"""

import argparse
import math
import os
import tempfile

import pandas as pd
import yaml


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "preprocessing.yml")


def _load_yaml(path: str) -> dict:
    """YAML 파일을 읽고 최상위 구조만 검증한다."""
    config_path = os.path.abspath(path)
    try:
        with open(config_path, encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise ValueError(f"설정 파일을 찾을 수 없습니다: {config_path}") from error
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"설정 파일을 읽을 수 없습니다: {config_path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"설정 파일의 최상위 항목은 딕셔너리여야 합니다: {config_path}")
    return data


def load_preprocessing_config(path: str = DEFAULT_CONFIG_FILE) -> dict:
    """
    전처리 범위, 원본 컬럼 매핑, 출력 파일 설정을 불러온다.

    Args:
        path: preprocessing.yml 파일 경로

    Returns:
        검증을 마친 전처리 설정

    Raises:
        ValueError: 필수 설정이 없거나 컬럼 매핑이 올바르지 않은 경우
    """
    config = _load_yaml(path)
    required_sections = {"analysis_scope", "paths", "source_datasets", "outputs", "segments"}
    missing = sorted(required_sections.difference(config))
    if missing:
        raise ValueError(f"전처리 설정에 필수 섹션이 없습니다: {', '.join(missing)}")

    common = config.get("common_source_columns", {})
    if not isinstance(common, dict):
        raise ValueError("common_source_columns는 딕셔너리여야 합니다.")
    for name, dataset in config["source_datasets"].items():
        if not isinstance(dataset, dict) or not {"filename", "primary_key", "columns"}.issubset(dataset):
            raise ValueError(f"원본 데이터셋 설정이 올바르지 않습니다: {name}")
        mapping = {**common, **dataset["columns"]} if dataset.get("use_common_columns") else dataset["columns"]
        if len(mapping.values()) != len(set(mapping.values())):
            raise ValueError(f"{name} 원본 컬럼 매핑 결과에 중복이 있습니다.")
        if not set(dataset["primary_key"]).issubset(mapping.values()):
            raise ValueError(f"{name} 기본키가 컬럼 매핑 결과에 없습니다.")
    return config


def load_analysis_schema(path: str) -> dict:
    """
    최종 파일별 기본키, 컬럼 순서, dtype 계약을 불러온다.

    Args:
        path: analysis_schema.yml 파일 경로

    Returns:
        데이터셋별 분석 스키마

    Raises:
        ValueError: 컬럼과 dtype 정의가 일치하지 않는 경우
    """
    config = _load_yaml(path)
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("분석 스키마에 datasets 설정이 없습니다.")
    allowed_dtypes = {"string", "category", "Int64", "Float64", "boolean", "datetime64[ns]"}
    for name, schema in datasets.items():
        if not isinstance(schema, dict) or not {"primary_key", "columns", "dtypes"}.issubset(schema):
            raise ValueError(f"분석 스키마가 올바르지 않습니다: {name}")
        columns = schema["columns"]
        assigned = [column for dtype, values in schema["dtypes"].items() for column in values]
        unknown_dtypes = set(schema["dtypes"]).difference(allowed_dtypes)
        if unknown_dtypes:
            raise ValueError(f"{name}에 지원하지 않는 dtype이 있습니다: {', '.join(sorted(unknown_dtypes))}")
        if len(columns) != len(set(columns)) or len(assigned) != len(set(assigned)):
            raise ValueError(f"{name} 스키마에 중복 컬럼이 있습니다.")
        if set(columns) != set(assigned):
            raise ValueError(f"{name}의 columns와 dtypes 컬럼이 일치하지 않습니다.")
        if not set(schema["primary_key"]).issubset(columns):
            raise ValueError(f"{name} 기본키가 columns에 없습니다.")
    return datasets


def load_analysis_column_mapping(path: str, required_columns: list[str]) -> dict[str, str]:
    """
    영문에서 한국어로 바꿀 분석 컬럼 매핑을 불러온다.

    Args:
        path: analysis_column_mapping_ko.yml 파일 경로
        required_columns: 최종 분석 스키마에 필요한 영문 컬럼

    Returns:
        영문 컬럼명과 한국어 컬럼명의 매핑

    Raises:
        ValueError: 필수 컬럼 매핑이 없거나 한국어 컬럼명이 중복된 경우
    """
    config = _load_yaml(path)
    mapping = config.get("columns")
    if not isinstance(mapping, dict):
        raise ValueError("한국어 분석 매핑에 columns 설정이 없습니다.")
    missing = sorted(set(required_columns).difference(mapping))
    if missing:
        raise ValueError(f"한국어 분석 매핑에 없는 컬럼입니다: {', '.join(missing)}")
    targets = [mapping[column] for column in required_columns]
    if len(targets) != len(set(targets)):
        raise ValueError("한국어 분석 컬럼명에 중복이 있습니다.")
    return {str(source): str(target) for source, target in mapping.items()}


def _project_path(relative_path: str) -> str:
    return os.path.abspath(os.path.join(PROJECT_ROOT, relative_path))


PREPROCESSING_CONFIG = load_preprocessing_config()
SCHEMA_PATH = _project_path(PREPROCESSING_CONFIG["paths"]["analysis_schema"])
MAPPING_PATH = _project_path(PREPROCESSING_CONFIG["paths"]["analysis_column_mapping_ko"])
ANALYSIS_SCHEMAS = load_analysis_schema(SCHEMA_PATH)
ALL_ANALYSIS_COLUMNS = list(dict.fromkeys(
    column for schema in ANALYSIS_SCHEMAS.values() for column in schema["columns"]
))
ANALYSIS_COLUMN_MAPPING_KO = load_analysis_column_mapping(MAPPING_PATH, ALL_ANALYSIS_COLUMNS)

DEFAULT_INPUT_DIR = _project_path(PREPROCESSING_CONFIG["paths"]["input_dir"])
DEFAULT_ENGLISH_OUTPUT_DIR = _project_path(PREPROCESSING_CONFIG["paths"]["english_output_dir"])
DEFAULT_KOREAN_OUTPUT_DIR = _project_path(PREPROCESSING_CONFIG["paths"]["korean_output_dir"])
ANALYSIS_QUARTER_MIN = str(PREPROCESSING_CONFIG["analysis_scope"]["quarter_min"])
ANALYSIS_QUARTER_MAX = str(PREPROCESSING_CONFIG["analysis_scope"]["quarter_max"])
SELECTED_INDUSTRY_CODES = tuple(map(str, PREPROCESSING_CONFIG["analysis_scope"]["industry_codes"]))
SOURCE_DATASETS = PREPROCESSING_CONFIG["source_datasets"]
SOURCE_DATASET_NAMES = tuple(SOURCE_DATASETS)
OUTPUTS = PREPROCESSING_CONFIG["outputs"]
SEGMENTS = PREPROCESSING_CONFIG["segments"]
CANONICAL_TRADE_AREA_NAMES = {
    str(code): str(name)
    for code, name in PREPROCESSING_CONFIG.get("canonical_trade_area_names", {}).items()
}


def _source_mapping(dataset_name: str) -> dict[str, str]:
    dataset = SOURCE_DATASETS[dataset_name]
    if not dataset.get("use_common_columns"):
        return dict(dataset["columns"])
    return {**PREPROCESSING_CONFIG.get("common_source_columns", {}), **dataset["columns"]}


def _require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name}에 필수 컬럼이 없습니다: {', '.join(missing)}")


def _ensure_unique(frame: pd.DataFrame, keys: list[str] | tuple[str, ...], name: str) -> None:
    _require_columns(frame, keys, name)
    missing = int(frame[list(keys)].isna().any(axis=1).sum())
    duplicates = int(frame.duplicated(list(keys), keep=False).sum())
    if missing or duplicates:
        raise ValueError(f"{name} 기본키 오류: 결측 {missing:,}행, 중복 관련 {duplicates:,}행")


def _quarter_ordinal(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    valid = text.str.fullmatch(r"\d{4}[1-4]").fillna(False)
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    year = pd.to_numeric(text.str[:4], errors="coerce")
    quarter = pd.to_numeric(text.str[4:5], errors="coerce")
    result.loc[valid] = (year.loc[valid] * 4 + quarter.loc[valid] - 1).astype("Int64")
    return result


def _parse_source_dates(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip()
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    quarter_mask = text.str.fullmatch(r"\d{4}[1-4]").fillna(False)
    if quarter_mask.any():
        year = pd.to_numeric(text.loc[quarter_mask].str[:4]).astype(int)
        month = pd.to_numeric(text.loc[quarter_mask].str[4]).astype(int) * 3
        dates = pd.to_datetime({"year": year, "month": month, "day": 1}) + pd.offsets.MonthEnd(0)
        result.loc[quarter_mask] = dates.to_numpy()
    other_mask = text.notna() & ~quarter_mask
    result.loc[other_mask] = pd.to_datetime(text.loc[other_mask], errors="coerce").to_numpy()
    return result


def _safe_divide(numerator: pd.Series, denominator: pd.Series | int | float) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").astype("Float64")
    if not isinstance(denominator, pd.Series):
        denominator = pd.Series(denominator, index=numerator.index)
    denominator = pd.to_numeric(denominator, errors="coerce").astype("Float64")
    valid = numerator.notna() & denominator.notna() & denominator.ne(0)
    result = pd.Series(pd.NA, index=numerator.index, dtype="Float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _apply_schema_dtypes(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for dtype, columns in ANALYSIS_SCHEMAS[dataset_name]["dtypes"].items():
        for column in columns:
            if dtype == "datetime64[ns]":
                frame[column] = _parse_source_dates(frame[column])
            elif dtype in {"Int64", "Float64"}:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(dtype)
            else:
                frame[column] = frame[column].astype(dtype)
    return frame


def optimize_dtypes(frame: pd.DataFrame, dataset_name: str | None = None) -> pd.DataFrame:
    """
    DataFrame에 손실 없는 nullable dtype을 적용한다.

    Args:
        frame: dtype을 정리할 DataFrame
        dataset_name: 분석 스키마 또는 원본 데이터셋 이름

    Returns:
        dtype을 적용한 새 DataFrame

    Raises:
        ValueError: 원본 숫자 컬럼에 변환할 수 없는 값이 있는 경우
    """
    result = frame.copy()
    if dataset_name in ANALYSIS_SCHEMAS:
        return _apply_schema_dtypes(result, dataset_name)

    for column in result.columns:
        if column.endswith("_code"):
            result[column] = result[column].astype("string")
        elif column.endswith("_name"):
            result[column] = result[column].astype("string")
        elif column == "source_base_date":
            result[column] = _parse_source_dates(result[column])
        else:
            numeric = pd.to_numeric(result[column], errors="coerce")
            if result[column].notna().sum() != numeric.notna().sum():
                raise ValueError(f"{dataset_name or '원본'}의 {column} 숫자 변환에 실패했습니다.")
            non_missing = numeric.dropna()
            result[column] = (
                numeric.astype("Int64")
                if non_missing.empty or non_missing.mod(1).eq(0).all()
                else numeric.astype("Float64")
            )
    return result


def _finish_dataset(frame: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    columns = ANALYSIS_SCHEMAS[dataset_name]["columns"]
    keys = ANALYSIS_SCHEMAS[dataset_name]["primary_key"]
    _require_columns(frame, columns, dataset_name)
    result = frame[columns].sort_values(keys).reset_index(drop=True)
    result = _apply_schema_dtypes(result, dataset_name)
    _ensure_unique(result, keys, dataset_name)
    return result


def _read_source_csv(
    dataset_name: str,
    input_dir: str,
    industry_codes: list[str] | tuple[str, ...],
    chunksize: int,
) -> pd.DataFrame:
    dataset = SOURCE_DATASETS[dataset_name]
    mapping = _source_mapping(dataset_name)
    path = os.path.join(input_dir, dataset["filename"])
    if not os.path.isfile(path):
        raise ValueError(f"원본 CSV를 찾을 수 없습니다: {path}")

    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns
    missing = [column for column in mapping if column not in header]
    if missing:
        raise ValueError(f"{dataset_name} 원본에 필수 컬럼이 없습니다: {', '.join(missing)}")
    code_columns = [
        source for source, target in mapping.items()
        if target.endswith("_code") or target == "source_base_date"
    ]
    options = {
        "encoding": "utf-8-sig",
        "usecols": list(mapping),
        "dtype": {column: "string" for column in code_columns},
    }

    if dataset.get("industry_filtered"):
        industry_source = next(source for source, target in mapping.items() if target == "industry_code")
        parts = []
        for chunk in pd.read_csv(path, chunksize=chunksize, **options):
            parts.append(chunk.loc[chunk[industry_source].isin(industry_codes)])
        frame = pd.concat(parts, ignore_index=True)
    else:
        frame = pd.read_csv(path, low_memory=False, **options)

    frame = frame.rename(columns=mapping)
    if "quarter_code" in frame:
        frame = frame.loc[frame["quarter_code"].between(ANALYSIS_QUARTER_MIN, ANALYSIS_QUARTER_MAX)].copy()
    frame = optimize_dtypes(frame, dataset_name)
    _ensure_unique(frame, dataset["primary_key"], dataset_name)
    return frame


def load_source_data(
    input_dir: str = DEFAULT_INPUT_DIR,
    industry_codes: list[str] | tuple[str, ...] = SELECTED_INDUSTRY_CODES,
    chunksize: int = 200_000,
) -> dict[str, pd.DataFrame]:
    """
    원본 CSV 7종을 코드 dtype을 보존하여 불러온다.

    Args:
        input_dir: 원본 CSV 디렉터리
        industry_codes: 분석할 서비스 업종 코드
        chunksize: 점포와 매출 CSV를 나누어 읽을 행 수

    Returns:
        데이터셋 이름과 원본 DataFrame의 딕셔너리

    Raises:
        ValueError: 파일·필수 컬럼·기본키에 오류가 있는 경우
    """
    if chunksize < 1:
        raise ValueError("chunksize는 1 이상이어야 합니다.")
    if not industry_codes:
        raise ValueError("분석할 업종 코드가 없습니다.")
    source_dir = os.path.abspath(input_dir)
    return {
        name: _read_source_csv(name, source_dir, tuple(industry_codes), chunksize)
        for name in SOURCE_DATASET_NAMES
    }


def _inverse_epsg5181(x, y) -> tuple[float | None, float | None]:
    """EPSG:5181(GRS80 TM) 좌표 하나를 WGS84 경위도로 변환한다."""
    if pd.isna(x) or pd.isna(y):
        return None, None
    a = 6_378_137.0
    f = 1 / 298.257222101
    e2 = 2 * f - f**2
    ep2 = e2 / (1 - e2)
    lat0, lon0 = math.radians(38), math.radians(127)

    def meridional_arc(latitude: float) -> float:
        return a * (
            (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * latitude
            - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * latitude)
            + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * latitude)
            - 35 * e2**3 / 3072 * math.sin(6 * latitude)
        )

    m = meridional_arc(lat0) + float(y) - 500_000
    mu = m / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))
    fp = (
        mu + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + 151 * e1**3 / 96 * math.sin(6 * mu)
        + 1097 * e1**4 / 512 * math.sin(8 * mu)
    )
    sin_fp, cos_fp, tan_fp = math.sin(fp), math.cos(fp), math.tan(fp)
    n1 = a / math.sqrt(1 - e2 * sin_fp**2)
    r1 = a * (1 - e2) / (1 - e2 * sin_fp**2) ** 1.5
    t1, c1 = tan_fp**2, ep2 * cos_fp**2
    d = (float(x) - 200_000) / n1
    latitude = fp - n1 * tan_fp / r1 * (
        d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * ep2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * ep2 - 3 * c1**2) * d**6 / 720
    )
    longitude = lon0 + (
        d - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * ep2 + 24 * t1**2) * d**5 / 120
    ) / cos_fp
    return math.degrees(longitude), math.degrees(latitude)


def build_trade_area_master(trade_area: pd.DataFrame) -> pd.DataFrame:
    """
    상권별 기준정보, 면적, WGS84 좌표를 생성한다.

    Args:
        trade_area: 상권 영역 원본 DataFrame

    Returns:
        상권코드가 기본키인 상권 기준정보 DataFrame
    """
    _ensure_unique(trade_area, ["trade_area_code"], "trade_area")
    result = trade_area.copy()
    original_names = result["trade_area_name"].astype("string")
    corrected = result["trade_area_code"].map(CANONICAL_TRADE_AREA_NAMES)
    result["trade_area_name"] = corrected.fillna(original_names)
    result["name_changed_flag"] = corrected.notna() & result["trade_area_name"].ne(original_names)
    result["area_ha"] = _safe_divide(result["area_sqm"], 10_000)
    coordinates = [_inverse_epsg5181(x, y) for x, y in zip(result["x_epsg5181"], result["y_epsg5181"])]
    result["longitude"] = [coordinate[0] for coordinate in coordinates]
    result["latitude"] = [coordinate[1] for coordinate in coordinates]
    return _finish_dataset(result, "trade_area_master")


def _repeated_flags(frame: pd.DataFrame, value_columns: list[str], flag_name: str) -> pd.DataFrame:
    keys = ["quarter_code", "trade_area_code"]
    result = frame[[*keys, *value_columns]].copy()
    result["_ordinal"] = _quarter_ordinal(result["quarter_code"])
    result = result.sort_values(["trade_area_code", "_ordinal"])
    previous = result.groupby("trade_area_code", observed=True)[list(value_columns)].shift()
    previous_ordinal = result.groupby("trade_area_code", observed=True)["_ordinal"].shift()
    same = (result[list(value_columns)].eq(previous) | (result[list(value_columns)].isna() & previous.isna())).all(axis=1)
    result[flag_name] = same & result["_ordinal"].sub(previous_ordinal).eq(1)
    return result[[*keys, flag_name]]


def _area_component(
    frame: pd.DataFrame,
    value_columns: list[str],
    prefix: str,
    repeated_columns: list[str] | None = None,
) -> pd.DataFrame:
    keys = ["quarter_code", "trade_area_code"]
    _ensure_unique(frame, keys, prefix)
    result = frame[[*keys, *value_columns, "source_base_date"]].copy()
    result[f"has_{prefix}"] = True
    result = result.rename(columns={"source_base_date": f"{prefix}_source_base_date"})
    if repeated_columns:
        result = result.merge(
            _repeated_flags(frame, repeated_columns, f"{prefix}_repeated_flag"),
            on=keys, how="left", validate="one_to_one",
        )
    return result


def build_area_quarter(
    floating_population: pd.DataFrame,
    resident_population: pd.DataFrame,
    working_population: pd.DataFrame,
    facilities: pd.DataFrame,
    trade_area_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    인구·가구·시설을 분기와 상권 단위로 결합한다.

    Args:
        floating_population: 유동인구 원본 DataFrame
        resident_population: 상주인구 원본 DataFrame
        working_population: 직장인구 원본 DataFrame
        facilities: 집객시설 원본 DataFrame
        trade_area_master: 상권 기준정보 DataFrame

    Returns:
        분기와 상권이 기본키인 인구·가구·시설 DataFrame
    """
    keys = ["quarter_code", "trade_area_code"]
    result = pd.concat(
        [frame[keys] for frame in (floating_population, resident_population, working_population, facilities)],
        ignore_index=True,
    ).drop_duplicates()

    resident_values = ["total_resident_population", "total_household_count", "apartment_household_count", "non_apartment_household_count"]
    resident_repeat = [column for column in resident_population if column.endswith(("_population", "_count"))]
    working_repeat = [column for column in working_population if column.endswith("_population")]
    facility_values = [target for target in _source_mapping("facilities").values() if target.endswith("_count")]
    components = [
        _area_component(floating_population, ["total_floating_population"], "floating_population").rename(
            columns={"floating_population_source_base_date": "floating_source_base_date"}
        ),
        _area_component(resident_population, resident_values, "resident_population", resident_repeat).rename(
            columns={"resident_population_source_base_date": "resident_source_base_date", "resident_population_repeated_flag": "resident_repeated_flag"}
        ),
        _area_component(working_population, ["total_working_population"], "working_population", working_repeat).rename(
            columns={"working_population_source_base_date": "working_source_base_date", "working_population_repeated_flag": "working_repeated_flag"}
        ),
        _area_component(facilities, facility_values, "facilities", facility_values),
    ]
    for component in components:
        result = result.merge(component, on=keys, how="left", validate="one_to_one")

    availability = ["has_floating_population", "has_resident_population", "has_working_population", "has_facilities"]
    for column in availability:
        result[column] = result[column].fillna(False).astype("boolean")
    master_columns = ["trade_area_code", "trade_area_name", "trade_area_type_code", "trade_area_type_name", "district_code", "district_name", "area_sqm", "area_ha"]
    result = result.merge(trade_area_master[master_columns], on="trade_area_code", how="left", validate="many_to_one")
    result["year"] = pd.to_numeric(result["quarter_code"].str[:4])
    result["quarter"] = pd.to_numeric(result["quarter_code"].str[4:5])
    result["floating_population_density"] = _safe_divide(result["total_floating_population"], result["area_ha"])
    result["resident_population_density"] = _safe_divide(result["total_resident_population"], result["area_ha"])
    result["working_population_density"] = _safe_divide(result["total_working_population"], result["area_ha"])
    result["facility_density"] = _safe_divide(result["total_facility_count"], result["area_ha"])
    result["apartment_household_share"] = _safe_divide(result["apartment_household_count"], result["total_household_count"])
    result["resident_to_working_ratio"] = _safe_divide(result["total_resident_population"], result["total_working_population"])
    return _finish_dataset(result, "area_quarter")


def _add_change(frame: pd.DataFrame, value: str, prefix: str, periods: int, ordinal: pd.Series) -> None:
    group_keys = ["trade_area_code", "industry_code"]
    previous = frame.groupby(group_keys, observed=True)[value].shift(periods)
    previous_ordinal = ordinal.groupby([frame[column] for column in group_keys], observed=True).shift(periods)
    valid = ordinal.sub(previous_ordinal).eq(periods) & frame[value].notna() & previous.notna()
    change = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    change.loc[valid] = frame.loc[valid, value].astype("Float64") - previous.loc[valid].astype("Float64")
    if prefix in {"sales_qoq", "sales_yoy", "store_qoq", "store_yoy"}:
        frame[f"{prefix}_change"] = change
    rate_valid = valid & previous.ne(0)
    rate = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    rate.loc[rate_valid] = change.loc[rate_valid] / previous.loc[rate_valid].astype("Float64") * 100
    frame[f"{prefix}_rate"] = rate


def _growth_counts(frame: pd.DataFrame, ordinal: pd.Series) -> tuple[pd.Series, pd.Series]:
    recent = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    consecutive = pd.Series(0, index=frame.index, dtype="Int64")
    for _, indices in frame.groupby(["trade_area_code", "industry_code"], observed=True, sort=False).groups.items():
        previous_ordinal: int | None = None
        streak = 0
        window: list[int] = []
        for index in indices:
            current = ordinal.loc[index]
            change = frame.at[index, "sales_qoq_change"]
            is_next = previous_ordinal is not None and not pd.isna(current) and int(current) == previous_ordinal + 1
            if not is_next or pd.isna(change):
                streak = 0
                window.clear()
            else:
                grew = int(change > 0)
                streak = streak + 1 if grew else 0
                window.append(grew)
                window = window[-4:]
                if len(window) == 4:
                    recent.at[index] = sum(window)
            consecutive.at[index] = streak
            previous_ordinal = None if pd.isna(current) else int(current)
    return recent, consecutive


def _add_reason(reasons: pd.Series, mask: pd.Series, text: str) -> None:
    mask = mask.astype("boolean").fillna(False).astype(bool)
    current = reasons.loc[mask]
    reasons.loc[mask] = current.where(current.eq(""), current + ";") + text


def build_commercial_quarter(
    stores: pd.DataFrame,
    sales: pd.DataFrame,
    area_quarter: pd.DataFrame,
    industry_codes: list[str] | tuple[str, ...] = SELECTED_INDUSTRY_CODES,
) -> pd.DataFrame:
    """
    점포 기준행에 매출과 상권 수요를 결합하고 파생값을 생성한다.

    Args:
        stores: 점포 원본 DataFrame
        sales: 매출 원본 DataFrame
        area_quarter: 분기별 상권 데이터 DataFrame
        industry_codes: 분석할 서비스 업종 코드

    Returns:
        분기·상권·업종 단위 분석 DataFrame
    """
    keys = ["quarter_code", "trade_area_code", "industry_code"]
    stores = stores.loc[stores["industry_code"].isin(industry_codes)].copy()
    sales = sales.loc[sales["industry_code"].isin(industry_codes)].copy()
    _ensure_unique(stores, keys, "stores")
    _ensure_unique(sales, keys, "sales")

    store_columns = [*keys, "industry_name", "total_store_count", "general_store_count", "franchise_store_count", "opened_store_count", "open_rate", "closed_store_count", "close_rate", "source_base_date"]
    sales_columns = [*keys, "quarterly_sales_amount", "quarterly_sales_count", "weekday_sales_amount", "weekend_sales_amount", "weekday_sales_count", "weekend_sales_count", "source_base_date"]
    result = stores[store_columns].rename(columns={"source_base_date": "store_source_base_date"})
    sales_part = sales[sales_columns].rename(columns={"source_base_date": "sales_source_base_date"})
    result = result.merge(sales_part, on=keys, how="left", validate="one_to_one", indicator="_sales_join")
    result["has_store"] = True
    result["has_sales"] = result.pop("_sales_join").eq("both")

    area_columns = [column for column in ANALYSIS_SCHEMAS["area_quarter"]["columns"] if column not in {"resident_repeated_flag", "working_repeated_flag", "facilities_repeated_flag", "non_apartment_household_count", "floating_population_density", "resident_population_density", "working_population_density", "facility_density"}]
    result = result.merge(area_quarter[area_columns], on=["quarter_code", "trade_area_code"], how="left", validate="many_to_one")

    result["monthly_average_sales_amount"] = _safe_divide(result["quarterly_sales_amount"], 3)
    result["quarterly_sales_per_store"] = _safe_divide(result["quarterly_sales_amount"], result["total_store_count"])
    result["monthly_average_sales_per_store"] = _safe_divide(result["quarterly_sales_per_store"], 3)
    result["quarterly_sales_count_per_store"] = _safe_divide(result["quarterly_sales_count"], result["total_store_count"])
    result["average_transaction_value"] = _safe_divide(result["quarterly_sales_amount"], result["quarterly_sales_count"])
    result["sales_per_floating_population"] = _safe_divide(result["quarterly_sales_amount"], result["total_floating_population"])
    result["floating_population_per_store"] = _safe_divide(result["total_floating_population"], result["total_store_count"])
    result["store_density"] = _safe_divide(result["total_store_count"], result["area_ha"])
    result["franchise_share"] = _safe_divide(result["franchise_store_count"], result["total_store_count"])
    result["weekend_sales_share"] = _safe_divide(result["weekend_sales_amount"], result["quarterly_sales_amount"])
    result["zero_store_flag"] = result["total_store_count"].eq(0)
    result["zero_sales_count_flag"] = result["has_sales"] & result["quarterly_sales_count"].eq(0)
    result["open_rate_over_100_flag"] = result["open_rate"].gt(100)
    result["close_rate_over_100_flag"] = result["close_rate"].gt(100)

    result = result.sort_values(["trade_area_code", "industry_code", "quarter_code"]).reset_index(drop=True)
    ordinal = _quarter_ordinal(result["quarter_code"])
    for value, prefix, periods in (
        ("quarterly_sales_amount", "sales_qoq", 1), ("quarterly_sales_amount", "sales_yoy", 4),
        ("total_store_count", "store_qoq", 1), ("total_store_count", "store_yoy", 4),
        ("quarterly_sales_per_store", "sales_per_store_qoq", 1),
        ("quarterly_sales_per_store", "sales_per_store_yoy", 4),
        ("total_floating_population", "floating_qoq", 1),
        ("total_floating_population", "floating_yoy", 4),
    ):
        _add_change(result, value, prefix, periods, ordinal)
    result["demand_store_yoy_gap"] = result["floating_yoy_rate"] - result["store_yoy_rate"]
    result["recent_4q_sales_growth_count"], result["consecutive_sales_growth_quarters"] = _growth_counts(result, ordinal)

    for flag in ("has_floating_population", "has_resident_population", "has_working_population", "has_facilities"):
        result[flag] = result[flag].fillna(False).astype("boolean")
    valid_area = result["area_ha"].notna() & result["area_ha"].gt(0)
    result["core_analysis_available"] = result["has_sales"] & result["has_floating_population"] & result["total_store_count"].gt(0) & valid_area
    result["full_analysis_available"] = result["core_analysis_available"] & result["has_resident_population"] & result["has_working_population"] & result["has_facilities"]
    reasons = pd.Series("", index=result.index, dtype="string")
    _add_reason(reasons, result["has_sales"].eq(False), "missing_sales")
    _add_reason(reasons, result["has_floating_population"].eq(False), "missing_floating_population")
    _add_reason(reasons, result["total_store_count"].isna(), "missing_store_count")
    _add_reason(reasons, result["total_store_count"].eq(0), "zero_store_count")
    _add_reason(reasons, valid_area.eq(False), "missing_or_zero_area")
    _add_reason(reasons, result["has_resident_population"].eq(False), "missing_resident_population")
    _add_reason(reasons, result["has_working_population"].eq(False), "missing_working_population")
    _add_reason(reasons, result["has_facilities"].eq(False), "missing_facilities")
    result["analysis_exclusion_reason"] = reasons.mask(reasons.eq(""), pd.NA)
    return _finish_dataset(result, "commercial_quarter")


def build_segment_comparison(
    sales: pd.DataFrame,
    floating_population: pd.DataFrame,
    trade_area_master: pd.DataFrame | None = None,
    industry_codes: list[str] | tuple[str, ...] = SELECTED_INDUSTRY_CODES,
) -> pd.DataFrame:
    """
    매출과 유동인구 구성 컬럼을 long 형식으로 변환해 비교한다.

    Args:
        sales: 매출 원본 DataFrame
        floating_population: 유동인구 원본 DataFrame
        trade_area_master: 상권명을 통일할 상권 기준정보 DataFrame
        industry_codes: 분석할 서비스 업종 코드

    Returns:
        분기·상권·업종·구성항목 단위 비교 DataFrame
    """
    sales = sales.loc[sales["industry_code"].isin(industry_codes)].copy()
    sales_keys = ["quarter_code", "trade_area_code", "industry_code"]
    floating_keys = ["quarter_code", "trade_area_code"]
    _ensure_unique(sales, sales_keys, "sales")
    _ensure_unique(floating_population, floating_keys, "floating_population")

    quarter_codes = sales["quarter_code"].dropna().drop_duplicates().tolist()
    if len(quarter_codes) > 1:
        quarter_frames = [
            build_segment_comparison(
                sales.loc[sales["quarter_code"].eq(quarter_code)],
                floating_population.loc[floating_population["quarter_code"].eq(quarter_code)],
                trade_area_master,
                industry_codes,
            )
            for quarter_code in quarter_codes
        ]
        category_columns = ANALYSIS_SCHEMAS["segment_comparison"]["dtypes"]["category"]
        for column in category_columns:
            categories = pd.Index([])
            for frame in quarter_frames:
                categories = categories.union(frame[column].cat.categories, sort=False)
            for frame in quarter_frames:
                frame[column] = frame[column].cat.set_categories(categories)
        result = pd.concat(quarter_frames, ignore_index=True)
        return _finish_dataset(result, "segment_comparison")

    # Long 변환 중 반복 문자열이 메모리를 과도하게 차지하지 않도록 임시 범주형을 쓴다.
    if trade_area_master is not None:
        name_mapping = trade_area_master.set_index("trade_area_code")["trade_area_name"]
        corrected_names = sales["trade_area_code"].map(name_mapping)
        sales["trade_area_name"] = corrected_names.fillna(sales["trade_area_name"])
    for column in floating_keys:
        categories = pd.concat([sales[column], floating_population[column]]).dropna().unique()
        sales[column] = pd.Categorical(sales[column], categories=categories)
        floating_population[column] = pd.Categorical(floating_population[column], categories=categories)
    for column in ("industry_code", "trade_area_name", "industry_name"):
        sales[column] = sales[column].astype("category")

    segment_types = list(SEGMENTS)
    segment_codes = [str(item["code"]) for items in SEGMENTS.values() for item in items]
    segment_names = [item["name"] for items in SEGMENTS.values() for item in items]

    sales_parts, floating_parts = [], []
    sales_base = [*sales_keys, "trade_area_name", "industry_name", "quarterly_sales_amount", "quarterly_sales_count"]
    for segment_type, definitions in SEGMENTS.items():
        for order, definition in enumerate(definitions, start=1):
            amount = definition["sales_amount_column"]
            count = definition["sales_count_column"]
            population = definition["floating_column"]
            sales_part = sales[[*sales_base, amount, count]].rename(columns={amount: "sales_amount", count: "sales_count"})
            sales_part = sales_part.assign(
                segment_type=segment_type, segment_code=str(definition["code"]),
                segment_name=definition["name"], segment_order=order,
            )
            sales_part["segment_type"] = sales_part["segment_type"].astype(pd.CategoricalDtype(segment_types))
            sales_part["segment_code"] = sales_part["segment_code"].astype(pd.CategoricalDtype(segment_codes))
            sales_part["segment_name"] = sales_part["segment_name"].astype(pd.CategoricalDtype(segment_names))
            sales_parts.append(sales_part)
            floating_part = floating_population[[*floating_keys, population]].rename(columns={population: "floating_population"})
            floating_part = floating_part.assign(segment_type=segment_type, segment_code=str(definition["code"]))
            floating_part["segment_type"] = floating_part["segment_type"].astype(pd.CategoricalDtype(segment_types))
            floating_part["segment_code"] = floating_part["segment_code"].astype(pd.CategoricalDtype(segment_codes))
            floating_parts.append(floating_part)

    result = pd.concat(sales_parts, ignore_index=True)
    floating_long = pd.concat(floating_parts, ignore_index=True)
    del sales_parts, floating_parts
    join_keys = [*floating_keys, "segment_type", "segment_code"]
    _ensure_unique(floating_long, join_keys, "floating_population_long")
    result = result.merge(floating_long, on=join_keys, how="left", validate="many_to_one", indicator="_floating_join")
    del floating_long
    group_keys = [*sales_keys, "segment_type"]
    amount_sum = result.groupby(group_keys, observed=True)["sales_amount"].transform(lambda values: values.sum(min_count=1))
    count_sum = result.groupby(group_keys, observed=True)["sales_count"].transform(lambda values: values.sum(min_count=1))
    floating_sum = result.groupby(group_keys, observed=True)["floating_population"].transform(lambda values: values.sum(min_count=1))
    result["sales_amount_share"] = _safe_divide(result["sales_amount"], amount_sum)
    result["sales_count_share"] = _safe_divide(result["sales_count"], count_sum)
    result["floating_population_share"] = _safe_divide(result["floating_population"], floating_sum)
    result["sales_amount_coverage"] = _safe_divide(amount_sum, result["quarterly_sales_amount"])
    result["sales_count_coverage"] = _safe_divide(count_sum, result["quarterly_sales_count"])
    result["comparison_available"] = result["sales_amount_share"].notna() & result["floating_population_share"].notna() & result["_floating_join"].eq("both")
    reasons = pd.Series("", index=result.index, dtype="string")
    _add_reason(reasons, result["sales_amount"].isna(), "missing_sales_segment")
    _add_reason(reasons, amount_sum.eq(0), "zero_sales_segment_total")
    _add_reason(reasons, result["_floating_join"].ne("both") | result["floating_population"].isna(), "missing_floating_segment")
    _add_reason(reasons, floating_sum.eq(0), "zero_floating_segment_total")
    result["comparison_exclusion_reason"] = reasons.mask(reasons.eq(""), pd.NA)
    result["sales_floating_share_gap"] = result["sales_amount_share"] - result["floating_population_share"]
    result["sales_floating_share_ratio"] = _safe_divide(result["sales_amount_share"], result["floating_population_share"])
    return _finish_dataset(result, "segment_comparison")


def _component_difference(dataset_name: str, frame: pd.DataFrame) -> tuple[int, float]:
    differences: list[pd.Series] = []
    if dataset_name == "area_quarter":
        differences.append((frame["apartment_household_count"] + frame["non_apartment_household_count"] - frame["total_household_count"]).abs())
    elif dataset_name == "commercial_quarter":
        differences.extend([
            (frame["weekday_sales_amount"] + frame["weekend_sales_amount"] - frame["quarterly_sales_amount"]).abs(),
            (frame["weekday_sales_count"] + frame["weekend_sales_count"] - frame["quarterly_sales_count"]).abs(),
        ])
    elif dataset_name == "segment_comparison":
        exact = frame.loc[frame["segment_type"].isin(["weekday", "time_band"])]
        keys = ["quarter_code", "trade_area_code", "industry_code", "segment_type"]
        differences.append((exact.groupby(keys, observed=True)["sales_amount_coverage"].first() - 1).abs())
    combined = pd.concat(differences, ignore_index=True).dropna() if differences else pd.Series(dtype="Float64")
    return (int(combined.gt(1e-9).sum()), float(combined.max())) if not combined.empty else (0, 0.0)


def build_data_quality(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    최종 데이터셋의 키, 조인, 합계, 반복값 품질을 요약한다.

    Args:
        datasets: 품질을 검사할 최종 분석 DataFrame 딕셔너리

    Returns:
        데이터셋과 분기 단위 품질 검사 DataFrame

    Raises:
        ValueError: 필수 분석 데이터셋이 없는 경우
    """
    expected = {"trade_area_master", "area_quarter", "commercial_quarter", "segment_comparison"}
    missing = sorted(expected.difference(datasets))
    if missing:
        raise ValueError(f"품질 검사 대상이 없습니다: {', '.join(missing)}")
    rows = []
    for name in sorted(expected):
        frame = datasets[name]
        keys = ANALYSIS_SCHEMAS[name]["primary_key"]
        groups = frame.groupby("quarter_code", observed=True, dropna=False) if "quarter_code" in frame else [("ALL", frame)]
        for quarter_code, group in groups:
            missing_keys = int(group[keys].isna().any(axis=1).sum())
            duplicate_keys = int(group.duplicated(keys, keep=False).sum())
            numeric_columns = [column for column in group.select_dtypes(include="number") if not any(token in column for token in ("change", "rate", "gap", "longitude", "latitude"))]
            negative_values = int(sum(group[column].lt(0).sum() for column in numeric_columns))
            invalid_values = 0
            if name == "commercial_quarter":
                invalid_values = int(group["open_rate_over_100_flag"].fillna(False).sum() + group["close_rate_over_100_flag"].fillna(False).sum())

            if name == "area_quarter":
                target = "floating,resident,working,facilities"
                success = group[["has_floating_population", "has_resident_population", "has_working_population", "has_facilities"]].fillna(False).all(axis=1)
            elif name == "commercial_quarter":
                target, success = "sales,area_quarter", group["core_analysis_available"].fillna(False)
            elif name == "segment_comparison":
                target, success = "sales,floating_population", group["comparison_available"].fillna(False)
            else:
                target, success = "none", pd.Series(True, index=group.index)

            mismatch_count, max_difference = _component_difference(name, group)
            repeated_columns = [column for column in group if column.endswith("_repeated_flag")]
            repeated_rate = float(group[repeated_columns].fillna(False).any(axis=1).mean()) if repeated_columns else 0.0
            success_count = int(success.sum())
            notes = []
            if success_count != len(group): notes.append(f"join unavailable={len(group) - success_count}")
            if invalid_values: notes.append(f"invalid values={invalid_values}")
            if mismatch_count: notes.append(f"component mismatch={mismatch_count}")
            if repeated_rate: notes.append(f"repeated value rate={repeated_rate:.6f}")
            if missing_keys or duplicate_keys or negative_values:
                status = "error"
            elif notes:
                status = "warning"
            else:
                status = "ok"
            rows.append({
                "dataset_name": name, "quarter_code": str(quarter_code),
                "row_count": len(group), "column_count": len(frame.columns),
                "missing_key_count": missing_keys, "duplicate_key_count": duplicate_keys,
                "missing_value_count": int(group.isna().sum().sum()),
                "negative_value_count": negative_values, "invalid_value_count": invalid_values,
                "join_target": target, "join_success_count": success_count,
                "join_success_rate": success_count / len(group) if len(group) else math.nan,
                "component_mismatch_count": mismatch_count,
                "component_max_difference": max_difference, "repeated_value_rate": repeated_rate,
                "quality_status": status, "quality_note": "; ".join(notes) or "no issue",
            })
    return _finish_dataset(pd.DataFrame(rows), "data_quality")


def _atomic_to_csv(frame: pd.DataFrame, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8-sig", newline="", dir=output_dir,
            prefix=f".{os.path.basename(output_path)}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = temporary.name
            frame.to_csv(temporary, index=False, date_format="%Y-%m-%d")
        os.replace(temporary_path, output_path)
        temporary_path = None
    except OSError as error:
        raise ValueError(f"CSV를 저장하지 못했습니다: {output_path}") from error
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)


def write_csv_pair(
    frame: pd.DataFrame,
    english_filename: str,
    korean_filename: str,
    english_output_dir: str = DEFAULT_ENGLISH_OUTPUT_DIR,
    korean_output_dir: str = DEFAULT_KOREAN_OUTPUT_DIR,
    column_mapping: dict[str, str] = ANALYSIS_COLUMN_MAPPING_KO,
) -> tuple[str, str]:
    """
    영문 CSV와 헤더만 한국어로 바꾼 동일 값 CSV를 저장한다.

    Args:
        frame: 저장할 분석 DataFrame
        english_filename: 영문 CSV 파일명
        korean_filename: 한국어 CSV 파일명
        english_output_dir: 영문 CSV 출력 디렉터리
        korean_output_dir: 한국어 CSV 출력 디렉터리
        column_mapping: 영문 컬럼명과 한국어 컬럼명의 매핑

    Returns:
        저장한 영문 CSV와 한국어 CSV 경로

    Raises:
        ValueError: 매핑이 없거나 두 출력 경로가 같은 경우
    """
    missing = [column for column in frame if column not in column_mapping]
    if missing:
        raise ValueError(f"한국어 분석 매핑에 없는 컬럼입니다: {', '.join(missing)}")
    english_path = os.path.abspath(os.path.join(english_output_dir, english_filename))
    korean_path = os.path.abspath(os.path.join(korean_output_dir, korean_filename))
    if english_path == korean_path:
        raise ValueError("영문과 한국어 CSV의 출력 경로는 달라야 합니다.")
    _atomic_to_csv(frame, english_path)
    _atomic_to_csv(frame.rename(columns=column_mapping), korean_path)
    return english_path, korean_path


def run_preprocessing(
    input_dir: str = DEFAULT_INPUT_DIR,
    english_output_dir: str = DEFAULT_ENGLISH_OUTPUT_DIR,
    korean_output_dir: str = DEFAULT_KOREAN_OUTPUT_DIR,
    industry_codes: list[str] | tuple[str, ...] = SELECTED_INDUSTRY_CODES,
) -> dict[str, pd.DataFrame]:
    """
    원본 로드부터 영문·한국어 CSV 저장까지 전체 전처리를 실행한다.

    Args:
        input_dir: 원본 CSV 디렉터리
        english_output_dir: 영문 분석 CSV 출력 디렉터리
        korean_output_dir: 한국어 참고 CSV 출력 디렉터리
        industry_codes: 분석할 서비스 업종 코드

    Returns:
        최종 분석 DataFrame 5종
    """
    source = load_source_data(input_dir, industry_codes)
    master = build_trade_area_master(source["trade_area"])
    area = build_area_quarter(
        source["floating_population"], source["resident_population"],
        source["working_population"], source["facilities"], master,
    )
    commercial = build_commercial_quarter(source["stores"], source["sales"], area, industry_codes)
    segments = build_segment_comparison(source["sales"], source["floating_population"], master, industry_codes)
    datasets = {
        "trade_area_master": master, "area_quarter": area,
        "commercial_quarter": commercial, "segment_comparison": segments,
    }
    datasets["data_quality"] = build_data_quality(datasets)
    for name, frame in datasets.items():
        output = OUTPUTS[name]
        write_csv_pair(
            frame, output["english_filename"], output["korean_filename"],
            english_output_dir, korean_output_dir,
        )
    return datasets


def main() -> None:
    """터미널에서 서울 상권 데이터 전처리를 실행한다."""
    parser = argparse.ArgumentParser(description="서울 상권 원본 7종을 분석 CSV로 전처리합니다.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--english-output-dir", default=DEFAULT_ENGLISH_OUTPUT_DIR)
    parser.add_argument("--korean-output-dir", default=DEFAULT_KOREAN_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        datasets = run_preprocessing(args.input_dir, args.english_output_dir, args.korean_output_dir)
    except (ValueError, MemoryError) as error:
        parser.exit(1, f"전처리 실패: {error}\n")
    for name, frame in datasets.items():
        print(f"저장 완료: {OUTPUTS[name]['english_filename']} / {OUTPUTS[name]['korean_filename']} ({len(frame):,}행)")


if __name__ == "__main__":
    main()
