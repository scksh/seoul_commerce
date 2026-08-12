"""대시보드용 서울 상권 분석 마트 생성 모듈.

전처리된 영문 CSV와 저장된 EBM 평가 결과를 읽어 대시보드가 원본을
재집계하지 않고 사용할 수 있는 요약·추세·구성·경쟁·맥락 마트를 만든다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "analysis_pipeline.yml"
KEY_COLUMNS = ["base_quarter", "trade_area_code", "industry_code"]


def load_analysis_config(path: str | os.PathLike[str] = DEFAULT_CONFIG_FILE) -> dict[str, Any]:
    """분석 범위, 경로, 모델 후보가 든 YAML 설정을 검증해 반환한다."""
    config_path = Path(path).resolve()
    try:
        with config_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"분석 설정을 읽을 수 없습니다: {config_path}") from error

    required = {"analysis", "paths", "mart_files", "context_metrics", "model"}
    if not isinstance(config, dict) or not required.issubset(config):
        missing = sorted(required.difference(config or {}))
        raise ValueError(f"분석 설정에 필수 섹션이 없습니다: {', '.join(missing)}")
    reference = str(config["analysis"].get("reference_quarter", ""))
    if not _valid_quarter(reference):
        raise ValueError(f"기준분기 형식이 올바르지 않습니다: {reference}")
    if int(config["analysis"].get("recent_quarters", 0)) < 4:
        raise ValueError("recent_quarters는 4 이상이어야 합니다.")
    if not config["model"].get("candidates"):
        raise ValueError("model.candidates가 비어 있습니다.")
    return config


def _valid_quarter(value: str) -> bool:
    return len(value) == 5 and value[:4].isdigit() and value[-1] in "1234"


def quarter_to_ordinal(code: str) -> int:
    """YYYYQ 형식 분기 코드를 연속 정수로 변환한다."""
    code = str(code)
    if not _valid_quarter(code):
        raise ValueError(f"분기 코드 형식이 올바르지 않습니다: {code}")
    return int(code[:4]) * 4 + int(code[-1]) - 1


def ordinal_to_quarter(value: int) -> str:
    """연속 분기 정수를 YYYYQ 형식으로 변환한다."""
    year, offset = divmod(int(value), 4)
    return f"{year}{offset + 1}"


def recent_quarter_codes(reference_quarter: str, count: int) -> list[str]:
    """기준분기를 포함한 최근 연속 분기 코드를 오래된 순서로 반환한다."""
    end = quarter_to_ordinal(reference_quarter)
    return [ordinal_to_quarter(value) for value in range(end - count + 1, end + 1)]


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name}에 필수 컬럼이 없습니다: {', '.join(missing)}")


def _read_csv(path: Path, code_columns: list[str]) -> pd.DataFrame:
    dtype = {column: "string" for column in code_columns}
    try:
        return pd.read_csv(path, dtype=dtype, low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise ValueError(f"분석 CSV를 읽을 수 없습니다: {path}") from error


def build_eligibility(
    commercial: pd.DataFrame,
    segments: pd.DataFrame,
    reference_quarter: str,
    recent_quarters: int = 8,
    minimum_trade_areas: int = 50,
) -> pd.DataFrame:
    """업종별 최근 연속 자료와 21개 구성항목을 기준으로 제공 상태를 계산한다."""
    commercial_required = [
        "quarter_code", "trade_area_code", "industry_code", "full_analysis_available",
        "quarterly_sales_amount", "quarterly_sales_count", "total_store_count", "area_ha",
    ]
    segment_required = [
        "quarter_code", "trade_area_code", "industry_code", "segment_type", "segment_code",
        "comparison_available",
    ]
    _require_columns(commercial, commercial_required, "commercial_quarter")
    _require_columns(segments, segment_required, "segment_comparison")

    quarters = recent_quarter_codes(reference_quarter, recent_quarters)
    history = commercial[commercial["quarter_code"].isin(quarters)].copy()
    positive = history[[
        "quarterly_sales_amount", "quarterly_sales_count", "total_store_count", "area_ha"
    ]].gt(0).all(axis=1)
    complete = history[history["full_analysis_available"].eq(True) & positive]
    commercial_counts = complete.groupby(
        ["industry_code", "trade_area_code"], observed=True
    )["quarter_code"].nunique()

    available_segments = segments[
        segments["quarter_code"].isin(quarters) & segments["comparison_available"].eq(True)
    ].drop_duplicates([
        "quarter_code", "trade_area_code", "industry_code", "segment_type", "segment_code"
    ])
    item_counts = available_segments.groupby(
        ["quarter_code", "industry_code", "trade_area_code"], observed=True
    ).size()
    complete_segment_quarters = item_counts[item_counts.eq(21)].reset_index()
    segment_counts = complete_segment_quarters.groupby(
        ["industry_code", "trade_area_code"], observed=True
    )["quarter_code"].nunique()

    current = commercial[commercial["quarter_code"].eq(reference_quarter)][
        ["industry_code", "trade_area_code"]
    ].drop_duplicates()
    current_index = pd.MultiIndex.from_frame(current)
    commercial_ok = commercial_counts.reindex(current_index).eq(recent_quarters).to_numpy()
    segments_ok = segment_counts.reindex(current_index).eq(recent_quarters).to_numpy()
    result = current.copy()
    result["commercial_history_complete"] = commercial_ok
    result["segment_history_complete"] = segments_ok
    result["eligible"] = result["commercial_history_complete"] & result["segment_history_complete"]
    counts = result.groupby("industry_code", observed=True)["eligible"].transform("sum").astype("Int64")
    result["eligible_trade_area_count"] = counts
    result["minimum_trade_areas_met"] = counts.ge(minimum_trade_areas)
    result["report_available"] = result["eligible"] & result["minimum_trade_areas_met"]
    result["availability_reason"] = np.select(
        [
            ~result["commercial_history_complete"],
            ~result["segment_history_complete"],
            ~result["minimum_trade_areas_met"],
        ],
        ["incomplete_commercial_history", "incomplete_segment_history", "insufficient_peers"],
        default="available",
    )
    return result.reset_index(drop=True)


def add_rank_metrics(sample: pd.DataFrame) -> pd.DataFrame:
    """기준분기 표본에 전체 순위와 동일 상권유형 백분위를 추가한다."""
    required = [
        "industry_code", "trade_area_code", "trade_area_type_name",
        "quarterly_sales_amount", "quarterly_sales_per_store", "sales_yoy_rate",
    ]
    _require_columns(sample, required, "순위 표본")
    result = sample.copy()

    def add_metric(metric: str, rank_column: str, percentile_column: str) -> None:
        ordered_parts: list[pd.DataFrame] = []
        for _, industry in result.groupby("industry_code", observed=True):
            ordered = industry.sort_values(
                [metric, "sales_yoy_rate", "trade_area_code"],
                ascending=[False, False, True], na_position="last",
            ).copy()
            ordered[rank_column] = np.arange(1, len(ordered) + 1)
            ordered_parts.append(ordered)
        ordered = pd.concat(ordered_parts)
        result[rank_column] = ordered[rank_column].sort_index().astype("Int64")

        peer_keys = ["industry_code", "trade_area_type_name"]
        peer_rank = ordered.groupby(peer_keys, observed=True).cumcount() + 1
        peer_count = ordered.groupby(peer_keys, observed=True)["trade_area_code"].transform("size")
        percentile = (
            100 * (peer_count - peer_rank + 1) / peer_count
        )
        result[percentile_column] = percentile.sort_index()

    add_metric("quarterly_sales_amount", "total_sales_rank", "total_sales_peer_percentile")
    add_metric(
        "quarterly_sales_per_store", "sales_per_store_rank",
        "sales_per_store_peer_percentile",
    )
    peer_keys = ["industry_code", "trade_area_type_name"]
    result["peer_trade_area_count"] = result.groupby(peer_keys, observed=True)[
        "trade_area_code"
    ].transform("size").astype("Int64")
    return result


def _growth_metrics(commercial: pd.DataFrame, reference_quarter: str, recent_count: int) -> pd.DataFrame:
    keys = ["trade_area_code", "industry_code"]
    current = commercial[commercial["quarter_code"].eq(reference_quarter)].copy()
    previous_quarter = ordinal_to_quarter(quarter_to_ordinal(reference_quarter) - 4)
    previous = commercial[commercial["quarter_code"].eq(previous_quarter)][
        keys + [
            "total_store_count", "quarterly_sales_count_per_store",
            "average_transaction_value", "quarterly_sales_amount",
        ]
    ].rename(columns={
        "total_store_count": "previous_store_count",
        "quarterly_sales_count_per_store": "previous_sales_count_per_store",
        "average_transaction_value": "previous_transaction_value",
        "quarterly_sales_amount": "previous_sales_amount",
    })
    result = current[keys + [
        "total_store_count", "quarterly_sales_count_per_store", "average_transaction_value",
        "quarterly_sales_amount", "recent_4q_sales_growth_count",
        "consecutive_sales_growth_quarters",
    ]].merge(previous, on=keys, how="left")

    factor_pairs = [
        ("total_store_count", "previous_store_count", "store_log_contribution"),
        ("quarterly_sales_count_per_store", "previous_sales_count_per_store", "volume_log_contribution"),
        ("average_transaction_value", "previous_transaction_value", "ticket_log_contribution"),
    ]
    for current_column, previous_column, output_column in factor_pairs:
        valid = result[current_column].gt(0) & result[previous_column].gt(0)
        result[output_column] = np.nan
        result.loc[valid, output_column] = 100 * np.log(
            result.loc[valid, current_column] / result.loc[valid, previous_column]
        )
    valid_sales = result["quarterly_sales_amount"].gt(0) & result["previous_sales_amount"].gt(0)
    result["total_sales_log_change"] = np.nan
    result.loc[valid_sales, "total_sales_log_change"] = 100 * np.log(
        result.loc[valid_sales, "quarterly_sales_amount"]
        / result.loc[valid_sales, "previous_sales_amount"]
    )
    contribution_columns = [
        "store_log_contribution", "volume_log_contribution", "ticket_log_contribution"
    ]
    positive_count = result[contribution_columns].gt(0).sum(axis=1)
    has_contribution = result[contribution_columns].notna().any(axis=1)
    sole_driver = result[contribution_columns].fillna(-np.inf).idxmax(axis=1).map({
        "store_log_contribution": "store_expansion",
        "volume_log_contribution": "transaction_growth",
        "ticket_log_contribution": "ticket_growth",
    })
    sole_driver = sole_driver.where(has_contribution, "unavailable")
    result["growth_type"] = np.select(
        [
            result["total_sales_log_change"].isna(),
            result["total_sales_log_change"].le(0),
            positive_count.eq(1),
        ],
        ["unavailable", "decline_or_stagnation", sole_driver],
        default="joint_growth",
    )

    quarters = recent_quarter_codes(reference_quarter, recent_count)
    history = commercial[commercial["quarter_code"].isin(quarters)].copy()

    def summarize(group: pd.DataFrame) -> pd.Series:
        ordered = group.sort_values("quarter_code")
        continuous = ordered["quarter_code"].astype(str).tolist() == quarters
        sales = pd.to_numeric(ordered["quarterly_sales_amount"], errors="coerce")
        if not continuous or len(ordered) != recent_count or sales.isna().any() or sales.le(0).any():
            return pd.Series({"quarterly_trend_rate": np.nan, "sales_yoy_volatility": np.nan})
        slope = np.polyfit(np.arange(recent_count), np.log(sales.to_numpy(dtype=float)), 1)[0]
        return pd.Series({
            "quarterly_trend_rate": 100 * np.expm1(slope),
            "sales_yoy_volatility": pd.to_numeric(
                ordered["sales_yoy_rate"], errors="coerce"
            ).std(ddof=0),
        })

    trends = history.groupby(keys, observed=True).apply(summarize, include_groups=False).reset_index()
    return result[keys + contribution_columns + [
        "total_sales_log_change", "growth_type",
    ]].merge(trends, on=keys, how="left")


def build_trend_mart(
    commercial: pd.DataFrame,
    reference_quarter: str,
    recent_count: int,
    industry_codes: list[str],
) -> pd.DataFrame:
    """최근 연속 분기의 핵심 KPI와 전년 동기 증감률을 반환한다."""
    quarters = recent_quarter_codes(reference_quarter, recent_count)
    columns = [
        "quarter_code", "trade_area_code", "trade_area_name", "industry_code", "industry_name",
        "quarterly_sales_amount", "total_store_count", "quarterly_sales_count",
        "average_transaction_value", "total_floating_population", "sales_yoy_rate",
        "store_yoy_rate", "floating_yoy_rate", "sales_per_store_yoy_rate",
    ]
    result = commercial[
        commercial["quarter_code"].isin(quarters)
        & commercial["industry_code"].isin(industry_codes)
    ][columns].copy()
    result.insert(0, "base_quarter", reference_quarter)
    result = result.rename(columns={"quarter_code": "observed_quarter"})
    return result.sort_values(KEY_COLUMNS + ["observed_quarter"]).reset_index(drop=True)


def build_segments_mart(
    segments: pd.DataFrame,
    reference_quarter: str,
    industry_codes: list[str],
) -> pd.DataFrame:
    """기준분기의 21개 구성비와 유형별 구성 차이 지수를 반환한다."""
    result = segments[
        segments["quarter_code"].eq(reference_quarter)
        & segments["industry_code"].isin(industry_codes)
    ].copy()
    result["composition_distance"] = result.groupby(
        ["trade_area_code", "industry_code", "segment_type"], observed=True
    )["sales_floating_share_gap"].transform(lambda values: values.abs().sum(min_count=1) / 2)
    result = result.rename(columns={"quarter_code": "base_quarter"})
    columns = KEY_COLUMNS + [
        "trade_area_name", "industry_name", "segment_type", "segment_code", "segment_name",
        "segment_order", "sales_amount_share", "sales_count_share", "floating_population_share",
        "sales_floating_share_gap", "sales_floating_share_ratio", "composition_distance",
        "sales_amount_coverage", "comparison_available", "comparison_exclusion_reason",
    ]
    return result[columns].sort_values(KEY_COLUMNS + ["segment_type", "segment_order"]).reset_index(drop=True)


def _assign_density_groups(sample: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for industry_code, group in sample.groupby("industry_code", observed=True):
        valid = group.dropna(subset=[
            "store_density", "quarterly_sales_amount", "quarterly_sales_per_store"
        ]).copy()
        positive = valid[[
            "store_density", "quarterly_sales_amount", "quarterly_sales_per_store"
        ]].gt(0).all(axis=1)
        valid = valid[positive].sort_values(["store_density", "trade_area_code"])
        if len(valid) < len(labels):
            continue
        valid["density_group"] = pd.qcut(
            valid["store_density"].rank(method="first"), q=len(labels), labels=labels
        ).astype("string")
        parts.append(valid)
    if not parts:
        return sample.iloc[0:0].assign(density_group=pd.Series(dtype="string"))
    return pd.concat(parts, ignore_index=True)


def build_competition_mart(
    current_sample: pd.DataFrame,
    reference_quarter: str,
    density_groups: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """점포밀도 5분위 요약과 개별 상권의 밀도 그룹을 반환한다."""
    assigned = _assign_density_groups(current_sample, density_groups)
    summary = assigned.groupby(["industry_code", "density_group"], observed=True).agg(
        sample_count=("trade_area_code", "size"),
        density_min=("store_density", "min"),
        density_median=("store_density", "median"),
        density_max=("store_density", "max"),
        total_sales_median=("quarterly_sales_amount", "median"),
        sales_per_store_median=("quarterly_sales_per_store", "median"),
    ).reset_index()
    summary.insert(0, "base_quarter", reference_quarter)
    area_groups = assigned[["trade_area_code", "industry_code", "density_group"]].copy()
    return summary, area_groups


def _time_profiles(segments: pd.DataFrame, reference_quarter: str) -> pd.DataFrame:
    codes = ["sat", "sun", "11_14", "17_21", "21_24", "00_06"]
    rows = segments[
        segments["quarter_code"].eq(reference_quarter)
        & segments["segment_code"].isin(codes)
        & segments["comparison_available"].eq(True)
    ][["trade_area_code", "industry_code", "segment_code", "floating_population_share"]]
    pivot = rows.pivot(
        index=["trade_area_code", "industry_code"],
        columns="segment_code",
        values="floating_population_share",
    )
    required = set(codes)
    if not required.issubset(pivot.columns):
        return pd.DataFrame(columns=["trade_area_code", "industry_code"])
    result = pd.DataFrame(index=pivot.index)
    result["weekend_floating_share"] = pivot["sat"] + pivot["sun"]
    result["lunch_floating_share"] = pivot["11_14"]
    result["evening_floating_share"] = pivot["17_21"]
    result["late_night_floating_share"] = pivot["21_24"] + pivot["00_06"]
    return result.reset_index()


def build_context_mart(
    commercial: pd.DataFrame,
    segments: pd.DataFrame,
    eligibility: pd.DataFrame,
    reference_quarter: str,
    recent_count: int,
    context_count: int,
    definitions: list[dict[str, Any]],
) -> pd.DataFrame:
    """상권 환경 지표 값과 동일 업종·상권유형 내 백분위를 긴 형식으로 만든다."""
    commercial = commercial.copy()
    if "facility_density" not in commercial and {
        "total_facility_count", "area_ha"
    }.issubset(commercial):
        valid_area = pd.to_numeric(commercial["area_ha"], errors="coerce").gt(0)
        commercial["facility_density"] = np.nan
        commercial.loc[valid_area, "facility_density"] = (
            pd.to_numeric(commercial.loc[valid_area, "total_facility_count"], errors="coerce")
            / pd.to_numeric(commercial.loc[valid_area, "area_ha"], errors="coerce")
        )
    eligible_keys = eligibility[eligibility["eligible"]][["trade_area_code", "industry_code"]]
    current = commercial[commercial["quarter_code"].eq(reference_quarter)].merge(
        eligible_keys, on=["trade_area_code", "industry_code"], how="inner"
    )
    slow_columns = [
        "total_resident_population", "total_working_population", "resident_to_working_ratio",
        "total_household_count", "apartment_household_share", "facility_density",
        "subway_station_count", "bus_stop_count",
    ]
    context_quarters = recent_quarter_codes(reference_quarter, recent_count)[-context_count:]
    slow = commercial[
        commercial["quarter_code"].isin(context_quarters)
    ].merge(eligible_keys, on=["trade_area_code", "industry_code"], how="inner")
    slow = slow.groupby(["trade_area_code", "industry_code"], observed=True)[slow_columns].median().reset_index()
    context = current[[
        "trade_area_code", "trade_area_name", "industry_code", "industry_name",
        "trade_area_type_name", "store_density", "franchise_share",
    ]].merge(slow, on=["trade_area_code", "industry_code"], how="inner")
    context = context.merge(_time_profiles(segments, reference_quarter),
                            on=["trade_area_code", "industry_code"], how="inner")

    rows: list[pd.DataFrame] = []
    peer_keys = ["industry_code", "trade_area_type_name"]
    for definition in definitions:
        metric = str(definition["column"])
        if metric not in context:
            continue
        value = pd.to_numeric(context[metric], errors="coerce")
        percentile = context.assign(_value=value).groupby(peer_keys, observed=True)["_value"].rank(
            method="average", pct=True
        ) * 100
        available = context.assign(_value=value).groupby(peer_keys, observed=True)["_value"].transform(
            "nunique"
        ).ge(2) & value.notna()
        part = context.loc[available, [
            "trade_area_code", "trade_area_name", "industry_code", "industry_name"
        ]].copy()
        part["domain"] = str(definition["domain"])
        part["metric"] = str(definition["label"])
        part["value"] = value.loc[available].to_numpy() * float(definition.get("scale", 1))
        part["unit"] = str(definition["unit"])
        part["peer_percentile"] = percentile.loc[available].to_numpy()
        rows.append(part)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.insert(0, "base_quarter", reference_quarter)
    return result.sort_values(KEY_COLUMNS + ["domain", "metric"]).reset_index(drop=True)


def build_summary_mart(
    commercial: pd.DataFrame,
    master: pd.DataFrame,
    eligibility: pd.DataFrame,
    context: pd.DataFrame,
    area_density_groups: pd.DataFrame,
    reference_quarter: str,
    recent_count: int,
    industry_codes: list[str],
    predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """제공 상태, KPI, 순위, 성장, 맥락, 모델 결과를 한 행으로 종합한다."""
    sample = commercial[
        commercial["quarter_code"].eq(reference_quarter)
        & commercial["industry_code"].isin(industry_codes)
    ].copy()
    ranked = add_rank_metrics(sample[sample["core_analysis_available"].eq(True)])
    rank_columns = [
        "trade_area_code", "industry_code", "total_sales_rank", "sales_per_store_rank",
        "total_sales_peer_percentile", "sales_per_store_peer_percentile", "peer_trade_area_count",
    ]
    summary = sample.merge(ranked[rank_columns], on=["trade_area_code", "industry_code"], how="left")
    summary = summary.merge(master[["trade_area_code", "longitude", "latitude"]],
                            on="trade_area_code", how="left")
    summary = summary.merge(eligibility, on=["trade_area_code", "industry_code"], how="left")
    summary = summary.merge(_growth_metrics(commercial, reference_quarter, recent_count),
                            on=["trade_area_code", "industry_code"], how="left")
    summary = summary.merge(area_density_groups, on=["trade_area_code", "industry_code"], how="left")

    if not context.empty:
        values = context.pivot(index=["trade_area_code", "industry_code"], columns="metric", values="value")
        values.columns = [f"context_{column}" for column in values.columns]
        percentiles = context.pivot(
            index=["trade_area_code", "industry_code"], columns="metric", values="peer_percentile"
        )
        percentiles.columns = [f"context_{column}_percentile" for column in percentiles.columns]
        summary = summary.merge(values.join(percentiles).reset_index(),
                                on=["trade_area_code", "industry_code"], how="left")
    if predictions is not None and not predictions.empty:
        prediction_columns = [
            "execution_id", "quarter_code", "trade_area_code", "industry_code",
            "actual_sales_per_store", "predicted_sales_per_store", "prediction_error_rate",
        ]
        summary = summary.merge(predictions[prediction_columns], on=[
            "quarter_code", "trade_area_code", "industry_code"
        ], how="left")

    summary = summary.rename(columns={"quarter_code": "base_quarter"})
    return summary.sort_values(KEY_COLUMNS).reset_index(drop=True)


MART_KEYS = {
    "summary": KEY_COLUMNS,
    "trend": KEY_COLUMNS + ["observed_quarter"],
    "segments": KEY_COLUMNS + ["segment_type", "segment_code"],
    "competition": ["base_quarter", "industry_code", "density_group"],
    "context": KEY_COLUMNS + ["metric"],
    "model_metrics": ["execution_id", "candidate", "validation_fold"],
    "model_global": ["execution_id", "term"],
    "model_local": KEY_COLUMNS + ["execution_id", "term"],
}


def validate_mart(frame: pd.DataFrame, name: str) -> None:
    """마트 필수 키의 존재, 결측, 중복을 검사한다."""
    keys = MART_KEYS[name]
    _require_columns(frame, keys, name)
    missing = int(frame[keys].isna().any(axis=1).sum())
    duplicates = int(frame.duplicated(keys, keep=False).sum())
    if missing or duplicates:
        raise ValueError(f"{name} 키 오류: 결측 {missing:,}행, 중복 관련 {duplicates:,}행")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", suffix=".tmp",
        prefix=f".{path.name}.", dir=path.parent, delete=False,
    ) as file:
        temporary = Path(file.name)
        frame.to_csv(file, index=False)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", prefix=f".{path.name}.",
        dir=path.parent, delete=False,
    ) as file:
        temporary = Path(file.name)
        json.dump(data, file, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_analysis_marts(
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_FILE,
    reference_quarter: str | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    model_evaluation_path: str | os.PathLike[str] | None = None,
) -> dict[str, Path]:
    """전처리 CSV와 저장된 모델 평가 결과로 모든 대시보드 마트를 생성한다."""
    config = load_analysis_config(config_path)
    analysis = config["analysis"]
    reference = str(reference_quarter or analysis["reference_quarter"])
    if not _valid_quarter(reference):
        raise ValueError(f"기준분기 형식이 올바르지 않습니다: {reference}")
    industry_codes = [str(value) for value in analysis["industry_codes"]]
    paths = {name: _project_path(value) for name, value in config["paths"].items()}
    mart_dir = Path(output_dir).resolve() if output_dir else paths["mart_dir"]

    commercial = _read_csv(paths["commercial_quarter"], [
        "quarter_code", "trade_area_code", "industry_code"
    ])
    master = _read_csv(paths["trade_area_master"], ["trade_area_code"])
    segments = _read_csv(paths["segment_comparison"], [
        "quarter_code", "trade_area_code", "industry_code", "segment_code"
    ])
    commercial = commercial[commercial["industry_code"].isin(industry_codes)].copy()
    segments = segments[segments["industry_code"].isin(industry_codes)].copy()

    recent_count = int(analysis["recent_quarters"])
    eligibility = build_eligibility(
        commercial, segments, reference, recent_count,
        int(analysis["minimum_eligible_trade_areas"]),
    )
    current = commercial[
        commercial["quarter_code"].eq(reference)
        & commercial["core_analysis_available"].eq(True)
    ]
    competition, area_groups = build_competition_mart(
        current, reference, [str(value) for value in config["density_groups"]]
    )
    context = build_context_mart(
        commercial, segments, eligibility, reference, recent_count,
        int(analysis["context_quarters"]), config["context_metrics"],
    )

    evaluation_path = Path(model_evaluation_path).resolve() if model_evaluation_path else (
        paths["model_dir"] / f"ebm_evaluation_{reference}.joblib"
    )
    model_results: dict[str, Any] = {}
    if evaluation_path.exists():
        loaded = joblib.load(evaluation_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"모델 평가 파일 형식이 올바르지 않습니다: {evaluation_path}")
        model_results = loaded

    marts = {
        "trend": build_trend_mart(commercial, reference, recent_count, industry_codes),
        "segments": build_segments_mart(segments, reference, industry_codes),
        "competition": competition,
        "context": context,
    }
    marts["summary"] = build_summary_mart(
        commercial, master, eligibility, context, area_groups, reference, recent_count,
        industry_codes, model_results.get("predictions"),
    )
    empty_frames = {
        "model_metrics": pd.DataFrame(columns=MART_KEYS["model_metrics"]),
        "model_global": pd.DataFrame(columns=MART_KEYS["model_global"]),
        "model_local": pd.DataFrame(columns=MART_KEYS["model_local"]),
    }
    for name, empty in empty_frames.items():
        marts[name] = model_results.get(name, empty)

    output_paths: dict[str, Path] = {}
    for name, frame in marts.items():
        validate_mart(frame, name)
        destination = mart_dir / config["mart_files"][name]
        _atomic_write_csv(frame, destination)
        output_paths[name] = destination

    source_paths = [paths["commercial_quarter"], paths["trade_area_master"], paths["segment_comparison"]]
    manifest = {
        "schema_version": str(config["schema_version"]),
        "execution_id": model_results.get("execution_id"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_quarter": reference,
        "source_rows": {
            "commercial_quarter": len(commercial), "trade_area_master": len(master),
            "segment_comparison": len(segments),
        },
        "source_sha256": {path.name: _file_hash(path) for path in source_paths},
        "model_version": model_results.get("model_version"),
        "model_evaluation_file": str(evaluation_path) if evaluation_path.exists() else None,
        "mart_rows": {name: len(frame) for name, frame in marts.items()},
    }
    manifest_path = mart_dir / "manifest.json"
    _atomic_write_json(manifest, manifest_path)
    output_paths["manifest"] = manifest_path
    return output_paths


def main() -> None:
    """분석 마트 생성 CLI 진입점."""
    parser = argparse.ArgumentParser(description="대시보드용 서울 상권 분석 마트를 생성합니다.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--reference-quarter")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-evaluation")
    args = parser.parse_args()
    outputs = build_analysis_marts(
        args.config, args.reference_quarter, args.output_dir, args.model_evaluation
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
