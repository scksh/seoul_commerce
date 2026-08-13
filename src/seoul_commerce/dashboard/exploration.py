"""Pure calculations for the map-centered exploration screen."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LATEST_QUARTER = "20261"
ALL_DISTRICTS = "서울시 전체"
ALL_ADMIN_DONGS = "전체 행정동"
ALL_TRADE_AREAS = "전체 상권"
ALL_AREA_TYPES = "전체"


@dataclass(frozen=True)
class MetricSpec:
    """Columns and display rules for one representative metric."""

    label: str
    value_column: str
    unit: str
    qoq_column: str | None = None
    yoy_column: str | None = None


METRICS: dict[str, MetricSpec] = {
    "점포수": MetricSpec(
        label="점포수",
        value_column="total_store_count",
        unit="개",
        qoq_column="store_qoq_rate",
        yoy_column="store_yoy_rate",
    ),
    "추정매출": MetricSpec(
        label="추정매출",
        value_column="quarterly_sales_amount",
        unit="원",
        qoq_column="sales_qoq_rate",
        yoy_column="sales_yoy_rate",
    ),
    "월평균 점포당 추정매출": MetricSpec(
        label="월평균 점포당 추정매출",
        value_column="monthly_average_sales_per_store",
        unit="원",
        qoq_column="sales_per_store_qoq_rate",
        yoy_column="sales_per_store_yoy_rate",
    ),
    "유동인구": MetricSpec(
        label="유동인구",
        value_column="total_floating_population",
        unit="명",
        qoq_column="floating_qoq_rate",
        yoy_column="floating_yoy_rate",
    ),
    "주거인구": MetricSpec(
        label="주거인구",
        value_column="total_resident_population",
        unit="명",
    ),
}

COMPARISON_COLUMNS = {
    "절대값": None,
    "전분기 대비": "qoq_column",
    "전년 동기 대비": "yoy_column",
}


@dataclass(frozen=True)
class ExplorationFilters:
    district: str
    admin_dong: str
    trade_area: str
    area_type: str
    industry: str
    metric: str
    comparison: str


@dataclass(frozen=True)
class ExplorationView:
    frame: pd.DataFrame
    top10: pd.DataFrame
    filters: ExplorationFilters
    metric: MetricSpec
    ranking_column: str
    ranking_label: str
    state_key: str


def available_comparisons(metric_label: str) -> tuple[str, ...]:
    """Return comparison choices supported by the selected metric."""
    metric = METRICS[metric_label]
    choices = ["절대값"]
    if metric.qoq_column:
        choices.append("전분기 대비")
    if metric.yoy_column:
        choices.append("전년 동기 대비")
    return tuple(choices)


def format_quarter(value: str) -> str:
    """Format a YYYYQ-like source code such as 20261 for display."""
    text = str(value)
    if len(text) == 5 and text.isdigit():
        return f"{text[:4]}년 {text[4]}분기"
    return text


def format_metric_value(value: object, metric: MetricSpec) -> str:
    """Format a representative metric for compact cards and labels."""
    if pd.isna(value):
        return "데이터 없음"
    number = float(value)
    if metric.unit == "원":
        if abs(number) >= 100_000_000:
            return f"{number / 100_000_000:,.1f}억 원"
        if abs(number) >= 10_000:
            return f"{number / 10_000:,.0f}만 원"
        return f"{number:,.0f}원"
    return f"{number:,.0f}{metric.unit}"


def format_change(value: object) -> str:
    """Format a rate stored in percentage-point units."""
    if pd.isna(value):
        return "비교 데이터 없음"
    return f"{float(value):+,.1f}%"


def build_exploration(
    summary: pd.DataFrame,
    filters: ExplorationFilters,
) -> ExplorationView:
    """Filter the summary mart and prepare ranking and map columns."""
    metric = METRICS[filters.metric]
    frame = summary.loc[summary["base_quarter"].astype(str) == LATEST_QUARTER].copy()
    frame = frame.loc[frame["industry_name"] == filters.industry]
    if filters.district != ALL_DISTRICTS:
        frame = frame.loc[frame["district_name"] == filters.district]
    if filters.admin_dong != ALL_ADMIN_DONGS:
        frame = frame.loc[frame["admin_dong_code"] == filters.admin_dong]
    if filters.area_type != ALL_AREA_TYPES:
        frame = frame.loc[frame["trade_area_type_name"] == filters.area_type]

    frame = frame.loc[frame[["longitude", "latitude"]].notna().all(axis=1)].copy()
    frame["trade_area_code"] = frame["trade_area_code"].astype(str)

    comparison_attribute = COMPARISON_COLUMNS[filters.comparison]
    ranking_column = (
        metric.value_column
        if comparison_attribute is None
        else getattr(metric, comparison_attribute)
    )
    if ranking_column is None:
        raise ValueError(f"{metric.label}은(는) {filters.comparison} 비교를 지원하지 않습니다.")

    frame["metric_value"] = pd.to_numeric(frame[metric.value_column], errors="coerce")
    frame["ranking_value"] = pd.to_numeric(frame[ranking_column], errors="coerce")
    frame["metric_display"] = frame["metric_value"].map(
        lambda value: format_metric_value(value, metric)
    )
    if filters.comparison == "절대값":
        frame["ranking_display"] = frame["metric_display"]
    else:
        frame["ranking_display"] = frame["ranking_value"].map(format_change)

    ranked = (
        frame.dropna(subset=["ranking_value"])
        .sort_values(
            ["ranking_value", "trade_area_name", "trade_area_code"],
            ascending=[False, True, True],
        )
        .copy()
    )
    ranked["rank"] = pd.array(range(1, len(ranked) + 1), dtype="Int64")
    rank_by_code = ranked.set_index("trade_area_code")["rank"]
    frame["rank"] = frame["trade_area_code"].map(rank_by_code).astype("Int64")
    top10 = ranked.head(10).copy()
    if filters.trade_area != ALL_TRADE_AREAS:
        top10 = ranked.loc[ranked["trade_area_code"] == filters.trade_area].head(1).copy()

    state_key = "|".join(
        (
            LATEST_QUARTER,
            filters.district,
            filters.admin_dong,
            filters.trade_area,
            filters.area_type,
            filters.industry,
            filters.metric,
            filters.comparison,
        )
    )
    return ExplorationView(
        frame=frame,
        top10=top10,
        filters=filters,
        metric=metric,
        ranking_column=ranking_column,
        ranking_label=(metric.label if filters.comparison == "절대값" else filters.comparison),
        state_key=state_key,
    )
