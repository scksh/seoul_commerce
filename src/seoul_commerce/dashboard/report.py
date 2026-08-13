"""Calculations and rendering for the selected trade-area analysis report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st

from seoul_commerce.dashboard.data import ReportMarts
from seoul_commerce.dashboard.exploration import LATEST_QUARTER, format_change, format_quarter


DENSITY_LABELS = {
    "low": "낮음",
    "lower_middle": "다소 낮음",
    "middle": "보통",
    "upper_middle": "다소 높음",
    "high": "높음",
}
DENSITY_ORDER = tuple(DENSITY_LABELS)
AVAILABILITY_LABELS = {
    "available": "분석 가능",
    "incomplete_commercial_history": "연속 8개 분기 이력 부족",
}
GROWTH_TYPE_LABELS = {
    "decline_or_stagnation": "감소·정체형",
    "joint_growth": "동반 성장형",
    "store_expansion": "점포 확장형",
    "ticket_growth": "객단가 성장형",
    "transaction_growth": "거래량 성장형",
    "unavailable": "분석 불가",
}
SEGMENT_LABELS = {
    "gender": "성별",
    "age": "연령",
    "weekday": "요일",
    "time_band": "시간대",
}
SEGMENT_ORDER = tuple(SEGMENT_LABELS)
DOMAIN_LABELS = {
    "usage_time": "이용 시간",
    "hinterland": "배후 수요",
    "housing": "주거",
    "access": "접근성",
    "competition": "경쟁",
}
DOMAIN_COLORS = {
    "이용 시간": "#277DA1",
    "배후 수요": "#43AA8B",
    "주거": "#90BE6D",
    "접근성": "#F9C74F",
    "경쟁": "#F9844A",
}
CHART_BLUE = "#1687FF"
CHART_BLUE_DARK = "#0866D9"
CHART_TEAL = "#2A9D8F"
CHART_CORAL = "#E76F51"
CHART_GOLD = "#E9B949"
CHART_INK = "#334155"
CHART_MUTED = "#64748B"
CHART_GRID = "#E8EEF5"
CONTEXT_METRIC_LABELS = {
    "weekend_floating_share": "주말 유동인구 비중",
    "lunch_floating_share": "점심 유동인구 비중",
    "evening_floating_share": "저녁 유동인구 비중",
    "late_night_floating_share": "심야 유동인구 비중",
    "resident_population": "상주인구",
    "working_population": "직장인구",
    "resident_to_working_ratio": "상주/직장 인구 비율",
    "household_count": "가구 수",
    "apartment_household_share": "아파트 가구 비중",
    "facility_density": "집객시설 밀도",
    "subway_station_count": "지하철역 수",
    "bus_stop_count": "버스정류장 수",
    "store_density": "점포 밀도",
    "franchise_share": "프랜차이즈 비중",
}
TERM_LABELS = {
    "log_floating_population": "유동인구",
    "log_resident_population": "상주인구",
    "log_working_population": "직장인구",
    "log_household_count": "가구 수",
    "weekend_floating_share": "주말 유동인구 비중",
    "lunch_floating_share": "점심 유동인구 비중",
    "evening_floating_share": "저녁 유동인구 비중",
    "late_night_floating_share": "심야 유동인구 비중",
    "log_area_ha": "상권 면적",
    "log_facility_density": "집객시설 밀도",
    "subway_station_count": "지하철역 수",
    "bus_stop_count": "버스정류장 수",
    "log_store_density": "점포 밀도",
    "franchise_share": "프랜차이즈 비중",
}

GROWTH_FORMULAS = r"""
**기호**  
`S`: 분기 총매출 · `N`: 점포 수 · `C`: 분기 점포당 거래건수 · `A`: 평균 객단가

$$S=N\times C\times A$$

- 점포당 분기 매출: $E=S/N=C\times A$
- 월평균 점포당 매출: $E/3$
- 점포 수 기여도: $100\ln(N_t/N_{t-4})$
- 거래량 기여도: $100\ln(C_t/C_{t-4})$
- 객단가 기여도: $100\ln(A_t/A_{t-4})$
- 총매출 로그 변화: 세 기여도의 합
- 8분기 분기 추세율: 로그 총매출 회귀기울기 $b$를 $100(\exp(b)-1)$로 변환
- 변동성: 최근 8분기 총매출 전년 동기 증감률의 표준편차

로그 성장 기여도는 관측된 매출 변화의 회계적 분해이며 인과효과가 아닙니다.
"""

CUSTOMER_FORMULAS = r"""
각 비중은 **기준분기·선택 상권**에서 구성 유형별로 따로 계산합니다. 매출은 선택 업종 기준이며, 유동인구는 업종 구분이 없는 상권 전체 인구입니다.

- 매출 비중$_i$: $\frac{\text{선택 업종의 구성항목 }i\text{ 매출액}}{\sum_{j\in\text{같은 구성 유형}}\text{선택 업종의 구성항목 }j\text{ 매출액}}$
- 유동인구 비중$_i$: $\frac{\text{구성항목 }i\text{ 유동인구}}{\sum_{j\in\text{같은 구성 유형}}\text{구성항목 }j\text{ 유동인구}}$
- 예: `30대 매출 비중`은 30대 매출을 전체 연령대 매출 합으로 나눕니다. 성별·연령·요일·시간대는 서로 섞어 합산하지 않습니다.
- 수요–점포 성장 격차: `유동인구 전년 동기 증감률 - 점포 수 전년 동기 증감률`
- 구성 차이 지수: $\frac{1}{2}\sum|\text{매출 비중}-\text{유동인구 비중}|$
- 매출 과대표 비율: `매출 비중 ÷ 유동인구 비중`

구성항목 합계가 0이거나 원천값이 누락되면 비교에서 제외합니다. 과대표 비율은 실제 방문자의 구매 전환율이 아닙니다.
"""

COMPETITION_FORMULAS = r"""
- 점포 밀도: `동종 업종 점포 수 ÷ 상권 면적(ha)`
- 같은 분기·업종의 상권을 점포 밀도 순으로 정렬해 상권 수가 비슷한 5개 그룹으로 구분
- 각 그룹에서 총매출과 점포당 매출의 **중앙값**을 계산

다른 입지 조건을 통제하지 않은 기술적 비교이므로 점포 밀도의 인과효과로 해석하지 않습니다.
"""

CONTEXT_FORMULAS = r"""
- 백분위: 같은 업종·같은 공식 상권유형에서 값이 낮은 상권의 비율
- 집객시설 밀도: `집객시설 수 ÷ 상권 면적(ha)`
- 점포 밀도: `동종 업종 점포 수 ÷ 상권 면적(ha)`
- 상주/직장 인구 비율: `상주인구 ÷ 직장인구`
- 시간대 비중: 해당 시간대 유동인구를 전체 유동인구로 나눈 값

인구·가구·시설은 최근 4분기 중앙값, 시간대·경쟁 지표는 기준분기 값을 사용합니다.
"""

MODEL_FORMULAS = r"""
- 예측 대상: 로그 점포당 분기 매출
- 항 중요도 비중: 각 항의 중요도를 전체 항 중요도 합으로 나눈 상대 비중
- 선택 상권 예측 기여: $\exp(\text{로그 기여값})-1$
- 예측 오차율: $(\text{예측 점포당 매출}/\text{실제 점포당 매출}-1)\times100$

항 중요도는 매출 변화율이나 원인별 기여율이 아닙니다. 선택 상권의 기여값도 모델의 예측 근거이며 인과효과·통계적 유의성·출점 추천을 의미하지 않습니다.
"""


@dataclass(frozen=True)
class GrowthView:
    trend: pd.DataFrame
    contributions: pd.DataFrame
    growth_type_label: str
    growth_type_color: str
    summary_text: str
    per_store_summary_text: str
    current_total_sales_display: str
    current_monthly_sales_per_store_display: str
    total_sales_qoq_change: float | None
    total_sales_yoy_change: float | None
    sales_per_store_qoq_change: float | None
    sales_per_store_yoy_change: float | None
    total_sales_log_change: float | None
    quarterly_trend_rate: float | None
    sales_yoy_volatility: float | None
    recent_growth_count: float | None
    consecutive_growth_quarters: float | None


@dataclass(frozen=True)
class CustomerView:
    floating_yoy_change: float | None
    store_yoy_change: float | None
    demand_store_gap: float | None
    distances: pd.DataFrame
    overrepresented: pd.DataFrame
    summary_text: str


@dataclass(frozen=True)
class CompetitionView:
    selected_group: str
    summary: pd.DataFrame
    summary_text: str


@dataclass(frozen=True)
class ContextView:
    metrics: pd.DataFrame
    top_metrics: tuple[str, ...]
    summary_text: str


@dataclass(frozen=True)
class ModelView:
    candidate: str
    validation_log_r2: float | None
    validation_mape: float | None
    validation_rows: float | None
    training_period: str
    actual_sales_display: str
    predicted_sales_display: str
    prediction_error_rate: float | None
    global_terms: pd.DataFrame
    local_terms: pd.DataFrame
    summary_text: str


@dataclass(frozen=True)
class ReportView:
    """Display-ready values for one selected trade area and industry."""

    trade_area_code: str
    trade_area_name: str
    district_name: str
    admin_dong_name: str
    industry_code: str
    industry_name: str
    quarter_label: str
    report_available: bool
    availability_label: str
    monthly_sales_display: str
    yoy_change: float | None
    trend_change: float | None
    trend_change_display: str
    trend_direction: str
    peer_position_display: str
    peer_context_display: str
    density_display: str
    store_count_display: str
    summary_text: str
    tags: tuple[tuple[str, str], ...]
    review_points: tuple[str, ...]
    growth: GrowthView
    customer: CustomerView
    competition: CompetitionView
    context: ContextView
    model: ModelView


def build_report_view(
    summary: pd.DataFrame,
    marts: ReportMarts,
    trade_area_code: str,
    industry_name: str,
) -> ReportView:
    """Build every report section from saved dashboard marts."""
    current_rows = summary.loc[
        summary["base_quarter"].astype(str).eq(LATEST_QUARTER)
        & summary["trade_area_code"].astype(str).eq(str(trade_area_code))
        & summary["industry_name"].eq(industry_name)
    ]
    if current_rows.empty:
        raise ValueError("선택한 상권·업종의 최신 요약 데이터를 찾을 수 없습니다.")
    current = current_rows.iloc[0]
    industry_code = str(current["industry_code"])

    area_trend = _select_rows(
        marts.trend,
        trade_area_code=trade_area_code,
        industry_code=industry_code,
    ).sort_values("observed_quarter").tail(8)
    growth = _build_growth_view(current, area_trend)
    customer = _build_customer_view(
        current,
        _select_rows(
            marts.segments,
            trade_area_code=trade_area_code,
            industry_code=industry_code,
        ),
    )
    competition = _build_competition_view(
        current,
        _select_rows(marts.competition, industry_code=industry_code),
    )
    context = _build_context_view(
        _select_rows(
            marts.context,
            trade_area_code=trade_area_code,
            industry_code=industry_code,
        )
    )
    model = _build_model_view(
        current,
        marts.model_metrics,
        marts.model_global,
        _select_rows(
            marts.model_local,
            trade_area_code=trade_area_code,
            industry_code=industry_code,
        ),
    )

    trend_change = _first_to_last_change(growth.trend["monthly_sales_per_store"])
    trend_direction = _trend_direction(trend_change)
    yoy_change = _number(current.get("sales_per_store_yoy_rate"))
    peer_percentile = _number(current.get("sales_per_store_peer_percentile"))
    peer_count = _number(current.get("peer_trade_area_count"))
    density_group = str(current.get("density_group"))
    density_display = DENSITY_LABELS.get(density_group, "분석 불가")
    peer_position = _peer_position(peer_percentile, peer_count)
    report_available = bool(current.get("report_available", False))
    availability_reason = str(current.get("availability_reason", ""))

    report_summary = _summary_text(peer_position, trend_direction, density_display)
    tags = (
        (f"점포당 매출 {peer_position}", _peer_color(peer_percentile)),
        (f"최근 추세 {trend_direction}", _trend_color(trend_direction)),
        (f"경쟁 밀도 {density_display}", _density_color(density_group)),
    )
    return ReportView(
        trade_area_code=str(trade_area_code),
        trade_area_name=str(current["trade_area_name"]),
        district_name=str(current["district_name"]),
        admin_dong_name=str(current["admin_dong_name"]),
        industry_code=industry_code,
        industry_name=industry_name,
        quarter_label=format_quarter(LATEST_QUARTER),
        report_available=report_available,
        availability_label=AVAILABILITY_LABELS.get(
            availability_reason,
            availability_reason or "상태 정보 없음",
        ),
        monthly_sales_display=_format_won(current.get("monthly_average_sales_per_store")),
        yoy_change=yoy_change,
        trend_change=trend_change,
        trend_change_display=format_change(trend_change),
        trend_direction=trend_direction,
        peer_position_display=peer_position,
        peer_context_display=(
            f"{industry_name} · {current['trade_area_type_name']} "
            f"{int(peer_count):,}곳 중 {peer_position}"
            if peer_count is not None and peer_position != "비교 불가"
            else "비교집단 정보 없음"
        ),
        density_display=density_display,
        store_count_display=_format_count(current.get("total_store_count")),
        summary_text=report_summary,
        tags=tags,
        review_points=_review_points(trend_direction, density_display),
        growth=growth,
        customer=customer,
        competition=competition,
        context=context,
        model=model,
    )


def _select_rows(
    frame: pd.DataFrame,
    *,
    trade_area_code: str | None = None,
    industry_code: str | None = None,
) -> pd.DataFrame:
    mask = frame["base_quarter"].astype(str).eq(LATEST_QUARTER)
    if trade_area_code is not None:
        mask &= frame["trade_area_code"].astype(str).eq(str(trade_area_code))
    if industry_code is not None:
        mask &= frame["industry_code"].astype(str).eq(str(industry_code))
    return frame.loc[mask].copy()


def _build_growth_view(current: pd.Series, trend: pd.DataFrame) -> GrowthView:
    trend = trend.copy()
    numeric_columns = [
        "quarterly_sales_amount",
        "quarterly_sales_count",
        "total_store_count",
        "average_transaction_value",
    ]
    for column in numeric_columns:
        trend[column] = pd.to_numeric(trend[column], errors="coerce")
    valid_stores = trend["total_store_count"].where(trend["total_store_count"].gt(0))
    trend["monthly_sales_per_store"] = trend["quarterly_sales_amount"].div(valid_stores).div(3)
    trend["total_sales_hundred_million"] = trend["quarterly_sales_amount"].div(100_000_000)
    trend["monthly_sales_million"] = trend["monthly_sales_per_store"].div(1_000_000)
    trend["quarter_label"] = trend["observed_quarter"].astype(str).map(format_quarter)

    contribution_rows = [
        ("점포 수", _number(current.get("store_log_contribution")), "total"),
        ("점포당 거래건수", _number(current.get("volume_log_contribution")), "both"),
        ("평균 객단가", _number(current.get("ticket_log_contribution")), "both"),
    ]
    contributions = pd.DataFrame(
        contribution_rows,
        columns=["component", "log_contribution", "scope"],
    ).dropna(subset=["log_contribution"])
    contributions["color"] = contributions["log_contribution"].map(
        lambda value: CHART_TEAL if value >= 0 else CHART_CORAL
    )

    raw_growth_type = str(current.get("growth_type", "unavailable"))
    growth_type = GROWTH_TYPE_LABELS.get(raw_growth_type, "분석 불가")
    total_sales_yoy = _number(current.get("sales_yoy_rate"))
    if contributions.empty:
        summary = "성장 구성 데이터를 계산할 수 없습니다."
    else:
        largest = contributions.loc[contributions["log_contribution"].abs().idxmax()]
        direction = "증가" if largest["log_contribution"] >= 0 else "감소"
        summary = (
            f"총 추정매출은 전년 동기 대비 {format_change(total_sales_yoy)}이며 "
            f"{growth_type}입니다. 가장 크게 움직인 요인은 {largest['component']}의 {direction}입니다."
        )
    per_store_contributions = contributions.loc[contributions["scope"].eq("both")]
    per_store_yoy = _number(current.get("sales_per_store_yoy_rate"))
    if per_store_contributions.empty:
        per_store_summary = "점포당 매출의 변화 요인을 계산할 수 없습니다."
    else:
        largest_per_store = per_store_contributions.loc[
            per_store_contributions["log_contribution"].abs().idxmax()
        ]
        per_store_direction = (
            "증가" if largest_per_store["log_contribution"] >= 0 else "감소"
        )
        per_store_summary = (
            f"월평균 점포당 추정매출은 전년 동기 대비 {format_change(per_store_yoy)}입니다. "
            f"점포 한 곳의 매출 변화에서 가장 크게 움직인 요인은 "
            f"{largest_per_store['component']}의 {per_store_direction}입니다."
        )
    return GrowthView(
        trend=trend.reset_index(drop=True),
        contributions=contributions.reset_index(drop=True),
        growth_type_label=growth_type,
        growth_type_color=_growth_type_color(raw_growth_type),
        summary_text=summary,
        per_store_summary_text=per_store_summary,
        current_total_sales_display=_format_won(current.get("quarterly_sales_amount")),
        current_monthly_sales_per_store_display=_format_won(
            current.get("monthly_average_sales_per_store")
        ),
        total_sales_qoq_change=_number(current.get("sales_qoq_rate")),
        total_sales_yoy_change=total_sales_yoy,
        sales_per_store_qoq_change=_number(current.get("sales_per_store_qoq_rate")),
        sales_per_store_yoy_change=per_store_yoy,
        total_sales_log_change=_number(current.get("total_sales_log_change")),
        quarterly_trend_rate=_number(current.get("quarterly_trend_rate")),
        sales_yoy_volatility=_number(current.get("sales_yoy_volatility")),
        recent_growth_count=_number(current.get("recent_4q_sales_growth_count")),
        consecutive_growth_quarters=_number(current.get("consecutive_sales_growth_quarters")),
    )


def _build_customer_view(
    current: pd.Series,
    segments: pd.DataFrame,
) -> CustomerView:
    available = segments["comparison_available"].eq(True) | segments[
        "comparison_available"
    ].astype(str).str.lower().eq("true")
    segments = segments.loc[available].copy()
    for column in [
        "composition_distance",
        "sales_floating_share_ratio",
        "sales_amount_share",
        "floating_population_share",
    ]:
        segments[column] = pd.to_numeric(segments[column], errors="coerce")

    distances = (
        segments.groupby("segment_type", observed=True)["composition_distance"]
        .first()
        .rename("distance")
        .reset_index()
    )
    distances["segment_label"] = distances["segment_type"].map(SEGMENT_LABELS)
    distances["order"] = distances["segment_type"].map(
        {name: index for index, name in enumerate(SEGMENT_ORDER)}
    )
    distances = distances.dropna(subset=["segment_label", "distance"]).sort_values("order")

    ratios = segments.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["sales_floating_share_ratio"]
    )
    ratios = ratios.loc[ratios["sales_floating_share_ratio"].gt(0)].nlargest(
        6, "sales_floating_share_ratio"
    )
    ratios = ratios.assign(
        segment_label=ratios["segment_type"].map(SEGMENT_LABELS)
        + " · "
        + ratios["segment_name"].astype(str),
        sales_share_percent=ratios["sales_amount_share"] * 100,
        floating_share_percent=ratios["floating_population_share"] * 100,
    )
    ratios = ratios[
        [
            "segment_label",
            "sales_floating_share_ratio",
            "sales_share_percent",
            "floating_share_percent",
        ]
    ].sort_values("sales_floating_share_ratio")

    gap = _number(current.get("demand_store_yoy_gap"))
    if ratios.empty:
        summary = "방문과 매출 구성을 비교할 수 있는 항목이 없습니다."
    else:
        top = ratios.iloc[-1]
        if gap is None:
            gap_text = "유동인구와 점포 수의 성장 격차를 계산할 수 없었고"
        elif gap >= 0:
            gap_text = "유동인구가 점포 수보다 빠르게 증가했고"
        else:
            gap_text = "점포 수가 유동인구보다 빠르게 증가했고"
        summary = (
            f"전년 동기 대비 {gap_text}, 방문 비중보다 매출 비중이 가장 크게 나타난 "
            f"항목은 {top['segment_label']}입니다."
        )
    return CustomerView(
        floating_yoy_change=_number(current.get("floating_yoy_rate")),
        store_yoy_change=_number(current.get("store_yoy_rate")),
        demand_store_gap=gap,
        distances=distances[["segment_label", "distance"]].reset_index(drop=True),
        overrepresented=ratios.reset_index(drop=True),
        summary_text=summary,
    )


def _build_competition_view(current: pd.Series, competition: pd.DataFrame) -> CompetitionView:
    competition = competition.copy()
    numeric_columns = [
        "sample_count",
        "density_min",
        "density_median",
        "density_max",
        "total_sales_median",
        "sales_per_store_median",
    ]
    for column in numeric_columns:
        competition[column] = pd.to_numeric(competition[column], errors="coerce")
    selected_raw = str(current.get("density_group", ""))
    selected_group = DENSITY_LABELS.get(selected_raw, "분석 불가")
    competition["density_label"] = competition["density_group"].map(DENSITY_LABELS)
    competition["order"] = competition["density_group"].map(
        {name: index for index, name in enumerate(DENSITY_ORDER)}
    )
    competition["total_sales_hundred_million"] = competition["total_sales_median"].div(
        100_000_000
    )
    competition["sales_per_store_million"] = competition["sales_per_store_median"].div(
        1_000_000
    )
    competition["color"] = competition["density_group"].map(
        lambda group: CHART_CORAL if str(group) == selected_raw else CHART_BLUE_DARK
    )
    competition = competition.sort_values("order")
    if competition.empty:
        summary = "점포 밀도 그룹 비교 데이터가 없습니다."
    else:
        low = competition.iloc[0]
        high = competition.iloc[-1]
        total_multiple = _safe_multiple(
            high["total_sales_median"], low["total_sales_median"]
        )
        efficiency_multiple = _safe_multiple(
            high["sales_per_store_median"], low["sales_per_store_median"]
        )
        summary = (
            f"선택 상권은 점포 밀도 {selected_group} 그룹입니다. 밀도 최고 그룹의 중앙값은 "
            f"최저 그룹보다 총매출 {total_multiple:.1f}배, 점포당 매출 {efficiency_multiple:.1f}배입니다."
            if total_multiple is not None and efficiency_multiple is not None
            else f"선택 상권은 점포 밀도 {selected_group} 그룹입니다."
        )
    return CompetitionView(
        selected_group=selected_group,
        summary=competition.reset_index(drop=True),
        summary_text=summary,
    )


def _build_context_view(context: pd.DataFrame) -> ContextView:
    context = context.copy()
    context["value"] = pd.to_numeric(context["value"], errors="coerce")
    context["peer_percentile"] = pd.to_numeric(context["peer_percentile"], errors="coerce")
    context["domain_label"] = context["domain"].map(DOMAIN_LABELS)
    context["metric_label"] = context["metric"].map(CONTEXT_METRIC_LABELS)
    context["display_value"] = context.apply(_format_context_row, axis=1)
    context["color"] = context["domain_label"].map(DOMAIN_COLORS)
    context = context.dropna(subset=["domain_label", "metric_label", "peer_percentile"])
    context = context.sort_values("peer_percentile")
    top_metrics = tuple(
        context.nlargest(3, "peer_percentile")["metric_label"].astype(str).tolist()
    )
    if top_metrics:
        summary = "같은 업종·상권유형과 비교해 " + ", ".join(top_metrics) + " 조건이 두드러집니다."
    else:
        summary = "상권 환경을 비교할 수 있는 지표가 없습니다."
    return ContextView(
        metrics=context.reset_index(drop=True),
        top_metrics=top_metrics,
        summary_text=summary,
    )


def _build_model_view(
    current: pd.Series,
    metrics: pd.DataFrame,
    global_terms: pd.DataFrame,
    local_terms: pd.DataFrame,
) -> ModelView:
    execution_id = str(current.get("execution_id", ""))
    selected_metrics = metrics.loc[metrics["execution_id"].astype(str).eq(execution_id)].copy()
    selected_global = global_terms.loc[
        global_terms["execution_id"].astype(str).eq(execution_id)
    ].copy()
    selected_local = local_terms.loc[
        local_terms["execution_id"].astype(str).eq(execution_id)
    ].copy()
    if selected_metrics.empty:
        selected_metrics = metrics.copy()
    if selected_global.empty:
        selected_global = global_terms.copy()

    selected_metrics["validation_median_absolute_percentage_error"] = pd.to_numeric(
        selected_metrics["validation_median_absolute_percentage_error"], errors="coerce"
    )
    valid_metrics = selected_metrics.dropna(
        subset=["validation_median_absolute_percentage_error"]
    )
    metric_row = (
        valid_metrics.loc[valid_metrics["validation_median_absolute_percentage_error"].idxmin()]
        if not valid_metrics.empty
        else None
    )

    selected_global = selected_global.loc[selected_global["role"].eq("analysis_factor")].copy()
    selected_global["importance_share"] = pd.to_numeric(
        selected_global["importance_share"], errors="coerce"
    )
    selected_global["term_label"] = selected_global["term"].map(_term_label)
    global_chart = selected_global.nlargest(6, "importance_share")[
        ["term_label", "importance_share"]
    ].sort_values("importance_share")

    join_columns = selected_global[["term", "term_type", "role"]].drop_duplicates()
    selected_local = selected_local.merge(
        join_columns,
        on=["term", "term_type"],
        how="inner",
    )
    selected_local["sales_ratio_contribution"] = pd.to_numeric(
        selected_local["sales_ratio_contribution"], errors="coerce"
    )
    selected_local["term_label"] = selected_local["term"].map(_term_label)
    selected_local["absolute_contribution"] = selected_local[
        "sales_ratio_contribution"
    ].abs()
    local_chart = selected_local.nlargest(6, "absolute_contribution")[
        ["term_label", "sales_ratio_contribution"]
    ].sort_values("sales_ratio_contribution")
    local_chart["color"] = local_chart["sales_ratio_contribution"].map(
        lambda value: CHART_TEAL if value >= 0 else CHART_CORAL
    )

    actual = _number(current.get("actual_sales_per_store"))
    predicted = _number(current.get("predicted_sales_per_store"))
    raw_error_rate = _number(current.get("prediction_error_rate"))
    error_rate = None if raw_error_rate is None else raw_error_rate * 100
    if local_chart.empty:
        summary = "선택 상권의 모델 예측 근거를 표시할 수 없습니다."
    else:
        strongest = local_chart.loc[local_chart["sales_ratio_contribution"].abs().idxmax()]
        direction = "높이는" if strongest["sales_ratio_contribution"] >= 0 else "낮추는"
        summary = (
            f"선택 상권에서 예측값을 가장 크게 {direction} 방향으로 사용된 특성은 "
            f"{strongest['term_label']}입니다."
        )

    if metric_row is None:
        candidate = "정보 없음"
        log_r2 = None
        mape = None
        validation_rows = None
        training_period = "학습 기간 정보 없음"
    else:
        candidate = str(metric_row["candidate"])
        log_r2 = _number(metric_row.get("validation_log_r2"))
        raw_mape = _number(metric_row.get("validation_median_absolute_percentage_error"))
        mape = None if raw_mape is None else raw_mape * 100
        validation_rows = _number(metric_row.get("validation_rows"))
        training_period = (
            f"{format_quarter(str(metric_row['training_start_quarter']))}~"
            f"{format_quarter(str(metric_row['training_end_quarter']))} 학습 · "
            f"{format_quarter(str(metric_row['validation_quarter']))} 검증"
        )
    return ModelView(
        candidate=candidate,
        validation_log_r2=log_r2,
        validation_mape=mape,
        validation_rows=validation_rows,
        training_period=training_period,
        actual_sales_display=_format_won(actual),
        predicted_sales_display=_format_won(predicted),
        prediction_error_rate=error_rate,
        global_terms=global_chart.reset_index(drop=True),
        local_terms=local_chart.reset_index(drop=True),
        summary_text=summary,
    )


def render_report(view: ReportView, on_close: Callable[[], None]) -> None:
    """Render the complete scrolling report below the map."""
    with st.container(border=True, key="selected-area-report"):
        _render_report_header(view, on_close)
        _render_kpis(view)
        st.divider()
        _render_growth_section(view)
        st.divider()
        _render_customer_section(view)
        st.divider()
        _render_competition_section(view)
        st.divider()
        _render_context_section(view)
        st.divider()
        _render_model_section(view)
        st.divider()
        _render_review_section(view)
        st.caption(
            "추정매출과 비교지표는 후보를 좁히기 위한 참고정보이며, "
            "실제 수익성·임대료·운영역량을 보장하지 않습니다."
        )


def _render_report_header(view: ReportView, on_close: Callable[[], None]) -> None:
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="top",
        gap=None,
        key="report-header",
    ):
        with st.container(gap=None, width="stretch", key="report-title"):
            st.caption("분석 리포트")
            st.subheader(view.trade_area_name)
        st.button(
            "리포트 닫기",
            icon=":material/close:",
            key="close_selected_area_report",
            on_click=on_close,
            width="content",
        )
    st.caption(
        f"{view.district_name} · {view.admin_dong_name}  |  "
        f"{view.industry_name}  |  {view.quarter_label}"
    )
    status_color = "green" if view.report_available else "orange"
    st.badge(view.availability_label, color=status_color, icon=":material/database:")
    if not view.report_available:
        st.warning(
            "연속 이력이 충분하지 않아 일부 추세·비교 해석은 제한적으로 제공됩니다.",
            icon="⚠️",
        )
    st.markdown(f"**요약**  \n{view.summary_text}")


def _render_kpis(view: ReportView) -> None:
    metric_columns = st.columns(4, gap="small", vertical_alignment="top", border=True)
    with metric_columns[0]:
        st.metric(
            "월평균 점포당 추정매출",
            view.monthly_sales_display,
            delta=view.yoy_change,
            delta_color="normal",
            delta_arrow="auto",
            delta_description="전년 동기 대비",
            format="%+.1f%%",
            width="stretch",
        )
    with metric_columns[1]:
        st.metric(
            "최근 8분기 추세",
            view.trend_direction,
            delta=view.trend_change,
            delta_color="normal",
            delta_arrow="auto",
            delta_description="8분기 전 대비",
            format="%+.1f%%",
            width="stretch",
        )
    with metric_columns[2]:
        st.metric(
            "점포당 추정매출 상대 위치",
            view.peer_position_display,
            delta=view.peer_context_display,
            delta_color="off",
            delta_arrow="off",
            width="stretch",
        )
    with metric_columns[3]:
        st.metric(
            "경쟁 수준",
            view.density_display,
            delta=f"현재 점포 {view.store_count_display}",
            delta_color="off",
            delta_arrow="off",
            width="stretch",
        )
    with st.container(horizontal=True, gap="small"):
        for label, color in view.tags:
            st.badge(label, color=color)


@st.fragment
def _render_growth_section(view: ReportView) -> None:
    _render_section_header(
        "성장",
        "상권 전체 시장의 변화와 점포 한 곳의 평균 매출 흐름을 구분해 확인합니다.",
        GROWTH_FORMULAS,
        "growth_formula",
    )
    metric_scope = st.segmented_control(
        "성장 지표",
        ["총매출", "점포당 매출"],
        default="총매출",
        required=True,
        key=f"growth_scope_{view.trade_area_code}_{view.industry_code}",
        width="content",
    )
    metric_scope = str(metric_scope or "총매출")

    if metric_scope == "총매출":
        metric_label = "분기 총 추정매출"
        metric_value = view.growth.current_total_sales_display
        yoy_change = view.growth.total_sales_yoy_change
        qoq_change = view.growth.total_sales_qoq_change
        insight = view.growth.summary_text
        insight_badge = view.growth.growth_type_label
        insight_badge_color = view.growth.growth_type_color
    else:
        metric_label = "월평균 점포당 추정매출"
        metric_value = view.growth.current_monthly_sales_per_store_display
        yoy_change = view.growth.sales_per_store_yoy_change
        qoq_change = view.growth.sales_per_store_qoq_change
        insight = view.growth.per_store_summary_text
        if yoy_change is None:
            insight_badge = "점포당 매출 비교 불가"
            insight_badge_color = "gray"
        elif yoy_change > 0:
            insight_badge = "점포당 매출 증가"
            insight_badge_color = "green"
        elif yoy_change < 0:
            insight_badge = "점포당 매출 감소"
            insight_badge_color = "red"
        else:
            insight_badge = "점포당 매출 정체"
            insight_badge_color = "blue"

    with st.container(border=True, gap="small", key="growth-primary-metrics"):
        st.caption(metric_label)
        st.markdown(f"### {metric_label}은 :blue[{metric_value}]입니다.")
        with st.container(
            horizontal=True,
            horizontal_alignment="center",
            vertical_alignment="center",
            gap="large",
        ):
            st.markdown(f"전년 동기 대비  {_change_markup(yoy_change)}")
            st.markdown(f"전분기 대비  {_change_markup(qoq_change)}")

    growth_columns = st.columns(3, gap="small", border=True)
    with growth_columns[0]:
        st.metric(
            "8분기 평균 분기 추세",
            _format_percent(view.growth.quarterly_trend_rate),
            width="stretch",
        )
    with growth_columns[1]:
        st.metric(
            "전년 동기 증감 변동성",
            _format_percentage_points(view.growth.sales_yoy_volatility),
            width="stretch",
        )
    with growth_columns[2]:
        recent = _format_integer(view.growth.recent_growth_count, "회")
        consecutive = _format_integer(view.growth.consecutive_growth_quarters, "분기")
        st.metric(
            "최근 성장 지속성",
            f"최근 4분기 중 {recent}",
            delta=f"연속 증가 {consecutive}",
            delta_color="off",
            delta_arrow="off",
            width="stretch",
        )

    _render_insight(insight, insight_badge, insight_badge_color, "growth-insight")

    chart_column, contribution_column = st.columns([0.65, 0.35], gap="medium")
    with chart_column:
        with st.container(border=True, height="stretch"):
            if metric_scope == "총매출":
                title = "최근 8분기 총 추정매출 추세"
                caption = "상권 전체의 분기 추정매출 · 단위: 억원"
                y_column = "total_sales_hundred_million"
                y_label = "분기 총 추정매출(억원)"
            else:
                title = "최근 8분기 점포당 추정매출 추세"
                caption = "분기 추정매출을 점포 수와 3개월로 나눈 월평균 값 · 단위: 백만원"
                y_column = "monthly_sales_million"
                y_label = "월평균 점포당 추정매출(백만원)"
            st.markdown(f"#### {title}")
            st.caption(caption)
            chart_data = view.growth.trend.dropna(subset=[y_column])
            if chart_data.empty:
                st.info("선택 조건에 해당하는 최근 분기 추세 데이터가 없습니다.")
            else:
                st.altair_chart(_growth_trend_chart(chart_data, y_column, y_label))
    with contribution_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 전년 동기 대비 성장 기여도")
            st.caption(
                "총매출은 세 요인, 점포당 매출은 거래건수·객단가 두 요인으로 해석합니다."
            )
            contributions = view.growth.contributions
            if metric_scope == "점포당 매출":
                contributions = contributions.loc[contributions["scope"].eq("both")]
            if contributions.empty:
                st.info("성장 기여도 데이터가 없습니다.")
            else:
                st.altair_chart(_contribution_chart(contributions))

def _render_customer_section(view: ReportView) -> None:
    _render_section_header(
        "고객·소비",
        "유동인구 증가와 점포 증가의 차이, 방문 구성과 실제 매출 구성의 차이를 확인합니다.",
        CUSTOMER_FORMULAS,
        "customer_formula",
    )
    metric_columns = st.columns(3, gap="small", border=True)
    with metric_columns[0]:
        st.metric(
            "유동인구 전년 동기 대비",
            _format_percent(view.customer.floating_yoy_change),
            width="stretch",
        )
    with metric_columns[1]:
        st.metric(
            "점포 수 전년 동기 대비",
            _format_percent(view.customer.store_yoy_change),
            width="stretch",
        )
    with metric_columns[2]:
        st.metric(
            "수요–점포 성장 격차",
            _format_percentage_points(view.customer.demand_store_gap, signed=True),
            width="stretch",
        )

    gap = view.customer.demand_store_gap
    if gap is None:
        customer_badge = "성장 격차 비교 불가"
        customer_badge_color = "gray"
    elif gap > 0:
        customer_badge = "수요 증가 우위"
        customer_badge_color = "green"
    elif gap < 0:
        customer_badge = "점포 증가 우위"
        customer_badge_color = "orange"
    else:
        customer_badge = "수요·점포 증가 동일"
        customer_badge_color = "blue"
    _render_insight(
        view.customer.summary_text,
        customer_badge,
        customer_badge_color,
        "customer-insight",
    )

    distance_column, ratio_column = st.columns([0.42, 0.58], gap="medium")
    with distance_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 방문–매출 구성 차이")
            st.caption("0에 가까울수록 유동인구와 매출 구성이 비슷합니다.")
            if view.customer.distances.empty:
                st.info("구성 차이 데이터가 없습니다.")
            else:
                st.altair_chart(_distance_chart(view.customer.distances))
    with ratio_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 매출 비중이 상대적으로 높은 항목")
            st.caption("1배보다 크면 유동인구 비중보다 매출 비중이 높습니다.")
            if view.customer.overrepresented.empty:
                st.info("방문–매출 구성 비교 데이터가 없습니다.")
            else:
                st.altair_chart(_ratio_chart(view.customer.overrepresented))
def _render_competition_section(view: ReportView) -> None:
    _render_section_header(
        "경쟁",
        "같은 업종의 점포 밀도 5개 그룹에서 시장 규모와 점포당 매출 중앙값을 함께 봅니다.",
        COMPETITION_FORMULAS,
        "competition_formula",
    )
    if view.competition.summary.empty:
        st.info("점포 밀도 그룹 비교 데이터가 없습니다.")
        return
    selected_rows = view.competition.summary.loc[
        view.competition.summary["density_label"].eq(view.competition.selected_group)
    ]
    selected_density = selected_rows.iloc[0] if not selected_rows.empty else pd.Series(dtype=object)
    competition_columns = st.columns(4, gap="small", border=True)
    with competition_columns[0]:
        st.metric(
            "선택 점포 밀도 그룹",
            view.competition.selected_group,
            delta=(
                f"비교 표본 {_format_integer(selected_density.get('sample_count'), '곳')}"
            ),
            delta_color="off",
            delta_arrow="off",
            width="stretch",
        )
    with competition_columns[1]:
        st.metric(
            "그룹 점포 밀도 중앙값",
            f"{_format_decimal(selected_density.get('density_median'), 2)}개/ha",
            width="stretch",
        )
    with competition_columns[2]:
        st.metric(
            "그룹 분기 총매출 중앙값",
            _format_won(selected_density.get("total_sales_median")),
            width="stretch",
        )
    with competition_columns[3]:
        st.metric(
            "그룹 분기 점포당 매출 중앙값",
            _format_won(selected_density.get("sales_per_store_median")),
            width="stretch",
        )
    _render_insight(
        view.competition.summary_text,
        f"점포 밀도 {view.competition.selected_group}",
        "orange",
        "competition-insight",
    )

    total_column, efficiency_column = st.columns(2, gap="medium")
    with total_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 점포 밀도별 총매출 중앙값")
            st.altair_chart(
                _density_chart(
                    view.competition.summary,
                    "total_sales_hundred_million",
                    "총 추정매출 중앙값(억원)",
                )
            )
    with efficiency_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 점포 밀도별 점포당 매출 중앙값")
            st.altair_chart(
                _density_chart(
                    view.competition.summary,
                    "sales_per_store_million",
                    "점포당 분기 추정매출 중앙값(백만원)",
                )
            )
    st.caption(
        "점포 밀도 그룹 간 관찰된 차이이며, 점포가 많아지면 매출이 증가한다는 뜻이 아닙니다."
    )


def _render_context_section(view: ReportView) -> None:
    _render_section_header(
        "상권 환경",
        "이용 시간·배후 수요·주거·접근성·경쟁 조건을 같은 업종·상권유형 안에서 비교합니다.",
        CONTEXT_FORMULAS,
        "context_formula",
    )
    if view.context.metrics.empty:
        st.info("상권 환경 비교 데이터가 없습니다.")
        return
    top_context = view.context.metrics.nlargest(3, "peer_percentile")
    context_columns = st.columns(len(top_context), gap="small", border=True)
    for column, (_, metric) in zip(context_columns, top_context.iterrows(), strict=True):
        with column:
            st.metric(
                str(metric["metric_label"]),
                str(metric["display_value"]),
                delta=f"비교집단 백분위 {float(metric['peer_percentile']):.1f}",
                delta_color="off",
                delta_arrow="off",
                width="stretch",
            )
    _render_insight(
        view.context.summary_text,
        "두드러진 환경 조건",
        "blue",
        "context-insight",
    )

    with st.container(border=True):
        st.markdown("#### 동일 업종·상권유형 내 상대 위치")
        st.caption("100에 가까울수록 비교집단에서 해당 환경지표 값이 높은 편입니다.")
        st.altair_chart(_context_chart(view.context.metrics))
    st.caption(
        "집객시설 등 일부 환경 자료는 기준시점이 오래될 수 있으므로 실제 시설 현황을 함께 확인하세요."
    )


def _render_model_section(view: ReportView) -> None:
    _render_section_header(
        "분석 근거",
        "설명 가능한 머신러닝이 점포당 매출 차이를 설명할 때 활용한 특성과 선택 상권의 예측 근거입니다.",
        MODEL_FORMULAS,
        "model_formula",
    )
    model_columns = st.columns(4, gap="small", border=True)
    with model_columns[0]:
        st.metric("선택 모델", f"후보 {view.model.candidate}", width="stretch")
    with model_columns[1]:
        st.metric(
            "검증 로그 R²",
            _format_decimal(view.model.validation_log_r2, 3),
            delta=(
                f"검증 표본 {_format_integer(view.model.validation_rows, '개')}"
                if view.model.validation_rows is not None
                else "검증 표본 정보 없음"
            ),
            delta_color="off",
            delta_arrow="off",
            width="stretch",
        )
    with model_columns[2]:
        st.metric(
            "중앙 절대 오차율",
            _format_percent(view.model.validation_mape, signed=False),
            width="stretch",
        )
    with model_columns[3]:
        st.metric(
            "실제 분기 점포당 매출",
            view.model.actual_sales_display,
            delta=(
                f"예측 {view.model.predicted_sales_display} · 오차 "
                f"{_format_percent(view.model.prediction_error_rate)}"
            ),
            delta_color="off",
            delta_arrow="off",
            width="stretch",
        )
    st.caption(view.model.training_period)
    _render_insight(
        view.model.summary_text,
        "모델 기반 설명",
        "blue",
        "model-insight",
    )

    global_column, local_column = st.columns(2, gap="medium")
    with global_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 전체 모델의 주요 분석 특성")
            st.caption("전체 표본의 예측에서 상대적으로 많이 활용한 특성입니다.")
            if view.model.global_terms.empty:
                st.info("전체 모델 설명 데이터가 없습니다.")
            else:
                st.altair_chart(_global_model_chart(view.model.global_terms))
    with local_column:
        with st.container(border=True, height="stretch"):
            st.markdown("#### 선택 상권의 주요 예측 기여")
            st.caption("초록은 예측값을 높이는 방향, 빨강은 낮추는 방향입니다.")
            if view.model.local_terms.empty:
                st.info("선택 상권 모델 설명 데이터가 없습니다.")
            else:
                st.altair_chart(_local_model_chart(view.model.local_terms))
    st.warning(
        "모델 중요도와 예측 기여값은 매출의 원인이나 출점 추천을 의미하지 않습니다.",
        icon=":material/info:",
    )


def _render_review_section(view: ReportView) -> None:
    with st.container(border=True):
        st.markdown("#### 현장 확인 포인트")
        for point in view.review_points:
            st.markdown(f"- {point}")


def _render_section_header(
    title: str,
    caption: str,
    formula_markdown: str,
    key: str,
) -> None:
    with st.container(
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
        gap="small",
    ):
        with st.container(width="stretch", gap=None):
            st.subheader(title)
            st.caption(caption)
        with st.popover(
            "계산식",
            icon=":material/function:",
            type="tertiary",
            key=key,
        ):
            st.markdown(formula_markdown)


def _render_insight(text: str, badge: str, color: str, key: str) -> None:
    """Render a consistent insight card immediately before the charts."""
    with st.container(border=True, key=key):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(badge, color=color)
            st.markdown(f":blue[**분석 인사이트**]  {text}")


def _growth_trend_chart(
    frame: pd.DataFrame,
    value_column: str,
    value_title: str,
) -> alt.Chart:
    quarter_order = frame["quarter_label"].astype(str).tolist()
    base = alt.Chart(frame).encode(
        x=alt.X(
            "quarter_label:N",
            title="분기",
            sort=quarter_order,
            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=95),
        ),
        y=alt.Y(
            f"{value_column}:Q",
            title=value_title,
            scale=alt.Scale(zero=False),
        ),
        tooltip=[
            alt.Tooltip("quarter_label:N", title="분기"),
            alt.Tooltip(f"{value_column}:Q", title=value_title, format=",.1f"),
        ],
    )
    area = base.mark_area(color=CHART_BLUE, opacity=0.09)
    line = base.mark_line(color=CHART_BLUE_DARK, strokeWidth=3)
    points = base.mark_point(
        color=CHART_BLUE_DARK,
        fill="white",
        size=72,
        strokeWidth=2,
    )
    latest_data = frame.tail(1)
    latest = (
        alt.Chart(latest_data)
        .mark_point(color=CHART_BLUE, filled=True, size=150, stroke="white", strokeWidth=2)
        .encode(
            x=alt.X("quarter_label:N", sort=quarter_order),
            y=alt.Y(f"{value_column}:Q"),
            tooltip=[
                alt.Tooltip("quarter_label:N", title="분기"),
                alt.Tooltip(f"{value_column}:Q", title=value_title, format=",.1f"),
            ],
        )
    )
    latest_label = (
        alt.Chart(latest_data)
        .mark_text(color=CHART_BLUE_DARK, dy=-14, fontWeight="bold", fontSize=12)
        .encode(
            x=alt.X("quarter_label:N", sort=quarter_order),
            y=alt.Y(f"{value_column}:Q"),
            text=alt.Text(f"{value_column}:Q", format=",.1f"),
        )
    )
    return _style_chart(
        (area + line + points + latest + latest_label).properties(height=290)
    )


def _contribution_chart(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "component:N",
            title=None,
            sort=None,
            axis=alt.Axis(labelLimit=120),
        ),
        x=alt.X("log_contribution:Q", title="로그 성장 기여도"),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[
            alt.Tooltip("component:N", title="구성 요소"),
            alt.Tooltip("log_contribution:Q", title="기여도", format="+.1f"),
        ],
    )
    bars = base.mark_bar(cornerRadius=5, size=28)
    positive_labels = (
        alt.Chart(frame.loc[frame["log_contribution"].ge(0)])
        .mark_text(align="left", dx=7, color=CHART_INK, fontWeight="bold")
        .encode(
            y=alt.Y("component:N", sort=None),
            x=alt.X("log_contribution:Q"),
            text=alt.Text("log_contribution:Q", format="+.1f"),
        )
    )
    negative_labels = (
        alt.Chart(frame.loc[frame["log_contribution"].lt(0)])
        .mark_text(align="right", dx=-7, color=CHART_INK, fontWeight="bold")
        .encode(
            y=alt.Y("component:N", sort=None),
            x=alt.X("log_contribution:Q"),
            text=alt.Text("log_contribution:Q", format="+.1f"),
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=CHART_MUTED, strokeWidth=1)
        .encode(x="x:Q")
    )
    return _style_chart(
        (bars + positive_labels + negative_labels + rule).properties(height=290)
    )


def _distance_chart(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(
        y=alt.Y("segment_label:N", title=None, sort=None),
        x=alt.X(
            "distance:Q",
            title="구성 차이 지수",
            scale=alt.Scale(domain=[0, 1]),
        ),
        tooltip=[
            alt.Tooltip("segment_label:N", title="구성 기준"),
            alt.Tooltip("distance:Q", title="차이 지수", format=".3f"),
        ],
    )
    bars = base.mark_bar(color=CHART_BLUE, cornerRadiusEnd=5, size=24)
    labels = base.mark_text(align="left", dx=7, color=CHART_INK).encode(
        text=alt.Text("distance:Q", format=".3f")
    )
    return _style_chart((bars + labels).properties(height=alt.Step(48)))


def _ratio_chart(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(
        y=alt.Y("segment_label:N", title=None, sort=None),
        x=alt.X(
            "sales_floating_share_ratio:Q",
            title="매출 비중 ÷ 유동인구 비중(배)",
        ),
        tooltip=[
            alt.Tooltip("segment_label:N", title="구성 항목"),
            alt.Tooltip("sales_floating_share_ratio:Q", title="과대표 비율", format=".2f"),
            alt.Tooltip("sales_share_percent:Q", title="매출 비중(%)", format=".1f"),
            alt.Tooltip("floating_share_percent:Q", title="유동인구 비중(%)", format=".1f"),
        ],
    )
    bars = base.mark_bar(color=CHART_GOLD, cornerRadiusEnd=5, size=22)
    labels = base.mark_text(align="left", dx=7, color=CHART_INK).encode(
        text=alt.Text("sales_floating_share_ratio:Q", format=".2f")
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [1]}))
        .mark_rule(color=CHART_MUTED, strokeDash=[5, 4])
        .encode(x="x:Q")
    )
    return _style_chart((bars + labels + rule).properties(height=alt.Step(34)))


def _density_chart(frame: pd.DataFrame, value_column: str, value_title: str) -> alt.Chart:
    base = alt.Chart(frame).encode(
        x=alt.X(
            "density_label:N",
            title="점포 밀도 수준",
            sort=None,
            axis=alt.Axis(labelAngle=0, labelOverlap=False, labelLimit=90),
        ),
        y=alt.Y(f"{value_column}:Q", title=value_title),
        color=alt.Color("color:N", scale=None, legend=None),
        tooltip=[
            alt.Tooltip("density_label:N", title="밀도 수준"),
            alt.Tooltip(f"{value_column}:Q", title=value_title, format=",.1f"),
            alt.Tooltip("sample_count:Q", title="표본 수", format=",.0f"),
            alt.Tooltip("density_min:Q", title="밀도 최솟값", format=".2f"),
            alt.Tooltip("density_max:Q", title="밀도 최댓값", format=".2f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=58)
    labels = base.mark_text(dy=-9, color=CHART_INK, fontWeight="bold").encode(
        text=alt.Text(f"{value_column}:Q", format=",.1f")
    )
    return _style_chart((bars + labels).properties(height=300))


def _context_chart(frame: pd.DataFrame) -> alt.Chart:
    rule = (
        alt.Chart(pd.DataFrame({"x": [50]}))
        .mark_rule(color=CHART_MUTED, strokeDash=[5, 4])
        .encode(x="x:Q")
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("metric_label:N", title=None, sort=None),
        x=alt.X(
            "peer_percentile:Q",
            title="동일 업종·상권유형 내 백분위",
            scale=alt.Scale(domain=[0, 100]),
        ),
        color=alt.Color(
            "domain_label:N",
            title="환경 영역",
            scale=alt.Scale(
                domain=list(DOMAIN_COLORS),
                range=list(DOMAIN_COLORS.values()),
            ),
        ),
        tooltip=[
            alt.Tooltip("domain_label:N", title="영역"),
            alt.Tooltip("metric_label:N", title="지표"),
            alt.Tooltip("display_value:N", title="선택 상권 값"),
            alt.Tooltip("peer_percentile:Q", title="백분위", format=".1f"),
        ],
    )
    bars = base.mark_bar(cornerRadiusEnd=5, size=20)
    labels = base.mark_text(align="right", dx=-6, color="white", fontWeight="bold").encode(
        text=alt.Text("peer_percentile:Q", format=".1f")
    )
    return _style_chart((bars + labels + rule).properties(height=alt.Step(32)))


def _global_model_chart(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(
        y=alt.Y("term_label:N", title=None, sort=None),
        x=alt.X("importance_share:Q", title="전체 중요도 비중(%)"),
        tooltip=[
            alt.Tooltip("term_label:N", title="분석 특성"),
            alt.Tooltip("importance_share:Q", title="중요도 비중(%)", format=".2f"),
        ],
    )
    bars = base.mark_bar(color=CHART_BLUE, cornerRadiusEnd=5, size=24)
    labels = base.mark_text(align="left", dx=7, color=CHART_INK).encode(
        text=alt.Text("importance_share:Q", format=".2f")
    )
    return _style_chart((bars + labels).properties(height=alt.Step(40)))


def _local_model_chart(frame: pd.DataFrame) -> alt.Chart:
    rule = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color=CHART_MUTED)
        .encode(x="x:Q")
    )
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=5, size=24)
        .encode(
            y=alt.Y("term_label:N", title=None, sort=None),
            x=alt.X("sales_ratio_contribution:Q", title="예측값 방향 기여(%)"),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=[
                alt.Tooltip("term_label:N", title="분석 특성"),
                alt.Tooltip(
                    "sales_ratio_contribution:Q",
                    title="예측값 방향 기여(%)",
                    format="+.1f",
                ),
            ],
        )
    )
    return _style_chart((bars + rule).properties(height=alt.Step(40)))


def _style_chart(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Apply the report's shared chart typography, grid, and legend styling."""
    return (
        chart.configure_axis(
            domain=False,
            gridColor=CHART_GRID,
            gridOpacity=1,
            labelAngle=0,
            labelColor=CHART_INK,
            labelFontSize=12,
            labelPadding=7,
            tickColor=CHART_GRID,
            titleColor=CHART_MUTED,
            titleFontSize=12,
            titleFontWeight=600,
            titlePadding=12,
        )
        .configure_view(stroke=None)
        .configure_legend(
            orient="bottom",
            direction="horizontal",
            labelColor=CHART_INK,
            labelFontSize=11,
            titleColor=CHART_MUTED,
            titleFontSize=11,
        )
    )


def render_report_error(message: str, on_close: Callable[[], None]) -> None:
    """Keep the report dismissible when its data cannot be loaded."""
    with st.container(border=True):
        st.button("리포트 닫기", icon=":material/close:", on_click=on_close)
        st.error(message)


def _number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return None if pd.isna(number) else number


def _format_won(value: object) -> str:
    number = _number(value)
    if number is None:
        return "데이터 없음"
    if abs(number) >= 100_000_000:
        return f"{number / 100_000_000:,.1f}억 원"
    if abs(number) >= 10_000:
        return f"{number / 10_000:,.0f}만 원"
    return f"{number:,.0f}원"


def _format_count(value: object) -> str:
    number = _number(value)
    return "데이터 없음" if number is None else f"{number:,.0f}개"


def _format_percent(value: object, *, signed: bool = True) -> str:
    number = _number(value)
    if number is None:
        return "데이터 없음"
    return f"{number:+.1f}%" if signed else f"{number:.1f}%"


def _format_percentage_points(value: object, *, signed: bool = False) -> str:
    number = _number(value)
    if number is None:
        return "데이터 없음"
    return f"{number:+.1f}%p" if signed else f"{number:.1f}%p"


def _change_markup(value: object) -> str:
    """Format a change with a color and a direction icon for inline Markdown."""
    number = _number(value)
    if number is None:
        return ":gray[비교 데이터 없음]"
    if number > 0:
        return f":green[:material/arrow_upward: {number:+.1f}%]"
    if number < 0:
        return f":red[:material/arrow_downward: {number:+.1f}%]"
    return ":gray[:material/arrow_forward: 0.0%]"


def _format_integer(value: object, unit: str) -> str:
    number = _number(value)
    return "데이터 없음" if number is None else f"{number:,.0f}{unit}"


def _format_decimal(value: object, digits: int) -> str:
    number = _number(value)
    return "데이터 없음" if number is None else f"{number:.{digits}f}"


def _format_context_row(row: pd.Series) -> str:
    value = _number(row.get("value"))
    if value is None:
        return "데이터 없음"
    unit = str(row.get("unit", ""))
    if unit == "percent":
        return f"{value:,.1f}%"
    if unit == "people":
        return f"{value:,.0f}명"
    if unit == "households":
        return f"{value:,.0f}가구"
    if unit == "facilities_per_ha":
        return f"{value:,.2f}개/ha"
    if unit == "stores_per_ha":
        return f"{value:,.2f}개/ha"
    if unit == "ratio":
        return f"{value:,.2f}배"
    return f"{value:,.0f}개"


def _first_to_last_change(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 2 or float(numeric.iloc[0]) <= 0:
        return None
    return 100 * (float(numeric.iloc[-1]) / float(numeric.iloc[0]) - 1)


def _trend_direction(change: float | None) -> str:
    if change is None:
        return "판단 유보"
    if change > 3:
        return "상승"
    if change < -3:
        return "하락"
    return "정체"


def _peer_position(percentile: float | None, count: float | None) -> str:
    if percentile is None or count is None or count <= 0:
        return "비교 불가"
    approximate_rank_share = 100 - percentile + (100 / count)
    top_share = min(100.0, max(0.0, approximate_rank_share))
    return f"상위 {top_share:.1f}%"


def _summary_text(peer: str, trend: str, density: str) -> str:
    return (
        f"점포당 매출은 유사 상권에서 {peer} 수준이고, 최근 8분기 흐름은 {trend}입니다. "
        f"경쟁 밀도는 {density} 수준이므로 아래의 성장·고객·환경 근거를 함께 확인하세요."
    )


def _review_points(trend: str, density: str) -> tuple[str, ...]:
    points = ["임대료·권리금과 실제 동종 점포의 방문량을 함께 확인하세요."]
    if density in {"다소 높음", "높음"}:
        points.append("피크 시간대 대기·회전율과 반경 내 직접 경쟁점 수를 확인하세요.")
    else:
        points.append("낮은 점포 밀도가 수요 공백인지 공급 기회인지 현장에서 구분하세요.")
    if trend == "하락":
        points.append("최근 감소가 일시적인지 여러 분기 동안 이어지는지 원인을 확인하세요.")
    elif trend == "상승":
        points.append("최근 상승이 특정 행사·신규 시설 같은 일시 요인인지 확인하세요.")
    else:
        points.append("정체 흐름 안에서도 요일·시간대별 수요 차이가 있는지 확인하세요.")
    return tuple(points)


def _safe_multiple(numerator: object, denominator: object) -> float | None:
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return top / bottom


def _term_label(term: object) -> str:
    parts = str(term).split(" & ")
    return " × ".join(TERM_LABELS.get(part, part.replace("_", " ")) for part in parts)


def _peer_color(percentile: float | None) -> str:
    if percentile is None:
        return "gray"
    if percentile >= 75:
        return "green"
    if percentile < 25:
        return "orange"
    return "blue"


def _trend_color(direction: str) -> str:
    return {"상승": "green", "하락": "orange", "정체": "blue"}.get(direction, "gray")


def _density_color(group: str) -> str:
    if group in {"high", "upper_middle"}:
        return "orange"
    if group in {"low", "lower_middle"}:
        return "green"
    return "blue" if group == "middle" else "gray"


def _growth_type_color(growth_type: str) -> str:
    if growth_type in {"joint_growth", "store_expansion", "ticket_growth", "transaction_growth"}:
        return "green"
    if growth_type == "decline_or_stagnation":
        return "red"
    return "gray"
