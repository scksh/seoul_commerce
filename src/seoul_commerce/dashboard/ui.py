"""Streamlit UI helpers for the map exploration screen."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from seoul_commerce.dashboard.exploration import (
    ALL_ADMIN_DONGS,
    ALL_AREA_TYPES,
    ALL_DISTRICTS,
    ALL_TRADE_AREAS,
    METRICS,
    ExplorationFilters,
    available_comparisons,
)


CSS_PATH = Path(__file__).with_name("dashboard.css")


def configure_page() -> None:
    """Configure Streamlit and load the dashboard stylesheet."""
    st.set_page_config(
        page_title="서울시 상권 분석",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if CSS_PATH.is_file():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="dashboard-header">
          <div class="dashboard-eyebrow">SEOUL COMMERCIAL AREA EXPLORER</div>
          <h1>서울시 상권 분석</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_filters(summary: pd.DataFrame) -> ExplorationFilters:
    """Render the left-side filters and return a validated selection."""
    st.markdown('<div class="section-label">조회 조건</div>', unsafe_allow_html=True)
    st.caption("기준분기 · 2026년 1분기")

    districts = [ALL_DISTRICTS, *sorted(summary["district_name"].dropna().unique())]
    _reset_invalid_selection("dashboard_district", districts, ALL_DISTRICTS)
    district = st.selectbox("자치구", districts, key="dashboard_district")

    geographic = summary
    if district != ALL_DISTRICTS:
        geographic = geographic.loc[geographic["district_name"] == district]

    admin_rows = (
        geographic[["admin_dong_code", "admin_dong_name"]]
        .drop_duplicates()
        .sort_values(["admin_dong_name", "admin_dong_code"])
    )
    admin_labels = dict(zip(admin_rows["admin_dong_code"], admin_rows["admin_dong_name"], strict=True))
    admin_options = [ALL_ADMIN_DONGS, *admin_rows["admin_dong_code"].astype(str)]
    _reset_invalid_selection("dashboard_admin_dong", admin_options, ALL_ADMIN_DONGS)
    admin_dong = st.selectbox(
        "행정동",
        admin_options,
        key="dashboard_admin_dong",
        format_func=lambda value: admin_labels.get(value, value),
    )

    if admin_dong != ALL_ADMIN_DONGS:
        geographic = geographic.loc[geographic["admin_dong_code"] == admin_dong]

    area_rows = (
        geographic[["trade_area_code", "trade_area_name"]]
        .drop_duplicates()
        .sort_values(["trade_area_name", "trade_area_code"])
    )
    area_labels = dict(zip(area_rows["trade_area_code"], area_rows["trade_area_name"], strict=True))
    area_options = [ALL_TRADE_AREAS, *area_rows["trade_area_code"].astype(str)]
    _reset_invalid_selection("dashboard_trade_area", area_options, ALL_TRADE_AREAS)
    trade_area = st.selectbox(
        "상권명",
        area_options,
        key="dashboard_trade_area",
        format_func=lambda value: area_labels.get(value, value),
    )

    area_types = [ALL_AREA_TYPES, *sorted(summary["trade_area_type_name"].dropna().unique())]
    area_type = st.selectbox("상권 유형", area_types)

    industries = (
        summary[["industry_code", "industry_name"]]
        .drop_duplicates()
        .sort_values("industry_code")["industry_name"]
        .tolist()
    )
    industry = st.selectbox("업종", industries)

    metric = st.selectbox("대표 지표", list(METRICS))
    comparison_choices = available_comparisons(metric)
    comparison_key = "dashboard_comparison"
    if st.session_state.get(comparison_key) not in comparison_choices:
        st.session_state[comparison_key] = comparison_choices[0]
    comparison = st.selectbox("비교 기준", comparison_choices, key=comparison_key)

    return ExplorationFilters(
        district=str(district),
        admin_dong=str(admin_dong),
        trade_area=str(trade_area),
        area_type=str(area_type),
        industry=str(industry),
        metric=str(metric),
        comparison=str(comparison),
    )


def render_empty_state() -> None:
    st.warning("선택 조건에서 지도에 표시할 수 있는 상권이 없습니다. 조건을 변경해주세요.")


def _reset_invalid_selection(key: str, options: list[str], default: str) -> None:
    """Reset a dependent selectbox before it is rendered."""
    if st.session_state.get(key) not in options:
        st.session_state[key] = default
