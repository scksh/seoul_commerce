"""Dashboard data loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_summary.csv"
TREND_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_trend.csv"
SEGMENTS_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_segments.csv"
COMPETITION_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_competition.csv"
CONTEXT_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_context.csv"
MODEL_METRICS_PATH = PROJECT_ROOT / "data" / "mart" / "model_metrics.csv"
MODEL_GLOBAL_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_model_global.csv"
MODEL_LOCAL_PATH = PROJECT_ROOT / "data" / "mart" / "dashboard_model_local.csv"
MASTER_PATH = PROJECT_ROOT / "data" / "analysis" / "english" / "trade_area_master.csv"

REQUIRED_COLUMNS = {
    "base_quarter",
    "trade_area_code",
    "trade_area_name",
    "trade_area_type_name",
    "district_name",
    "industry_code",
    "industry_name",
    "longitude",
    "latitude",
    "total_store_count",
    "quarterly_sales_amount",
    "total_floating_population",
    "total_resident_population",
    "sales_qoq_rate",
    "sales_yoy_rate",
    "store_qoq_rate",
    "store_yoy_rate",
    "floating_qoq_rate",
    "floating_yoy_rate",
    "monthly_average_sales_per_store",
    "sales_per_store_qoq_rate",
    "sales_per_store_yoy_rate",
    "sales_per_store_peer_percentile",
    "peer_trade_area_count",
    "density_group",
    "report_available",
    "availability_reason",
    "demand_store_yoy_gap",
    "recent_4q_sales_growth_count",
    "consecutive_sales_growth_quarters",
    "store_log_contribution",
    "volume_log_contribution",
    "ticket_log_contribution",
    "total_sales_log_change",
    "growth_type",
    "quarterly_trend_rate",
    "sales_yoy_volatility",
    "execution_id",
    "actual_sales_per_store",
    "predicted_sales_per_store",
    "prediction_error_rate",
}

TREND_REQUIRED_COLUMNS = {
    "base_quarter",
    "observed_quarter",
    "trade_area_code",
    "trade_area_name",
    "industry_code",
    "industry_name",
    "quarterly_sales_amount",
    "quarterly_sales_count",
    "total_store_count",
    "average_transaction_value",
}

SEGMENTS_REQUIRED_COLUMNS = {
    "base_quarter",
    "trade_area_code",
    "industry_code",
    "trade_area_name",
    "industry_name",
    "segment_type",
    "segment_code",
    "segment_name",
    "segment_order",
    "sales_amount_share",
    "floating_population_share",
    "sales_floating_share_ratio",
    "composition_distance",
    "comparison_available",
}

COMPETITION_REQUIRED_COLUMNS = {
    "base_quarter",
    "industry_code",
    "density_group",
    "sample_count",
    "density_min",
    "density_median",
    "density_max",
    "total_sales_median",
    "sales_per_store_median",
}

CONTEXT_REQUIRED_COLUMNS = {
    "base_quarter",
    "trade_area_code",
    "industry_code",
    "domain",
    "metric",
    "value",
    "unit",
    "peer_percentile",
}

MODEL_METRICS_REQUIRED_COLUMNS = {
    "execution_id",
    "candidate",
    "training_start_quarter",
    "training_end_quarter",
    "validation_quarter",
    "training_rows",
    "validation_rows",
    "validation_log_r2",
    "validation_median_absolute_percentage_error",
}

MODEL_GLOBAL_REQUIRED_COLUMNS = {
    "execution_id",
    "term",
    "term_type",
    "domain",
    "role",
    "importance_share",
    "high_20_low_20_prediction_difference",
}

MODEL_LOCAL_REQUIRED_COLUMNS = {
    "execution_id",
    "base_quarter",
    "trade_area_code",
    "industry_code",
    "term",
    "term_type",
    "sales_ratio_contribution",
}

MASTER_REQUIRED_COLUMNS = {
    "trade_area_code",
    "admin_dong_code",
    "admin_dong_name",
}


class DashboardDataError(RuntimeError):
    """Raised when a dashboard mart cannot be used safely."""


@dataclass(frozen=True)
class ReportMarts:
    """Cached analysis marts needed only after a report is opened."""

    trend: pd.DataFrame
    segments: pd.DataFrame
    competition: pd.DataFrame
    context: pd.DataFrame
    model_metrics: pd.DataFrame
    model_global: pd.DataFrame
    model_local: pd.DataFrame


def read_summary(
    path: Path = SUMMARY_PATH,
    master_path: Path = MASTER_PATH,
) -> pd.DataFrame:
    """Read and validate the map exploration summary mart."""
    if not path.is_file():
        raise DashboardDataError(f"대시보드 요약 마트를 찾을 수 없습니다: {path}")

    frame = pd.read_csv(
        path,
        dtype={
            "base_quarter": "string",
            "trade_area_code": "string",
            "industry_code": "string",
        },
        low_memory=False,
    )
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise DashboardDataError(
            "대시보드 요약 마트의 필수 컬럼이 없습니다: " + ", ".join(missing_columns)
        )
    if frame.empty:
        raise DashboardDataError("대시보드 요약 마트에 표시할 데이터가 없습니다.")

    if not master_path.is_file():
        raise DashboardDataError(f"상권 기준정보를 찾을 수 없습니다: {master_path}")
    master = pd.read_csv(
        master_path,
        dtype={"trade_area_code": "string", "admin_dong_code": "string"},
        low_memory=False,
    )
    missing_master_columns = sorted(MASTER_REQUIRED_COLUMNS.difference(master.columns))
    if missing_master_columns:
        raise DashboardDataError(
            "상권 기준정보의 필수 컬럼이 없습니다: " + ", ".join(missing_master_columns)
        )
    master = master[["trade_area_code", "admin_dong_code", "admin_dong_name"]].drop_duplicates()
    if master["trade_area_code"].duplicated().any():
        raise DashboardDataError("상권 기준정보에 중복된 상권코드가 있습니다.")

    frame = frame.merge(master, on="trade_area_code", how="left", validate="many_to_one")
    missing_admin = int(frame[["admin_dong_code", "admin_dong_name"]].isna().any(axis=1).sum())
    if missing_admin:
        raise DashboardDataError(
            f"행정동에 연결되지 않은 대시보드 행이 {missing_admin:,}개 있습니다."
        )

    frame = frame.copy()
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    if frame[["longitude", "latitude"]].notna().all(axis=1).sum() == 0:
        raise DashboardDataError("지도에 사용할 수 있는 위·경도 좌표가 없습니다.")

    return frame


@st.cache_data(show_spinner=False)
def _load_summary_cached(
    path_text: str,
    summary_modified_ns: int,
    master_path_text: str,
    master_modified_ns: int,
) -> pd.DataFrame:
    del summary_modified_ns, master_modified_ns
    return read_summary(Path(path_text), Path(master_path_text))


def load_summary(
    path: Path = SUMMARY_PATH,
    master_path: Path = MASTER_PATH,
) -> pd.DataFrame:
    """Load the summary mart with file-change-aware Streamlit caching."""
    if not path.is_file():
        raise DashboardDataError(f"대시보드 요약 마트를 찾을 수 없습니다: {path}")
    if not master_path.is_file():
        raise DashboardDataError(f"상권 기준정보를 찾을 수 없습니다: {master_path}")
    return _load_summary_cached(
        str(path),
        path.stat().st_mtime_ns,
        str(master_path),
        master_path.stat().st_mtime_ns,
    )


def read_trend(path: Path = TREND_PATH) -> pd.DataFrame:
    """Read and validate the recent-quarter trend mart."""
    if not path.is_file():
        raise DashboardDataError(f"대시보드 추세 마트를 찾을 수 없습니다: {path}")

    header = pd.read_csv(path, nrows=0).columns
    missing_columns = sorted(TREND_REQUIRED_COLUMNS.difference(header))
    if missing_columns:
        raise DashboardDataError(
            "대시보드 추세 마트의 필수 컬럼이 없습니다: " + ", ".join(missing_columns)
        )

    frame = pd.read_csv(
        path,
        usecols=sorted(TREND_REQUIRED_COLUMNS),
        dtype={
            "base_quarter": "string",
            "observed_quarter": "string",
            "trade_area_code": "string",
            "industry_code": "string",
        },
        low_memory=False,
    )
    if frame.empty:
        raise DashboardDataError("대시보드 추세 마트에 표시할 데이터가 없습니다.")
    return frame


@st.cache_data(show_spinner=False)
def _load_trend_cached(path_text: str, modified_ns: int) -> pd.DataFrame:
    del modified_ns
    return read_trend(Path(path_text))


def load_trend(path: Path = TREND_PATH) -> pd.DataFrame:
    """Load the trend mart only when a report is opened."""
    if not path.is_file():
        raise DashboardDataError(f"대시보드 추세 마트를 찾을 수 없습니다: {path}")
    return _load_trend_cached(str(path), path.stat().st_mtime_ns)


def _read_report_mart(
    path: Path,
    label: str,
    required_columns: tuple[str, ...],
    string_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError(f"{label} 마트를 찾을 수 없습니다: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing_columns = sorted(set(required_columns).difference(header))
    if missing_columns:
        raise DashboardDataError(
            f"{label} 마트의 필수 컬럼이 없습니다: " + ", ".join(missing_columns)
        )
    frame = pd.read_csv(
        path,
        usecols=list(required_columns),
        dtype={column: "string" for column in string_columns},
        low_memory=False,
    )
    if frame.empty:
        raise DashboardDataError(f"{label} 마트에 표시할 데이터가 없습니다.")
    return frame


@st.cache_data(show_spinner=False)
def _load_report_mart_cached(
    path_text: str,
    modified_ns: int,
    label: str,
    required_columns: tuple[str, ...],
    string_columns: tuple[str, ...],
) -> pd.DataFrame:
    del modified_ns
    return _read_report_mart(
        Path(path_text),
        label,
        required_columns,
        string_columns,
    )


def _load_report_mart(
    path: Path,
    label: str,
    required_columns: set[str],
    string_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError(f"{label} 마트를 찾을 수 없습니다: {path}")
    return _load_report_mart_cached(
        str(path),
        path.stat().st_mtime_ns,
        label,
        tuple(sorted(required_columns)),
        string_columns,
    )


def load_report_marts() -> ReportMarts:
    """Load every saved analysis result needed by the scrolling report."""
    return ReportMarts(
        trend=load_trend(),
        segments=_load_report_mart(
            SEGMENTS_PATH,
            "구성비",
            SEGMENTS_REQUIRED_COLUMNS,
            ("base_quarter", "trade_area_code", "industry_code"),
        ),
        competition=_load_report_mart(
            COMPETITION_PATH,
            "경쟁",
            COMPETITION_REQUIRED_COLUMNS,
            ("base_quarter", "industry_code"),
        ),
        context=_load_report_mart(
            CONTEXT_PATH,
            "상권 환경",
            CONTEXT_REQUIRED_COLUMNS,
            ("base_quarter", "trade_area_code", "industry_code"),
        ),
        model_metrics=_load_report_mart(
            MODEL_METRICS_PATH,
            "모델 성능",
            MODEL_METRICS_REQUIRED_COLUMNS,
            (
                "execution_id",
                "candidate",
                "training_start_quarter",
                "training_end_quarter",
                "validation_quarter",
            ),
        ),
        model_global=_load_report_mart(
            MODEL_GLOBAL_PATH,
            "모델 전체 설명",
            MODEL_GLOBAL_REQUIRED_COLUMNS,
            ("execution_id",),
        ),
        model_local=_load_report_mart(
            MODEL_LOCAL_PATH,
            "모델 개별 설명",
            MODEL_LOCAL_REQUIRED_COLUMNS,
            (
                "execution_id",
                "base_quarter",
                "trade_area_code",
                "industry_code",
            ),
        ),
    )
