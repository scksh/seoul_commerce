"""Folium map construction and future click-event parsing."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from typing import Any

import folium
import pandas as pd

from seoul_commerce.dashboard.exploration import (
    ALL_ADMIN_DONGS,
    ALL_DISTRICTS,
    ALL_TRADE_AREAS,
    ExplorationView,
)


SEOUL_CENTER = {"latitude": 37.5665, "longitude": 126.9780}
LABEL_WIDTH = 224
LABEL_HEIGHT = 56
RESPONSIVE_MAP_STYLE = """
<style>
  /* responsive-map-height */
  html,
  body,
  #root,
  #parent,
  .float-container.single,
  .float-child,
  #map_div.leaflet-container {
    height: 100% !important;
    min-height: 0 !important;
  }

  body {
    margin: 0 !important;
    overflow: hidden !important;
  }
</style>
"""


def build_map(view: ExplorationView, selected_code: str | None) -> folium.Map:
    """Build a Leaflet map containing only the ranked TOP 10 trade areas."""
    ranked = _prepare_layer_data(view.top10)
    center = _map_center(ranked)
    dashboard_map = folium.Map(
        location=[center["latitude"], center["longitude"]],
        zoom_start=_map_zoom(view),
        tiles="CartoDB positron",
        control_scale=True,
        zoom_control=True,
        prefer_canvas=False,
    )
    dashboard_map.get_root().header.add_child(
        folium.Element(RESPONSIVE_MAP_STYLE),
        name="responsive_map_height",
    )

    for index, row in ranked.reset_index(drop=True).iterrows():
        direction = 1 if index % 2 == 0 else -1
        code = str(row["trade_area_code"])
        folium.Marker(
            location=[float(row["latitude"]), float(row["longitude"])],
            icon=folium.DivIcon(
                icon_size=(LABEL_WIDTH, LABEL_HEIGHT),
                icon_anchor=(17 if direction == 1 else LABEL_WIDTH - 17, 28),
                class_name="trade-area-label",
                html=_label_html(row, direction, code == str(selected_code)),
            ),
            tooltip=folium.Tooltip(
                _tooltip_html(row),
                sticky=True,
                style=(
                    "background:#172554;color:#fff;border:0;border-radius:8px;"
                    "font-size:12px;padding:8px 10px;"
                ),
            ),
            rise_on_hover=True,
        ).add_to(dashboard_map)

    return dashboard_map


def selected_area_code(event: Any, frame: pd.DataFrame) -> str | None:
    """Resolve a future streamlit-folium marker click to a trade-area code."""
    clicked = _mapping_value(event, "last_object_clicked")
    latitude = _mapping_value(clicked, "lat") if clicked else None
    longitude = _mapping_value(clicked, "lng") if clicked else None
    if latitude is None or longitude is None or frame.empty:
        return None

    distances = (frame["latitude"] - float(latitude)).abs() + (
        frame["longitude"] - float(longitude)
    ).abs()
    nearest_index = distances.idxmin()
    if float(distances.loc[nearest_index]) > 0.000001:
        return None
    return str(frame.loc[nearest_index, "trade_area_code"])


def _prepare_layer_data(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_area_code",
        "trade_area_name",
        "district_name",
        "admin_dong_name",
        "industry_name",
        "longitude",
        "latitude",
        "metric_display",
        "ranking_display",
        "rank",
        "report_available",
    ]
    ranked = frame[columns].copy()
    ranked["rank_text"] = ranked["rank"].astype("Int64").astype("string")
    ranked["report_status"] = ranked["report_available"].map(
        lambda value: "상세 분석 가능" if bool(value) else "상세 분석 불가"
    )
    return ranked


def _label_html(row: pd.Series, direction: int, selected: bool) -> str:
    """Create one always-visible rank, name, and value label."""
    circle_side = "left:1px" if direction == 1 else "right:1px"
    boxes_side = "left:38px" if direction == 1 else "right:38px"
    alignment = "left" if direction == 1 else "right"
    marker_color = "#f72585" if selected else "#1268d6"
    name = escape(str(row["trade_area_name"]))
    value = escape(str(row["ranking_display"]))
    rank = escape(str(row["rank_text"]))
    return f"""
    <div style="position:relative;width:{LABEL_WIDTH}px;height:{LABEL_HEIGHT}px;cursor:pointer;
                font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;">
      <div style="position:absolute;{boxes_side};top:2px;width:max-content;max-width:184px;
                  height:24px;
                  box-sizing:border-box;border:1px solid #b8c8dc;border-radius:4px;
                  background:#fff;color:#172554;font-size:12px;font-weight:700;
                  line-height:22px;padding:0 5px;text-align:{alignment};
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                  box-shadow:0 2px 7px rgba(23,37,84,.12);">{name}</div>
      <div style="position:absolute;{boxes_side};top:30px;width:max-content;max-width:184px;
                  height:24px;
                  box-sizing:border-box;border:1px solid #93b1d6;border-radius:4px;
                  background:#eaf3ff;color:#0757b8;font-size:11px;font-weight:700;
                  line-height:22px;padding:0 5px;text-align:{alignment};
                  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{value}</div>
      <div style="position:absolute;{circle_side};top:11px;width:34px;height:34px;
                  box-sizing:border-box;border:2px solid #fff;border-radius:50%;
                  background:{marker_color};color:#fff;font-size:13px;font-weight:800;
                  line-height:30px;text-align:center;z-index:2;
                  box-shadow:0 2px 8px rgba(23,37,84,.25);">
        {rank}
      </div>
    </div>
    """


def _tooltip_html(row: pd.Series) -> str:
    location = f"{row['district_name']} · {row['admin_dong_name']}"
    return (
        f"<b>{escape(str(row['trade_area_name']))}</b><br>"
        f"{escape(location)}<br>"
        f"{escape(str(row['industry_name']))}<br>"
        f"현재값: {escape(str(row['metric_display']))}<br>"
        f"선택 기준: {escape(str(row['ranking_display']))}<br>"
        f"조건 내 순위: {escape(str(row['rank_text']))}위<br>"
        f"{escape(str(row['report_status']))}"
    )


def _map_zoom(view: ExplorationView) -> int:
    if view.filters.trade_area != ALL_TRADE_AREAS:
        return 14
    if view.filters.admin_dong != ALL_ADMIN_DONGS:
        return 13
    if view.filters.district != ALL_DISTRICTS:
        return 12
    return 11


def _map_center(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return SEOUL_CENTER
    return {
        "latitude": float(frame["latitude"].median()),
        "longitude": float(frame["longitude"].median()),
    }


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)
