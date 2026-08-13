"""Streamlit entry point for the Seoul commercial-area dashboard."""

from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from seoul_commerce.dashboard.data import DashboardDataError, load_report_marts, load_summary
from seoul_commerce.dashboard.exploration import build_exploration
from seoul_commerce.dashboard.map_view import build_map, selected_area_code
from seoul_commerce.dashboard.report import (
    build_report_view,
    clear_ai_insight_state,
    render_report,
    render_report_error,
)
from seoul_commerce.dashboard.ui import (
    configure_page,
    render_empty_state,
    render_filters,
    render_header,
)


SELECTED_AREA_KEY = "selected_trade_area_code"
MAP_NONCE_KEY = "dashboard_map_nonce"


def _initialize_state() -> None:
    st.session_state.setdefault(SELECTED_AREA_KEY, None)
    st.session_state.setdefault(MAP_NONCE_KEY, 0)


def _clear_selection() -> None:
    clear_ai_insight_state()
    st.session_state[SELECTED_AREA_KEY] = None
    st.session_state[MAP_NONCE_KEY] += 1


def main() -> None:
    """Assemble the map exploration and selected-area report."""
    configure_page()
    _initialize_state()
    selected_code = st.session_state[SELECTED_AREA_KEY]
    render_header()

    try:
        summary = load_summary()
    except DashboardDataError as exc:
        st.error(str(exc))
        st.stop()

    filter_column, map_column = st.columns([0.24, 0.76], gap="medium")
    with filter_column:
        with st.container(border=True, key="exploration-filters"):
            filters = render_filters(summary)

    view = build_exploration(summary, filters)
    visible_codes = set(view.top10["trade_area_code"].astype(str))
    if selected_code is not None and str(selected_code) not in visible_codes:
        _clear_selection()
        st.rerun()

    with map_column:
        with st.container(key="exploration-map"):
            if view.top10.empty:
                render_empty_state()
                return

            dashboard_map = build_map(view, selected_code=selected_code)
            map_event = st_folium(
                dashboard_map,
                # streamlit-folium requires an integer fallback. The visible height
                # is controlled responsively by dashboard.css.
                height=640,
                use_container_width=True,
                returned_objects=["last_object_clicked"],
                key=f"trade-area-map-{view.state_key}-{st.session_state[MAP_NONCE_KEY]}",
            )
            clicked_code = selected_area_code(map_event, view.top10)
            if clicked_code is not None and clicked_code != selected_code:
                clear_ai_insight_state()
                st.session_state[SELECTED_AREA_KEY] = clicked_code
                st.rerun()

    if selected_code is not None:
        try:
            marts = load_report_marts()
            report = build_report_view(
                summary,
                marts,
                trade_area_code=str(selected_code),
                industry_name=view.filters.industry,
            )
        except (DashboardDataError, ValueError) as exc:
            render_report_error(str(exc), _clear_selection)
        else:
            render_report(report, _clear_selection)


if __name__ == "__main__":
    main()
