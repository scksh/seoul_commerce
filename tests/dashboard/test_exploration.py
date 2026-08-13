from __future__ import annotations

import unittest

import pandas as pd

from seoul_commerce.dashboard.exploration import (
    ExplorationFilters,
    available_comparisons,
    build_exploration,
)
from seoul_commerce.dashboard.map_view import build_map, selected_area_code


class ExplorationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = pd.DataFrame(
            {
                "base_quarter": ["20261", "20261", "20261"],
                "trade_area_code": ["A", "B", "C"],
                "trade_area_name": ["가", "나", "다"],
                "trade_area_type_name": ["골목상권", "골목상권", "전통시장"],
                "district_name": ["강남구", "강남구", "종로구"],
                "admin_dong_code": ["D1", "D1", "D2"],
                "admin_dong_name": ["역삼동", "역삼동", "종로동"],
                "industry_code": ["I", "I", "I"],
                "industry_name": ["커피-음료", "커피-음료", "커피-음료"],
                "longitude": [127.0, 127.1, 126.9],
                "latitude": [37.5, 37.6, 37.57],
                "total_store_count": [10, 20, 30],
                "quarterly_sales_amount": [100.0, 200.0, 300.0],
                "monthly_average_sales_per_store": [10_000_000.0, 25_000_000.0, 15_000_000.0],
                "total_floating_population": [1000.0, 2000.0, 3000.0],
                "total_resident_population": [100.0, 200.0, 300.0],
                "sales_qoq_rate": [1.0, 5.0, -2.0],
                "sales_yoy_rate": [2.0, 3.0, 4.0],
                "sales_per_store_qoq_rate": [1.0, -1.0, 4.0],
                "sales_per_store_yoy_rate": [2.0, 5.0, 3.0],
                "store_qoq_rate": [0.0, 10.0, 2.0],
                "store_yoy_rate": [1.0, 2.0, 3.0],
                "floating_qoq_rate": [3.0, 2.0, 1.0],
                "floating_yoy_rate": [4.0, 5.0, 6.0],
                "report_available": [True, True, False],
                "availability_reason": ["", "", "표본 부족"],
            }
        )

    def test_filters_and_ranks_absolute_values(self) -> None:
        filters = ExplorationFilters(
            district="강남구",
            admin_dong="전체 행정동",
            trade_area="전체 상권",
            area_type="전체",
            industry="커피-음료",
            metric="점포수",
            comparison="절대값",
        )
        view = build_exploration(self.summary, filters)

        self.assertEqual(view.frame["trade_area_code"].tolist(), ["A", "B"])
        self.assertEqual(view.top10.iloc[0]["trade_area_code"], "B")
        self.assertEqual(int(view.top10.iloc[0]["rank"]), 1)

    def test_change_rate_drives_ranking(self) -> None:
        filters = ExplorationFilters(
            district="서울시 전체",
            admin_dong="전체 행정동",
            trade_area="전체 상권",
            area_type="전체",
            industry="커피-음료",
            metric="추정매출",
            comparison="전분기 대비",
        )
        view = build_exploration(self.summary, filters)

        self.assertEqual(view.ranking_column, "sales_qoq_rate")
        self.assertEqual(view.top10.iloc[0]["trade_area_code"], "B")

    def test_monthly_sales_per_store_supports_value_and_change_rankings(self) -> None:
        filters = ExplorationFilters(
            district="서울시 전체",
            admin_dong="전체 행정동",
            trade_area="전체 상권",
            area_type="전체",
            industry="커피-음료",
            metric="월평균 점포당 추정매출",
            comparison="절대값",
        )

        view = build_exploration(self.summary, filters)

        self.assertEqual(view.ranking_column, "monthly_average_sales_per_store")
        self.assertEqual(view.top10.iloc[0]["trade_area_code"], "B")
        self.assertEqual(view.top10.iloc[0]["ranking_display"], "2,500만 원")
        self.assertEqual(
            available_comparisons("월평균 점포당 추정매출"),
            ("절대값", "전분기 대비", "전년 동기 대비"),
        )

    def test_tied_values_still_receive_sequential_display_ranks(self) -> None:
        summary = self.summary.copy()
        summary["total_store_count"] = [10, 20, 20]
        filters = ExplorationFilters(
            district="서울시 전체",
            admin_dong="전체 행정동",
            trade_area="전체 상권",
            area_type="전체",
            industry="커피-음료",
            metric="점포수",
            comparison="절대값",
        )

        view = build_exploration(summary, filters)

        self.assertEqual(view.top10["rank"].astype(int).tolist(), [1, 2, 3])

    def test_resident_population_only_supports_absolute_value(self) -> None:
        self.assertEqual(available_comparisons("주거인구"), ("절대값",))

    def test_admin_dong_and_trade_area_filters_are_applied(self) -> None:
        filters = ExplorationFilters(
            district="강남구",
            admin_dong="D1",
            trade_area="A",
            area_type="전체",
            industry="커피-음료",
            metric="점포수",
            comparison="절대값",
        )
        view = build_exploration(self.summary, filters)

        self.assertEqual(view.top10["trade_area_code"].tolist(), ["A"])
        self.assertEqual(int(view.top10.iloc[0]["rank"]), 2)

    def test_folium_selection_extracts_area_code(self) -> None:
        event = {
            "last_object_clicked": {"lat": 37.5, "lng": 127.0},
        }
        self.assertEqual(selected_area_code(event, self.summary), "A")

    def test_map_embeds_responsive_height_style(self) -> None:
        filters = ExplorationFilters(
            district="서울시 전체",
            admin_dong="전체 행정동",
            trade_area="전체 상권",
            area_type="전체",
            industry="커피-음료",
            metric="점포수",
            comparison="절대값",
        )
        dashboard_map = build_map(build_exploration(self.summary, filters), selected_code=None)

        self.assertIn("responsive-map-height", dashboard_map.get_root().render())


if __name__ == "__main__":
    unittest.main()
