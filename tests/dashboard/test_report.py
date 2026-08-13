from __future__ import annotations

from dataclasses import replace
import unittest

import pandas as pd

from seoul_commerce.dashboard.data import ReportMarts
from seoul_commerce.dashboard.report import (
    _density_chart,
    _growth_trend_chart,
    build_report_view,
)


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = pd.DataFrame(
            {
                "base_quarter": ["20261"],
                "trade_area_code": ["A"],
                "trade_area_name": ["테스트상권"],
                "trade_area_type_name": ["발달상권"],
                "district_name": ["강남구"],
                "admin_dong_name": ["역삼동"],
                "industry_code": ["I"],
                "industry_name": ["한식음식점"],
                "monthly_average_sales_per_store": [36_000_000.0],
                "quarterly_sales_amount": [1_080_000_000.0],
                "sales_qoq_rate": [5.0],
                "sales_per_store_qoq_rate": [5.0],
                "sales_per_store_yoy_rate": [10.0],
                "sales_per_store_peer_percentile": [85.8],
                "peer_trade_area_count": [247],
                "density_group": ["high"],
                "total_store_count": [10],
                "report_available": [True],
                "availability_reason": ["available"],
                "sales_yoy_rate": [12.0],
                "store_yoy_rate": [2.0],
                "floating_yoy_rate": [8.0],
                "demand_store_yoy_gap": [6.0],
                "store_log_contribution": [2.0],
                "volume_log_contribution": [7.0],
                "ticket_log_contribution": [3.0],
                "total_sales_log_change": [12.0],
                "growth_type": ["transaction_growth"],
                "quarterly_trend_rate": [3.0],
                "sales_yoy_volatility": [1.5],
                "recent_4q_sales_growth_count": [4],
                "consecutive_sales_growth_quarters": [3],
                "execution_id": ["run-1"],
                "actual_sales_per_store": [108_000_000.0],
                "predicted_sales_per_store": [100_000_000.0],
                "prediction_error_rate": [0.08],
            }
        )
        quarters = ["20242", "20243", "20244", "20251", "20252", "20253", "20254", "20261"]
        self.trend = pd.DataFrame(
            {
                "base_quarter": ["20261"] * 8,
                "observed_quarter": quarters,
                "trade_area_code": ["A"] * 8,
                "industry_code": ["I"] * 8,
                "quarterly_sales_amount": [900_000_000, 930_000_000, 960_000_000, 990_000_000,
                                             1_020_000_000, 1_040_000_000, 1_060_000_000, 1_080_000_000],
                "quarterly_sales_count": [9_000, 9_300, 9_600, 9_000, 10_200, 10_400, 10_600, 10_800],
                "total_store_count": [10] * 8,
                "average_transaction_value": [30_000, 30_500, 31_000, 30_000, 31_000, 32_000, 32_500, 33_000],
                "total_floating_population": [100_000, 101_000, 102_000, 103_000, 105_000, 108_000, 111_000, 115_000],
            }
        )
        segments = pd.DataFrame(
            {
                "base_quarter": ["20261"] * 4,
                "trade_area_code": ["A"] * 4,
                "industry_code": ["I"] * 4,
                "segment_type": ["gender", "age", "weekday", "time_band"],
                "segment_name": ["여성", "30대", "주말", "저녁"],
                "sales_amount_share": [0.6, 0.4, 0.35, 0.3],
                "floating_population_share": [0.5, 0.2, 0.25, 0.2],
                "composition_distance": [0.1, 0.2, 0.15, 0.25],
                "sales_floating_share_ratio": [1.2, 2.0, 1.4, 1.5],
                "comparison_available": [True] * 4,
            }
        )
        competition = pd.DataFrame(
            {
                "base_quarter": ["20261"] * 5,
                "industry_code": ["I"] * 5,
                "density_group": ["low", "lower_middle", "middle", "upper_middle", "high"],
                "sample_count": [20] * 5,
                "density_min": [1, 2, 3, 4, 5],
                "density_median": [1.5, 2.5, 3.5, 4.5, 5.5],
                "density_max": [2, 3, 4, 5, 6],
                "total_sales_median": [100_000_000, 150_000_000, 200_000_000, 250_000_000, 300_000_000],
                "sales_per_store_median": [10_000_000, 12_000_000, 14_000_000, 16_000_000, 18_000_000],
            }
        )
        context = pd.DataFrame(
            {
                "base_quarter": ["20261"] * 3,
                "trade_area_code": ["A"] * 3,
                "industry_code": ["I"] * 3,
                "domain": ["usage_time", "access", "competition"],
                "metric": ["weekend_floating_share", "subway_station_count", "store_density"],
                "value": [0.4, 3.0, 5.5],
                "peer_percentile": [80.0, 70.0, 90.0],
            }
        )
        model_metrics = pd.DataFrame(
            {
                "execution_id": ["run-1"],
                "candidate": ["C"],
                "validation_log_r2": [0.64],
                "validation_median_absolute_percentage_error": [0.16],
                "validation_rows": [100],
                "training_start_quarter": ["20241"],
                "training_end_quarter": ["20254"],
                "validation_quarter": ["20261"],
            }
        )
        model_global = pd.DataFrame(
            {
                "execution_id": ["run-1", "run-1"],
                "term": ["log_floating_population", "log_store_density"],
                "term_type": ["main", "main"],
                "role": ["analysis_factor", "analysis_factor"],
                "importance_share": [0.6, 0.4],
            }
        )
        model_local = pd.DataFrame(
            {
                "base_quarter": ["20261", "20261"],
                "execution_id": ["run-1", "run-1"],
                "trade_area_code": ["A", "A"],
                "industry_code": ["I", "I"],
                "term": ["log_floating_population", "log_store_density"],
                "term_type": ["main", "main"],
                "sales_ratio_contribution": [0.12, -0.05],
            }
        )
        self.marts = ReportMarts(
            trend=self.trend,
            segments=segments,
            competition=competition,
            context=context,
            model_metrics=model_metrics,
            model_global=model_global,
            model_local=model_local,
        )

    def test_builds_all_storytelling_sections(self) -> None:
        report = build_report_view(self.summary, self.marts, "A", "한식음식점")

        self.assertEqual(report.monthly_sales_display, "3,600만 원")
        self.assertEqual(report.trend_change_display, "+20.0%")
        self.assertEqual(report.peer_position_display, "상위 14.6%")
        self.assertEqual(len(report.growth.trend), 8)
        self.assertEqual(len(report.growth.contributions), 3)
        self.assertEqual(report.growth.current_total_sales_display, "10.8억 원")
        self.assertEqual(
            report.growth.current_monthly_sales_per_store_display,
            "3,600만 원",
        )
        self.assertEqual(report.growth.total_sales_qoq_change, 5.0)
        self.assertEqual(report.growth.sales_per_store_yoy_change, 10.0)
        self.assertEqual(len(report.customer.distances), 4)
        self.assertEqual(len(report.customer.overrepresented), 4)
        self.assertEqual(report.competition.selected_group, "높음")
        self.assertEqual(len(report.competition.summary), 5)
        self.assertEqual(report.context.top_metrics[0], "점포 밀도")
        self.assertEqual(report.model.candidate, "C")
        self.assertEqual(report.model.prediction_error_rate, 8.0)
        self.assertEqual(len(report.model.global_terms), 2)
        self.assertEqual(len(report.model.local_terms), 2)

    def test_incomplete_history_is_exposed_without_inventing_a_trend(self) -> None:
        summary = self.summary.copy()
        summary["report_available"] = False
        summary["availability_reason"] = "incomplete_commercial_history"
        marts = replace(self.marts, trend=self.trend.iloc[0:0])

        report = build_report_view(summary, marts, "A", "한식음식점")

        self.assertFalse(report.report_available)
        self.assertEqual(report.availability_label, "연속 8개 분기 이력 부족")
        self.assertEqual(report.trend_direction, "판단 유보")
        self.assertEqual(report.trend_change_display, "비교 데이터 없음")
        self.assertTrue(report.growth.trend.empty)

    def test_missing_selected_area_is_reported(self) -> None:
        with self.assertRaisesRegex(ValueError, "최신 요약 데이터"):
            build_report_view(self.summary, self.marts, "missing", "한식음식점")

    def test_chart_style_keeps_category_tick_labels_horizontal(self) -> None:
        report = build_report_view(self.summary, self.marts, "A", "한식음식점")
        charts = (
            _growth_trend_chart(
                report.growth.trend,
                "total_sales_hundred_million",
                "분기 총 추정매출(억원)",
            ),
            _density_chart(
                report.competition.summary,
                "total_sales_hundred_million",
                "총 추정매출 중앙값(억원)",
            ),
        )

        for chart in charts:
            spec = chart.to_dict()
            self.assertEqual(spec["config"]["axis"]["labelAngle"], 0)


if __name__ == "__main__":
    unittest.main()
