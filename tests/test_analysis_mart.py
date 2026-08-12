"""대시보드 분석 마트의 핵심 산식과 계약을 검증한다."""

import unittest

import numpy as np
import pandas as pd

from seoul_commerce import analysis_mart


class AnalysisMartTests(unittest.TestCase):
    def test_recent_quarters_cross_year_boundary(self) -> None:
        self.assertEqual(
            analysis_mart.recent_quarter_codes("20261", 8),
            ["20242", "20243", "20244", "20251", "20252", "20253", "20254", "20261"],
        )

    def test_eligibility_requires_all_21_segments_in_every_quarter(self) -> None:
        quarters = analysis_mart.recent_quarter_codes("20261", 8)
        commercial = pd.DataFrame([
            {
                "quarter_code": quarter, "trade_area_code": "A", "industry_code": "I",
                "full_analysis_available": True, "quarterly_sales_amount": 100,
                "quarterly_sales_count": 10, "total_store_count": 2, "area_ha": 1,
            }
            for quarter in quarters
        ])
        segments = pd.DataFrame([
            {
                "quarter_code": quarter, "trade_area_code": "A", "industry_code": "I",
                "segment_type": "type", "segment_code": f"S{item:02d}",
                "comparison_available": True,
            }
            for quarter in quarters for item in range(21)
        ])

        eligible = analysis_mart.build_eligibility(
            commercial, segments, "20261", recent_quarters=8, minimum_trade_areas=1
        )
        self.assertTrue(bool(eligible.loc[0, "report_available"]))

        incomplete = segments.drop(segments.index[-1])
        unavailable = analysis_mart.build_eligibility(
            commercial, incomplete, "20261", recent_quarters=8, minimum_trade_areas=1
        )
        self.assertFalse(bool(unavailable.loc[0, "segment_history_complete"]))

    def test_growth_contributions_sum_to_log_sales_change(self) -> None:
        quarters = analysis_mart.recent_quarter_codes("20261", 8)
        rows = []
        for index, quarter in enumerate(quarters):
            store_count = 10 if quarter != "20261" else 20
            rows.append({
                "quarter_code": quarter, "trade_area_code": "A", "industry_code": "I",
                "total_store_count": store_count,
                "quarterly_sales_count_per_store": 100,
                "average_transaction_value": 10,
                "quarterly_sales_amount": store_count * 100 * 10,
                "sales_yoy_rate": float(index),
                "recent_4q_sales_growth_count": 1,
                "consecutive_sales_growth_quarters": 1,
            })
        result = analysis_mart._growth_metrics(pd.DataFrame(rows), "20261", 8).iloc[0]
        contribution_sum = result[[
            "store_log_contribution", "volume_log_contribution", "ticket_log_contribution"
        ]].sum()
        self.assertTrue(np.isclose(contribution_sum, result["total_sales_log_change"]))

    def test_rank_ties_use_growth_then_trade_area_code(self) -> None:
        sample = pd.DataFrame({
            "industry_code": ["I", "I", "I"],
            "trade_area_code": ["B", "A", "C"],
            "trade_area_type_name": ["T", "T", "T"],
            "quarterly_sales_amount": [100, 100, 50],
            "quarterly_sales_per_store": [10, 10, 5],
            "sales_yoy_rate": [1, 2, 3],
        })
        result = analysis_mart.add_rank_metrics(sample).set_index("trade_area_code")
        self.assertEqual(int(result.loc["A", "total_sales_rank"]), 1)
        self.assertEqual(int(result.loc["B", "total_sales_rank"]), 2)


if __name__ == "__main__":
    unittest.main()
