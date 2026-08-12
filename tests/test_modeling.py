"""EBM 표본 생성, 시간 분할, 후보 선택 계약을 검증한다."""

import unittest
import time

import numpy as np
import pandas as pd

from seoul_commerce import modeling


class ModelingTests(unittest.TestCase):
    def test_fit_progress_reports_start_heartbeat_and_completion(self) -> None:
        messages: list[str] = []

        class SlowModel:
            def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
                time.sleep(0.04)

        elapsed = modeling._fit_with_progress(
            SlowModel(), pd.DataFrame({"x": [1]}), pd.Series([1]),
            "테스트 학습", messages.append, heartbeat_seconds=0.01,
        )

        self.assertGreater(elapsed, 0)
        self.assertTrue(any("시작" in message for message in messages))
        self.assertTrue(any("진행 중" in message for message in messages))
        self.assertTrue(any("완료" in message for message in messages))

    def test_feature_contract_has_15_numeric_and_4_categorical_features(self) -> None:
        self.assertEqual(len(modeling.NUMERIC_FEATURES), 15)
        self.assertEqual(len(modeling.CATEGORICAL_FEATURES), 4)
        self.assertEqual(modeling.FEATURE_COLUMNS, modeling.NUMERIC_FEATURES + modeling.CATEGORICAL_FEATURES)

    def test_prepare_model_sample_uses_log_target_and_time_composition(self) -> None:
        commercial = pd.DataFrame([{
            "quarter_code": "20251", "trade_area_code": "A", "trade_area_name": "테스트",
            "industry_code": "I", "industry_name": "업종", "trade_area_type_name": "골목",
            "district_name": "자치구", "quarterly_sales_per_store": 100.0,
            "total_floating_population": 1000, "total_resident_population": 100,
            "total_working_population": 200, "total_household_count": 50, "area_ha": 2,
            "facility_density": 3, "subway_station_count": 1, "bus_stop_count": 4,
            "store_density": 2, "franchise_share": 0.1, "full_analysis_available": True,
        }])
        shares = {"sat": .1, "sun": .2, "11_14": .15, "17_21": .2, "21_24": .1, "00_06": .05}
        segments = pd.DataFrame([
            {
                "quarter_code": "20251", "trade_area_code": "A", "industry_code": "I",
                "segment_code": code, "floating_population_share": share,
                "comparison_available": True,
            }
            for code, share in shares.items()
        ])
        result = modeling.prepare_model_sample(commercial, segments)
        self.assertTrue(np.isclose(result.loc[0, modeling.TARGET_COLUMN], np.log(100)))
        self.assertTrue(np.isclose(result.loc[0, "weekend_floating_share"], .3))
        self.assertEqual(result[modeling.FEATURE_COLUMNS].shape[1], 19)

    def test_reference_split_uses_only_20261_for_validation(self) -> None:
        quarters = ["20241", "20242", "20243", "20244", "20251", "20252", "20253"]
        sample = pd.DataFrame({"quarter_code": quarters})
        train_index, validation_index = modeling.reference_quarter_split(sample, "20253")
        self.assertTrue((sample.loc[train_index, "quarter_code"] < "20253").all())
        self.assertTrue((sample.loc[validation_index, "quarter_code"] == "20253").all())

    def test_candidate_selection_prioritizes_median_error(self) -> None:
        metrics = pd.DataFrame({
            "candidate": ["A", "A", "B", "B"],
            "validation_median_absolute_percentage_error": [.2, .3, .1, .2],
            "validation_log_r2": [.8, .8, .5, .5],
            "train_validation_r2_gap": [.1, .1, .2, .2],
        })
        self.assertEqual(modeling.select_candidate(metrics, ["A", "B"]), "B")


if __name__ == "__main__":
    unittest.main()
