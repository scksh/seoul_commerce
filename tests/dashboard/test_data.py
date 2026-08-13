from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from seoul_commerce.dashboard.data import DashboardDataError, read_summary, read_trend


class DashboardDataTests(unittest.TestCase):
    def test_missing_columns_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            pd.DataFrame({"trade_area_code": ["A"]}).to_csv(path, index=False)

            with self.assertRaisesRegex(DashboardDataError, "필수 컬럼"):
                read_summary(path)

    def test_trend_missing_columns_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trend.csv"
            pd.DataFrame({"trade_area_code": ["A"]}).to_csv(path, index=False)

            with self.assertRaisesRegex(DashboardDataError, "필수 컬럼"):
                read_trend(path)


if __name__ == "__main__":
    unittest.main()
