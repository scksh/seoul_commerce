"""서울 상권 분석 데이터 전처리 기능을 검증하는 테스트 모듈.

Author:
    SeungHwan Kim

Created:
    2026-08-12

Description:
    설정 계약, 안전한 나눗셈, 영문·한국어 CSV 저장을 검증한다.
"""

import csv
import os
import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from seoul_commerce import preprocessing


class PreprocessingTests(unittest.TestCase):
    def test_config_covers_all_analysis_columns(self) -> None:
        schema_columns = {
            column
            for schema in preprocessing.ANALYSIS_SCHEMAS.values()
            for column in schema["columns"]
        }

        self.assertEqual(
            schema_columns.difference(preprocessing.ANALYSIS_COLUMN_MAPPING_KO),
            set(),
        )

    def test_safe_divide_keeps_zero_denominator_missing(self) -> None:
        result = preprocessing._safe_divide(
            pd.Series([10, 10, pd.NA], dtype="Int64"),
            pd.Series([2, 0, 2], dtype="Int64"),
        )

        self.assertEqual(result.iloc[0], 5)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertTrue(pd.isna(result.iloc[2]))

    def test_write_csv_pair_changes_only_header(self) -> None:
        frame = pd.DataFrame({
            "trade_area_code": pd.Series(["3110001"], dtype="string"),
            "trade_area_name": pd.Series(["테스트 상권"], dtype="string"),
        })

        with TemporaryDirectory() as directory:
            english_path, korean_path = preprocessing.write_csv_pair(
                frame,
                "trade_area.csv",
                "상권.csv",
                os.path.join(directory, "english"),
                os.path.join(directory, "korean"),
            )
            with open(english_path, encoding="utf-8-sig", newline="") as file:
                english_rows = list(csv.reader(file))
            with open(korean_path, encoding="utf-8-sig", newline="") as file:
                korean_rows = list(csv.reader(file))

        self.assertEqual(english_rows[1:], korean_rows[1:])
        self.assertEqual(korean_rows[0], ["상권_코드", "상권_명"])


if __name__ == "__main__":
    unittest.main()
