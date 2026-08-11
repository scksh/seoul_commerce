"""한국어 컬럼 CSV 복사본 생성 기능을 검증한다."""

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seoul_commerce.column_translator import (
    DEFAULT_INPUT_DIR,
    build_column_mapping,
    load_mapping_config,
    translate_csv,
)


class ColumnTranslatorTests(unittest.TestCase):
    def test_all_raw_csv_columns_have_korean_mapping(self) -> None:
        config = load_mapping_config()

        for input_path in DEFAULT_INPUT_DIR.glob("*.csv"):
            with self.subTest(dataset=input_path.stem):
                mapping = build_column_mapping(config, input_path.stem)
                with input_path.open(encoding="utf-8-sig", newline="") as file:
                    columns = next(csv.reader(file))

                self.assertEqual(
                    [column for column in columns if column not in mapping],
                    [],
                )
                korean_columns = [mapping[column] for column in columns]
                self.assertEqual(len(korean_columns), len(set(korean_columns)))

    def test_translate_csv_changes_only_header(self) -> None:
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            input_path = directory_path / "trade_area.csv"
            output_path = directory_path / "korean" / "trade_area.csv"
            original_rows = [
                ["TRDAR_CD", "TRDAR_CD_NM", "SOURCE_DATASET_ID"],
                ["100001", "테스트 상권", "OA-12345"],
            ]

            with input_path.open("w", encoding="utf-8-sig", newline="") as file:
                csv.writer(file).writerows(original_rows)

            row_count = translate_csv(
                input_path,
                output_path,
                "trade_area",
            )

            with input_path.open(encoding="utf-8-sig", newline="") as file:
                self.assertEqual(list(csv.reader(file)), original_rows)
            with output_path.open(encoding="utf-8-sig", newline="") as file:
                translated_rows = list(csv.reader(file))

            self.assertEqual(row_count, 1)
            self.assertEqual(
                translated_rows[0],
                ["상권_코드", "상권_코드명", "원본_데이터셋_ID"],
            )
            self.assertEqual(translated_rows[1:], original_rows[1:])

    def test_unmapped_column_raises_clear_error(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "trade_area.csv"
            output_path = Path(directory) / "result.csv"
            input_path.write_text("UNKNOWN_COLUMN\nvalue\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "매핑에 없는 컬럼"):
                translate_csv(input_path, output_path, "trade_area")

            self.assertFalse(output_path.exists())

    def test_input_and_output_must_be_different(self) -> None:
        with TemporaryDirectory() as directory:
            input_path = Path(directory) / "trade_area.csv"
            input_path.write_text("TRDAR_CD\n100001\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "입력 파일과 출력 파일"):
                translate_csv(input_path, input_path, "trade_area")


if __name__ == "__main__":
    unittest.main()
