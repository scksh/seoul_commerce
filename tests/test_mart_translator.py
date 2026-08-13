import csv
import tempfile
import unittest
from pathlib import Path

from seoul_commerce import mart_translator


class MartTranslatorTests(unittest.TestCase):
    def test_translate_mart_csv_preserves_numbers_and_translates_labels(self) -> None:
        config = {
            "columns": {"density_group": "점포밀도_그룹", "value": "값", "term": "항"},
            "values": {"density_group": {"high": "높음"}},
            "terms": {"industry": "업종", "log_store_density": "점포밀도_로그"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "korean.csv"
            source.write_text(
                "density_group,value,term\nhigh,1.234567890123,log_store_density & industry\n",
                encoding="utf-8",
            )
            rows = mart_translator.translate_mart_csv(source, output, config)
            with output.open(encoding="utf-8-sig", newline="") as file:
                result = list(csv.reader(file))
        self.assertEqual(rows, 1)
        self.assertEqual(result[0], ["점포밀도_그룹", "값", "항"])
        self.assertEqual(result[1], ["높음", "1.234567890123", "점포밀도_로그 & 업종"])

    def test_translate_mart_csv_quotes_every_field_for_csv_auto_detection(self) -> None:
        config = {
            "columns": {"trade_area_name": "상권_명", "value": "값"},
            "values": {},
            "terms": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "korean.csv"
            source.write_text(
                'trade_area_name,value\n"이화여대 3,5,7길 상점가",1\n', encoding="utf-8"
            )
            mart_translator.translate_mart_csv(source, output, config)
            raw = output.read_text(encoding="utf-8-sig")
        self.assertEqual(raw, '"상권_명","값"\n"이화여대 3,5,7길 상점가","1"\n')

    def test_translate_mart_csv_rejects_unmapped_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            source.write_text("unknown\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "매핑에 없는"):
                mart_translator.translate_mart_csv(
                    source, Path(directory) / "output.csv",
                    {"columns": {}, "values": {}, "terms": {}},
                )

    def test_translate_marts_keeps_source_and_destination_separate(self) -> None:
        with self.assertRaisesRegex(ValueError, "입력 및 출력"):
            mart_translator.translate_marts("same", "same")


if __name__ == "__main__":
    unittest.main()
