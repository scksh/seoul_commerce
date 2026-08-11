"""OpenAPI 데이터 수집 기능을 검증하는 테스트 모듈.

Author:
    SeungHwan Kim

Created:
    2026-08-11

Description:
    URL 생성, 응답 해석, 데이터 수집 및 CSV 저장 동작을 검증한다.
"""


import unittest
from unittest.mock import Mock, call, patch

import pandas as pd

from seoul_commerce.collector import (
    build_url,
    collect_dataset,
    load_contract,
    make_dataframe,
    parse_json,
    parse_xml,
    save_csv,
)


class CollectorTests(unittest.TestCase):
    def test_build_url_with_quarter(self) -> None:
        url = build_url("http://example.com", "key", "json", "Service", 1, 1000, "20252")
        self.assertEqual(url, "http://example.com/key/json/Service/1/1000/20252")

    def test_parse_json(self) -> None:
        response = Mock()
        response.json.return_value = {
            "Service": {
                "list_total_count": 1,
                "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다"},
                "row": [{"VALUE": "1"}],
            }
        }
        self.assertEqual(parse_json(response, "Service"), ([{"VALUE": "1"}], 1))

    def test_parse_xml(self) -> None:
        response = Mock()
        response.text = """<Service><list_total_count>1</list_total_count>
        <RESULT><CODE>INFO-000</CODE><MESSAGE>OK</MESSAGE></RESULT>
        <row><VALUE>1</VALUE></row></Service>"""
        self.assertEqual(parse_xml(response, "Service"), ([{"VALUE": "1"}], 1))

    def test_missing_contract_file_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "설정 파일을 찾을 수 없습니다"):
            load_contract("trade_area", "missing.yml")

    def test_invalid_xml_raises_clear_error(self) -> None:
        response = Mock()
        response.text = "<Service>"

        with self.assertRaisesRegex(ValueError, "XML 응답을 해석할 수 없습니다"):
            parse_xml(response, "Service")

    def test_csv_permission_error_has_clear_message(self) -> None:
        dataframe = pd.DataFrame([{"VALUE": 1}])

        with patch.object(dataframe, "to_csv", side_effect=PermissionError):
            with self.assertRaisesRegex(ValueError, "파일이 열려 있는지"):
                save_csv(dataframe, "result.csv")

    def test_make_dataframe_uses_row_quarter_as_source_base_date(self) -> None:
        dataframe = make_dataframe(
            rows=[{"STDR_YYQU_CD": "20241"}, {"STDR_YYQU_CD": "20242"}],
            required_columns=["STDR_YYQU_CD"],
            dataset={"dataset_id": "sample"},
            schema_version="0.1.0",
        )

        self.assertEqual(
            dataframe["SOURCE_BASE_DATE"].tolist(),
            ["20241", "20242"],
        )

    def test_make_dataframe_prefers_requested_quarter(self) -> None:
        dataframe = make_dataframe(
            rows=[{"STDR_YYQU_CD": "20241"}],
            required_columns=["STDR_YYQU_CD"],
            dataset={"dataset_id": "sample"},
            schema_version="0.1.0",
            quarter="20252",
        )

        self.assertEqual(dataframe["SOURCE_BASE_DATE"].tolist(), ["20252"])

    def test_make_dataframe_uses_reported_source_base_date(self) -> None:
        dataframe = make_dataframe(
            rows=[{"TRDAR_CD": "3110001"}],
            required_columns=["TRDAR_CD"],
            dataset={
                "dataset_id": "sample",
                "reported_source_base_date": "2023-06",
            },
            schema_version="0.1.0",
        )

        self.assertEqual(dataframe["SOURCE_BASE_DATE"].tolist(), ["2023-06"])

    @patch("seoul_commerce.collector.save_csv")
    @patch("seoul_commerce.collector.make_dataframe")
    @patch("seoul_commerce.collector.fetch_page")
    @patch("seoul_commerce.collector.load_api_key", return_value="test-key")
    @patch("seoul_commerce.collector.load_contract")
    @patch("seoul_commerce.collector.tqdm")
    def test_collect_dataset_updates_tqdm_with_collected_rows(
        self,
        tqdm_mock: Mock,
        load_contract_mock: Mock,
        _load_api_key_mock: Mock,
        fetch_page_mock: Mock,
        make_dataframe_mock: Mock,
        _save_csv_mock: Mock,
    ) -> None:
        contracts = {
            "api": {"max_rows_per_request": 2},
            "schema_version": "0.1.0",
        }
        dataset = {
            "required_columns": ["VALUE"],
            "period_filter_supported": False,
        }
        load_contract_mock.return_value = (contracts, dataset)
        fetch_page_mock.side_effect = [
            ([{"VALUE": "1"}, {"VALUE": "2"}], 3),
            ([{"VALUE": "3"}], 3),
        ]
        dataframe = pd.DataFrame([{"VALUE": "1"}, {"VALUE": "2"}, {"VALUE": "3"}])
        make_dataframe_mock.return_value = dataframe
        progress_bar = tqdm_mock.return_value

        result = collect_dataset(
            "sample",
            output="result.csv",
            session=Mock(),
        )

        self.assertIs(result, dataframe)
        tqdm_mock.assert_called_once_with(
            total=3,
            desc="sample",
            unit="행",
            dynamic_ncols=True,
        )
        self.assertEqual(
            progress_bar.update.call_args_list,
            [call(2), call(1)],
        )
        progress_bar.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
