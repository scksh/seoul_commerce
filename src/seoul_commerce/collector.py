"""서울 열린데이터광장 OpenAPI 데이터 수집 모듈.

Author:
    SeungHwan Kim

Created:
    2026-08-11

Description:
    서울 OpenAPI 데이터를 수집하고 검증하여 CSV 파일로 저장한다.
"""

import argparse
import os
import xml.parsers.expat
from datetime import datetime

import pandas as pd
import requests
import xmltodict
import yaml
from tqdm.auto import tqdm

from seoul_commerce.config import PROJECT_ROOT, load_api_key


CONTRACT_FILE = os.path.join(PROJECT_ROOT, "config", "data_contracts.yml")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def load_contract(dataset_name: str, contract_file: str = CONTRACT_FILE) -> tuple[dict, dict]:
    """
    YAML 파일에서 API 설정과 데이터셋 설정을 불러온다.

    Args:
        dataset_name: data_contracts.yml에 등록된 데이터셋 이름
        contract_file: 데이터 계약 YAML 파일 경로

    Returns:
        API 공통 설정과 선택한 데이터셋 설정

    Raises:
        ValueError: 데이터셋이 없거나 현재 수집할 수 없는 경우
    """
    try:
        with open(contract_file, encoding="utf-8") as file:
            contracts = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise ValueError(f"설정 파일을 찾을 수 없습니다: {contract_file}") from error
    except PermissionError as error:
        raise ValueError(f"설정 파일을 읽을 권한이 없습니다: {contract_file}") from error
    except UnicodeError as error:
        raise ValueError(f"설정 파일의 인코딩이 올바르지 않습니다: {contract_file}") from error
    except yaml.YAMLError as error:
        raise ValueError("YAML 설정 파일의 형식이 올바르지 않습니다.") from error
    except OSError as error:
        raise ValueError(f"설정 파일을 읽을 수 없습니다: {contract_file}") from error

    if not isinstance(contracts, dict):
        raise ValueError("YAML 설정 파일이 비어 있거나 구조가 올바르지 않습니다.")

    dataset = contracts.get("datasets", {}).get(dataset_name)

    if dataset is None:
        dataset_names = ", ".join(contracts.get("datasets", {}))
        raise ValueError(
            f"알 수 없는 데이터셋입니다: {dataset_name} "
            f"(사용 가능: {dataset_names})"
        )

    if dataset.get("collection_status") != "verified":
        raise ValueError(
            f"{dataset_name}은 현재 수집할 수 없습니다: "
            f"{dataset.get('collection_status')}"
        )

    return contracts, dataset


def build_url(
    base_url: str,
    api_key: str,
    response_format: str,
    service_name: str,
    start: int,
    end: int,
    quarter: str | None = None,
) -> str:
    """
    서울 OpenAPI 요청 URL을 생성한다.

    Args:
        base_url: 서울 OpenAPI 기본 주소
        api_key: 서울 OpenAPI 인증키
        response_format: 응답 형식(json 또는 xml)
        service_name: OpenAPI 서비스 이름
        start: 요청할 데이터의 시작 번호
        end: 요청할 데이터의 마지막 번호
        quarter: 수집할 기준 분기

    Returns:
        완성된 OpenAPI 요청 URL
    """
    url = (
        f"{base_url.rstrip('/')}/{api_key}/{response_format}/"
        f"{service_name}/{start}/{end}"
    )

    if quarter:
        url = f"{url}/{quarter}"

    return url


def check_result(result: dict) -> None:
    """OpenAPI 응답 코드가 정상인지 확인한다."""
    code = str(result.get("CODE", ""))

    if code != "INFO-000":
        message = result.get("MESSAGE", "알 수 없는 API 오류")
        raise ValueError(f"서울 OpenAPI 오류 {code}: {message}")


def parse_total_count(value) -> int:
    """API의 전체 데이터 수를 정수로 변환한다."""
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("API의 전체 데이터 수가 올바른 숫자가 아닙니다.") from error


def parse_json(response: requests.Response, service_name: str) -> tuple[list[dict], int]:
    """
    JSON 응답에서 데이터 행과 전체 데이터 수를 추출한다.

    Args:
        response: requests로 받은 JSON 응답
        service_name: OpenAPI 서비스 이름

    Returns:
        데이터 행 목록과 전체 데이터 수

    Raises:
        ValueError: 응답 코드나 전체 데이터 수가 올바르지 않은 경우
        KeyError: 응답에 요청한 서비스 데이터가 없는 경우
    """
    data = response.json()
    service_data = data.get(service_name)

    if service_data is None:
        result = data.get("RESULT")

        if isinstance(result, dict):
            check_result(result)

        raise KeyError(f"응답에 {service_name} 데이터가 없습니다.")

    check_result(service_data.get("RESULT", {}))
    rows = service_data.get("row", [])
    total_count = parse_total_count(service_data.get("list_total_count", 0))

    return rows, total_count


def parse_xml(response: requests.Response, service_name: str) -> tuple[list[dict], int]:
    """
    XML 응답을 딕셔너리로 변환하고 데이터 행을 추출한다.

    Args:
        response: requests로 받은 XML 응답
        service_name: OpenAPI 서비스 이름

    Returns:
        데이터 행 목록과 전체 데이터 수

    Raises:
        ValueError: JSON과 XML 요청이 모두 실패한 경우

    Raises:
        ValueError: 응답 코드가 오류이거나 서비스 데이터가 없는 경우
    """
    try:
        data = xmltodict.parse(response.text)
    except xml.parsers.expat.ExpatError as error:
        raise ValueError("OpenAPI XML 응답을 해석할 수 없습니다.") from error
    service_data = data.get(service_name)

    if service_data is None:
        error_data = next(iter(data.values()), {})
        check_result(error_data)
        raise ValueError(f"XML 응답에 {service_name} 데이터가 없습니다.")

    check_result(service_data.get("RESULT", {}))
    rows = service_data.get("row", [])

    # XML 응답이 한 행이면 딕셔너리로 반환되므로 리스트로 변경한다.
    if isinstance(rows, dict):
        rows = [rows]

    total_count = parse_total_count(service_data.get("list_total_count", 0))

    return rows, total_count


def fetch_page(
    session: requests.Session,
    api: dict,
    dataset: dict,
    api_key: str,
    start: int,
    end: int,
    quarter: str | None = None,
) -> tuple[list[dict], int]:
    """
    한 페이지를 JSON으로 요청하고 실패하면 XML로 다시 요청한다.

    Args:
        session: API 요청에 사용할 requests 세션
        api: OpenAPI 공통 설정
        dataset: 수집할 데이터셋 설정
        api_key: 서울 OpenAPI 인증키
        start: 요청할 데이터의 시작 번호
        end: 요청할 데이터의 마지막 번호
        quarter: 수집할 기준 분기

    Returns:
        데이터 행 목록과 전체 데이터 수
    """
    json_url = build_url(
        api["base_url"],
        api_key,
        "json",
        dataset["service_name"],
        start,
        end,
        quarter,
    )

    try:
        response = session.get(json_url, timeout=30)
        response.raise_for_status()
        return parse_json(response, dataset["service_name"])
    except (
        requests.RequestException,
        requests.exceptions.JSONDecodeError,
        KeyError,
    ):
        # JSON 요청이나 변환이 실패했을 때 같은 범위를 XML로 다시 요청한다.
        xml_url = build_url(
            api["base_url"],
            api_key,
            "xml",
            dataset["service_name"],
            start,
            end,
            quarter,
        )
        try:
            response = session.get(xml_url, timeout=30)
            response.raise_for_status()
            return parse_xml(response, dataset["service_name"])
        except requests.Timeout as error:
            raise ValueError("OpenAPI 요청 시간이 초과되었습니다.") from error
        except requests.ConnectionError as error:
            raise ValueError("서울 OpenAPI 서버에 연결할 수 없습니다.") from error
        except requests.HTTPError as error:
            raise ValueError(f"OpenAPI HTTP 오류가 발생했습니다: {error}") from error
        except requests.RequestException as error:
            raise ValueError(f"OpenAPI 요청에 실패했습니다: {error}") from error


def make_dataframe(
    rows: list[dict],
    required_columns: list[str],
    dataset: dict,
    schema_version: str,
    quarter: str | None = None,
) -> pd.DataFrame:
    """
    수집한 행을 필요한 컬럼으로 구성된 데이터프레임으로 만든다.

    Args:
        rows: API에서 수집한 데이터 행 목록
        required_columns: CSV에 저장할 필수 컬럼 목록
        dataset: 수집한 데이터셋 설정
        schema_version: 데이터 계약 버전
        quarter: 수집한 기준 분기

    Returns:
        필수 컬럼과 관리 컬럼이 포함된 데이터프레임

    Raises:
        ValueError: API 응답에 필수 컬럼이 없는 경우
    """
    response_columns = set()

    for row in rows:
        response_columns.update(row.keys())

    missing_columns = sorted(set(required_columns) - response_columns)

    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing_columns)}")

    dataframe = pd.DataFrame(rows, columns=required_columns)
    dataframe["SOURCE_DATASET_ID"] = dataset["dataset_id"]
    dataframe["COLLECTED_AT"] = datetime.now().astimezone().isoformat(timespec="seconds")

    if quarter:
        dataframe["SOURCE_BASE_DATE"] = quarter
    elif "STDR_YYQU_CD" in dataframe.columns:
        dataframe["SOURCE_BASE_DATE"] = dataframe["STDR_YYQU_CD"].astype("string")
    else:
        dataframe["SOURCE_BASE_DATE"] = dataset.get("reported_source_base_date", pd.NA)

    dataframe["SCHEMA_VERSION"] = schema_version

    return dataframe


def save_csv(dataframe: pd.DataFrame, file_path: str) -> None:
    """
    데이터프레임을 UTF-8 CSV 파일로 저장한다.

    Args:
        dataframe: 저장할 데이터프레임
        file_path: CSV 파일을 저장할 경로

    Raises:
        ValueError: 저장 경로나 파일 권한 문제로 저장하지 못한 경우
    """
    output_dir = os.path.dirname(file_path)

    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        dataframe.to_csv(file_path, index=False, encoding="utf-8-sig")
    except PermissionError as error:
        raise ValueError(
            "CSV 파일을 저장할 수 없습니다. "
            f"파일이 열려 있는지 확인해주세요: {file_path}"
        ) from error
    except OSError as error:
        raise ValueError(f"CSV 파일을 저장하지 못했습니다: {file_path}") from error


def collect_dataset(
    dataset_name: str,
    quarter: str | None = None,
    output: str | None = None,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """
    OpenAPI 데이터의 전체 페이지를 수집하고 CSV로 저장한다.

    Args:
        dataset_name: data_contracts.yml에 등록된 데이터셋 이름
        quarter: 수집할 기준 분기
        output: CSV 파일을 저장할 경로
        session: API 요청에 사용할 requests 세션

    Returns:
        수집이 완료된 데이터프레임

    Raises:
        ValueError: 설정이나 API 응답이 올바르지 않은 경우
    """
    contracts, dataset = load_contract(dataset_name)

    if quarter and not dataset.get("period_filter_supported"):
        raise ValueError(f"{dataset_name}은 분기 필터를 지원하지 않습니다.")

    api = contracts["api"]
    page_size = int(api["max_rows_per_request"])
    api_key = load_api_key()
    client = session or requests.Session()
    rows = []
    start = 1
    total_count = None
    progress_bar = None

    try:
        while total_count is None or start <= total_count:
            page_rows, total_count = fetch_page(
                client,
                api,
                dataset,
                api_key,
                start,
                start + page_size - 1,
                quarter,
            )

            if progress_bar is None:
                progress_bar = tqdm(
                    total=total_count,
                    desc=dataset_name,
                    unit="행",
                    dynamic_ncols=True,
                )

            rows.extend(page_rows)
            progress_bar.update(len(page_rows))

            if not page_rows:
                break

            start += page_size
    finally:
        if progress_bar is not None:
            progress_bar.close()

    dataframe = make_dataframe(
        rows,
        dataset["required_columns"],
        dataset,
        contracts["schema_version"],
        quarter,
    )

    if output is None:
        suffix = f"_{quarter}" if quarter else ""
        output = os.path.join(DEFAULT_OUTPUT_DIR, f"{dataset_name}{suffix}.csv")

    save_csv(dataframe, output)
    tqdm.write(f"저장 완료: {output} ({len(dataframe):,}행)")

    return dataframe


def main() -> None:
    """터미널에서 입력받은 데이터셋을 수집한다."""
    parser = argparse.ArgumentParser(
        description="서울 상권 OpenAPI 데이터를 CSV로 수집합니다."
    )
    parser.add_argument("dataset", help="data_contracts.yml의 데이터셋 이름")
    parser.add_argument("--quarter", help="기준 분기(예: 20252)")
    parser.add_argument("--output", help="저장할 CSV 경로")
    args = parser.parse_args()

    try:
        collect_dataset(args.dataset, args.quarter, args.output)
    except ValueError as error:
        print(f"수집 실패: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
