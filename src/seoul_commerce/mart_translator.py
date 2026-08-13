"""영문 분석 마트를 보존하면서 별도의 한국어 CSV 복사본을 생성한다."""

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from seoul_commerce.config import PROJECT_ROOT


DEFAULT_INPUT_DIR = Path(PROJECT_ROOT) / "data" / "mart"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "korean"
DEFAULT_BASE_MAPPING = Path(PROJECT_ROOT) / "config" / "analysis_column_mapping_ko.yml"
DEFAULT_MART_MAPPING = Path(PROJECT_ROOT) / "config" / "mart_mapping_ko.yml"


def _load_yaml(path: Path | str) -> dict[str, Any]:
    mapping_path = Path(path)
    try:
        with mapping_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"매핑 설정을 읽을 수 없습니다: {mapping_path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"매핑 설정의 구조가 올바르지 않습니다: {mapping_path}")
    return config


def load_mart_mapping(
    base_mapping_file: Path | str = DEFAULT_BASE_MAPPING,
    mart_mapping_file: Path | str = DEFAULT_MART_MAPPING,
) -> dict[str, Any]:
    """분석 공통 컬럼과 마트 전용 매핑을 합치고 중복을 검증한다."""
    base = _load_yaml(base_mapping_file)
    mart = _load_yaml(mart_mapping_file)
    base_columns = base.get("columns")
    mart_columns = mart.get("columns")
    files = mart.get("files")
    if not all(isinstance(value, dict) for value in (base_columns, mart_columns, files)):
        raise ValueError("매핑 설정에 columns 또는 files가 없습니다.")
    columns = {**base_columns, **mart_columns}
    duplicate_targets = sorted({
        target for target in columns.values() if list(columns.values()).count(target) > 1
    })
    if duplicate_targets:
        raise ValueError(f"중복된 한국어 컬럼명이 있습니다: {', '.join(duplicate_targets)}")
    return {**mart, "columns": columns}


def _translate_delimited(value: str, mapping: dict[str, str], delimiter: str) -> str:
    return delimiter.join(mapping.get(part, part) for part in value.split(delimiter))


def translate_value(column: str, value: str, config: dict[str, Any]) -> str:
    """설명용 범주값과 EBM 항 이름을 한국어로 변환한다."""
    if value == "":
        return value
    if column == "term":
        terms = config.get("terms", {})
        return _translate_delimited(value, terms, " & ")
    column_mapping = config.get("values", {}).get(column, {})
    if column in {"analysis_exclusion_reason", "comparison_exclusion_reason"}:
        return _translate_delimited(value, column_mapping, ";")
    return column_mapping.get(value, value)


def translate_mart_csv(
    input_file: Path | str,
    output_file: Path | str,
    config: dict[str, Any],
) -> int:
    """수치와 행 순서를 유지한 채 한 개 마트의 헤더와 설명값을 변환한다."""
    source_path = Path(input_file).resolve()
    destination_path = Path(output_file).resolve()
    if source_path == destination_path:
        raise ValueError("원본 보호를 위해 입력 파일과 출력 파일은 달라야 합니다.")
    if not source_path.is_file():
        raise ValueError(f"입력 마트 파일을 찾을 수 없습니다: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            try:
                columns = next(reader)
            except StopIteration as error:
                raise ValueError(f"입력 마트 파일이 비어 있습니다: {source_path}") from error
            missing = [column for column in columns if column not in config["columns"]]
            if missing:
                raise ValueError(f"한국어 매핑에 없는 마트 컬럼입니다: {', '.join(missing)}")
            korean_columns = [config["columns"][column] for column in columns]
            if len(korean_columns) != len(set(korean_columns)):
                raise ValueError(f"{source_path.name}의 변환 결과에 중복 컬럼명이 있습니다.")
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8-sig", newline="", dir=destination_path.parent,
                prefix=f".{destination_path.name}.", suffix=".tmp", delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                # DuckDB 등 표본 기반 CSV 자동 감지기가 앞부분에서 따옴표를
                # 발견하지 못해 상권명 내부의 쉼표를 구분자로 오인하지 않도록
                # 모든 필드를 일관되게 따옴표로 감싼다.
                writer = csv.writer(destination, quoting=csv.QUOTE_ALL)
                writer.writerow(korean_columns)
                row_count = 0
                for row_number, row in enumerate(reader, start=2):
                    if len(row) != len(columns):
                        raise ValueError(
                            f"{source_path.name}의 {row_number}행 컬럼 수가 헤더와 다릅니다."
                        )
                    writer.writerow([
                        translate_value(column, value, config)
                        for column, value in zip(columns, row, strict=True)
                    ])
                    row_count += 1
        os.replace(temporary_path, destination_path)
        temporary_path = None
        return row_count
    except OSError as error:
        raise ValueError(f"한국어 마트를 저장하지 못했습니다: {destination_path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def translate_marts(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    base_mapping_file: Path | str = DEFAULT_BASE_MAPPING,
    mart_mapping_file: Path | str = DEFAULT_MART_MAPPING,
) -> list[tuple[Path, int]]:
    """설정에 등록된 모든 마트의 한국어 복사본을 별도 디렉터리에 생성한다."""
    source_dir = Path(input_dir).resolve()
    destination_dir = Path(output_dir).resolve()
    if source_dir == destination_dir:
        raise ValueError("원본 보호를 위해 입력 및 출력 디렉터리는 달라야 합니다.")
    config = load_mart_mapping(base_mapping_file, mart_mapping_file)
    results: list[tuple[Path, int]] = []
    for english_name, korean_name in config["files"].items():
        output_path = destination_dir / korean_name
        rows = translate_mart_csv(source_dir / english_name, output_path, config)
        results.append((output_path, rows))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="data/mart 원본을 유지하고 별도의 한국어 분석 마트를 생성합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-mapping", type=Path, default=DEFAULT_BASE_MAPPING)
    parser.add_argument("--mart-mapping", type=Path, default=DEFAULT_MART_MAPPING)
    args = parser.parse_args()
    try:
        results = translate_marts(
            args.input_dir, args.output_dir, args.base_mapping, args.mart_mapping
        )
    except ValueError as error:
        parser.exit(1, f"변환 실패: {error}\n")
    for output_path, row_count in results:
        print(f"저장 완료: {output_path} ({row_count:,}행)")


if __name__ == "__main__":
    main()
