"""원본 CSV의 컬럼명만 한국어로 바꾼 복사본을 생성한다."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import yaml

from seoul_commerce.config import PROJECT_ROOT


DEFAULT_MAPPING_FILE = Path(PROJECT_ROOT) / "config" / "column_mapping_ko.yml"
DEFAULT_INPUT_DIR = Path(PROJECT_ROOT) / "data" / "raw"
DEFAULT_OUTPUT_DIR = Path(PROJECT_ROOT) / "data" / "korean"
CONFIG_KEYS = {"schema_version", "rename_stage", "raw_column_policy", "common", "management"}


def load_mapping_config(mapping_file: Path | str = DEFAULT_MAPPING_FILE) -> dict:
    """컬럼 매핑 YAML을 읽고 기본 구조를 검증한다."""
    mapping_path = Path(mapping_file)

    try:
        with mapping_path.open(encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise ValueError(f"컬럼 매핑 파일을 찾을 수 없습니다: {mapping_path}") from error
    except PermissionError as error:
        raise ValueError(f"컬럼 매핑 파일을 읽을 권한이 없습니다: {mapping_path}") from error
    except UnicodeError as error:
        raise ValueError(f"컬럼 매핑 파일의 인코딩이 올바르지 않습니다: {mapping_path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"컬럼 매핑 YAML 형식이 올바르지 않습니다: {mapping_path}") from error
    except OSError as error:
        raise ValueError(f"컬럼 매핑 파일을 읽을 수 없습니다: {mapping_path}") from error

    if not isinstance(config, dict):
        raise ValueError("컬럼 매핑 파일이 비어 있거나 구조가 올바르지 않습니다.")

    for section in ("common", "management"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"컬럼 매핑 파일에 {section} 설정이 없습니다.")

    return config


def get_dataset_names(config: dict) -> list[str]:
    """매핑 설정에 등록된 데이터셋 이름을 반환한다."""
    return sorted(key for key, value in config.items() if key not in CONFIG_KEYS and isinstance(value, dict))


def build_column_mapping(config: dict, dataset_name: str) -> dict[str, str]:
    """공통·관리·데이터셋 매핑을 하나로 합쳐 검증한다."""
    dataset_mapping = config.get(dataset_name)

    if not isinstance(dataset_mapping, dict):
        available = ", ".join(get_dataset_names(config))
        raise ValueError(f"알 수 없는 데이터셋입니다: {dataset_name} (사용 가능: {available})")

    mapping = {
        **config["common"],
        **config["management"],
        **dataset_mapping,
    }

    invalid_entries = [
        source
        for source, target in mapping.items()
        if not isinstance(source, str) or not isinstance(target, str) or not source or not target
    ]
    if invalid_entries:
        raise ValueError(f"비어 있거나 문자열이 아닌 컬럼 매핑이 있습니다: {invalid_entries}")

    duplicate_targets = sorted(
        target for target in set(mapping.values()) if list(mapping.values()).count(target) > 1
    )
    if duplicate_targets:
        raise ValueError(f"중복된 한국어 컬럼명이 있습니다: {', '.join(duplicate_targets)}")

    return mapping


def translate_csv(
    input_file: Path | str,
    output_file: Path | str,
    dataset_name: str,
    mapping_file: Path | str = DEFAULT_MAPPING_FILE,
) -> int:
    """CSV 값은 유지하면서 헤더만 한국어로 바꿔 별도 파일에 저장한다.

    Returns:
        헤더를 제외하고 복사한 데이터 행 수
    """
    input_path = Path(input_file).resolve()
    output_path = Path(output_file).resolve()

    if input_path == output_path:
        raise ValueError("원본 보호를 위해 입력 파일과 출력 파일은 달라야 합니다.")
    if not input_path.is_file():
        raise ValueError(f"입력 CSV 파일을 찾을 수 없습니다: {input_path}")

    config = load_mapping_config(mapping_file)
    mapping = build_column_mapping(config, dataset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source)
            try:
                original_columns = next(reader)
            except StopIteration as error:
                raise ValueError(f"입력 CSV 파일이 비어 있습니다: {input_path}") from error

            duplicate_sources = sorted(
                column
                for column in set(original_columns)
                if original_columns.count(column) > 1
            )
            if duplicate_sources:
                raise ValueError(f"원본 CSV에 중복 컬럼명이 있습니다: {', '.join(duplicate_sources)}")

            missing_columns = [column for column in original_columns if column not in mapping]
            if missing_columns:
                raise ValueError(
                    f"{dataset_name} 매핑에 없는 컬럼입니다: {', '.join(missing_columns)}"
                )

            korean_columns = [mapping[column] for column in original_columns]
            if len(korean_columns) != len(set(korean_columns)):
                raise ValueError("변환 결과에 중복된 한국어 컬럼명이 있습니다.")

            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8-sig",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.writer(destination)
                writer.writerow(korean_columns)

                row_count = 0
                for row_number, row in enumerate(reader, start=2):
                    if len(row) != len(original_columns):
                        raise ValueError(
                            f"{input_path.name}의 {row_number}행 컬럼 수가 헤더와 다릅니다."
                        )
                    writer.writerow(row)
                    row_count += 1

        os.replace(temporary_path, output_path)
        temporary_path = None
        return row_count
    except (PermissionError, OSError) as error:
        raise ValueError(f"한국어 CSV 복사본을 저장하지 못했습니다: {output_path}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def translate_datasets(
    dataset_names: list[str] | None = None,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    mapping_file: Path | str = DEFAULT_MAPPING_FILE,
) -> list[tuple[Path, int]]:
    """선택한 데이터셋 또는 원본 디렉터리의 모든 CSV를 변환한다."""
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    config = load_mapping_config(mapping_file)

    if dataset_names is None:
        dataset_names = sorted(path.stem for path in source_dir.glob("*.csv"))
    if not dataset_names:
        raise ValueError("변환할 데이터셋이 없습니다.")

    results = []
    for dataset_name in dataset_names:
        # 파일을 만들기 전에 데이터셋 이름이 설정에 있는지 확인한다.
        build_column_mapping(config, dataset_name)
        output_path = destination_dir / f"{dataset_name}.csv"
        row_count = translate_csv(
            source_dir / f"{dataset_name}.csv",
            output_path,
            dataset_name,
            mapping_file,
        )
        results.append((output_path, row_count))

    return results


def main() -> None:
    """터미널에서 한국어 컬럼 CSV 복사본을 생성한다."""
    parser = argparse.ArgumentParser(
        description="data/raw 원본은 유지하고 한국어 컬럼명의 CSV 복사본을 생성합니다."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="변환할 데이터셋 이름(생략하면 data/raw의 모든 CSV)",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_FILE)
    args = parser.parse_args()

    try:
        results = translate_datasets(
            args.datasets or None,
            args.input_dir,
            args.output_dir,
            args.mapping_file,
        )
    except ValueError as error:
        parser.exit(1, f"변환 실패: {error}\n")

    for output_path, row_count in results:
        print(f"저장 완료: {output_path} ({row_count:,}행)")


if __name__ == "__main__":
    main()
