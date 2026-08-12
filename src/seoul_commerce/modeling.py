"""서울 상권 EBM 학습, 시간 검증, 설명 결과 저장 모듈.

최신 기준분기를 후보 선택에서 격리하고 과거 3개 분기 롤링 검증으로
후보를 고른 뒤, 기준분기 평가 모델과 전체 데이터 운영 모델을 저장한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import interpret
import joblib
import numpy as np
import pandas as pd
import sklearn
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from seoul_commerce.analysis_mart import (
    DEFAULT_CONFIG_FILE,
    PROJECT_ROOT,
    _atomic_write_json,
    _atomic_write_csv,
    _project_path,
    _read_csv,
    load_analysis_config,
    quarter_to_ordinal,
    validate_mart,
)


NUMERIC_FEATURES = [
    "log_floating_population", "log_resident_population", "log_working_population",
    "log_household_count", "weekend_floating_share", "lunch_floating_share",
    "evening_floating_share", "late_night_floating_share", "log_area_ha",
    "log_facility_density", "subway_station_count", "bus_stop_count",
    "log_store_density", "franchise_share", "time_index",
]
CATEGORICAL_FEATURES = ["industry", "district", "trade_area_type", "quarter"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_TYPES = ["continuous"] * len(NUMERIC_FEATURES) + ["nominal"] * len(CATEGORICAL_FEATURES)
TARGET_COLUMN = "log_quarterly_sales_per_store"
ID_COLUMNS = [
    "quarter_code", "trade_area_code", "trade_area_name", "industry_code", "industry_name"
]
FEATURE_GROUPS = {
    "log_floating_population": "demand", "log_resident_population": "hinterland",
    "log_working_population": "hinterland", "log_household_count": "housing",
    "weekend_floating_share": "usage_time", "lunch_floating_share": "usage_time",
    "evening_floating_share": "usage_time", "late_night_floating_share": "usage_time",
    "log_area_ha": "trade_area_structure", "log_facility_density": "access",
    "subway_station_count": "access", "bus_stop_count": "access",
    "log_store_density": "competition", "franchise_share": "competition",
    "time_index": "time_control", "industry": "industry_control",
    "district": "district_control", "trade_area_type": "type_control",
    "quarter": "season_control",
}
CONTROL_FEATURES = {"time_index", "industry", "district", "trade_area_type", "quarter"}


def _print_progress(message: str) -> None:
    """CLI 진행 메시지를 버퍼링 없이 출력한다."""
    print(message, flush=True)


def _format_elapsed(seconds: float) -> str:
    """경과시간을 시·분·초 형식으로 읽기 쉽게 변환한다."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}시간 {minutes:02d}분 {seconds:02d}초"
    if minutes:
        return f"{minutes}분 {seconds:02d}초"
    return f"{seconds}초"


def _fit_with_progress(
    model: ExplainableBoostingRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    label: str,
    progress: Callable[[str], None] | None,
    heartbeat_seconds: float = 30,
) -> float:
    """EBM 학습 중 일정 간격으로 경과시간을 알리고 총 학습시간을 반환한다."""
    started = time.perf_counter()
    if progress is None:
        model.fit(features, target)
        return time.perf_counter() - started

    progress(f"{label} 시작 · 학습 {len(features):,}행")
    finished = threading.Event()

    def report_heartbeat() -> None:
        while not finished.wait(heartbeat_seconds):
            elapsed = _format_elapsed(time.perf_counter() - started)
            progress(f"{label} 진행 중 · 경과 {elapsed}")

    reporter = threading.Thread(target=report_heartbeat, daemon=True)
    reporter.start()
    try:
        model.fit(features, target)
    finally:
        finished.set()
        reporter.join()
    elapsed = time.perf_counter() - started
    progress(f"{label} 완료 · {_format_elapsed(elapsed)}")
    return elapsed


def _time_feature_frame(segments: pd.DataFrame) -> pd.DataFrame:
    keys = ["quarter_code", "trade_area_code", "industry_code"]
    codes = ["sat", "sun", "11_14", "17_21", "21_24", "00_06"]
    required = keys + ["segment_code", "floating_population_share", "comparison_available"]
    missing = [column for column in required if column not in segments]
    if missing:
        raise ValueError(f"segment_comparison에 필수 컬럼이 없습니다: {', '.join(missing)}")
    rows = segments[
        segments["segment_code"].isin(codes) & segments["comparison_available"].eq(True)
    ][keys + ["segment_code", "floating_population_share"]]
    duplicates = rows.duplicated(keys + ["segment_code"], keep=False)
    if duplicates.any():
        raise ValueError(f"시간 구성 키가 중복되었습니다: {int(duplicates.sum()):,}행")
    pivot = rows.pivot(index=keys, columns="segment_code", values="floating_population_share")
    missing_codes = sorted(set(codes).difference(pivot.columns))
    if missing_codes:
        raise ValueError(f"시간 구성 코드가 없습니다: {', '.join(missing_codes)}")
    result = pd.DataFrame(index=pivot.index)
    result["weekend_floating_share"] = pivot["sat"] + pivot["sun"]
    result["lunch_floating_share"] = pivot["11_14"]
    result["evening_floating_share"] = pivot["17_21"]
    result["late_night_floating_share"] = pivot["21_24"] + pivot["00_06"]
    return result.reset_index()


def prepare_model_sample(
    commercial: pd.DataFrame,
    segments: pd.DataFrame,
    industry_codes: list[str] | None = None,
) -> pd.DataFrame:
    """전처리 자료에서 누수 변수를 제외한 EBM 완전 사례 표본을 만든다."""
    commercial = commercial.copy()
    if "facility_density" not in commercial and {
        "total_facility_count", "area_ha"
    }.issubset(commercial):
        valid_area = pd.to_numeric(commercial["area_ha"], errors="coerce").gt(0)
        commercial["facility_density"] = np.nan
        commercial.loc[valid_area, "facility_density"] = (
            pd.to_numeric(commercial.loc[valid_area, "total_facility_count"], errors="coerce")
            / pd.to_numeric(commercial.loc[valid_area, "area_ha"], errors="coerce")
        )
    required = ID_COLUMNS + [
        "trade_area_type_name", "district_name", "quarterly_sales_per_store",
        "total_floating_population", "total_resident_population", "total_working_population",
        "total_household_count", "area_ha", "facility_density", "subway_station_count",
        "bus_stop_count", "store_density", "franchise_share", "full_analysis_available",
    ]
    missing = [column for column in required if column not in commercial]
    if missing:
        raise ValueError(f"commercial_quarter에 필수 컬럼이 없습니다: {', '.join(missing)}")

    source = commercial[commercial["full_analysis_available"].eq(True)].copy()
    if industry_codes:
        source = source[source["industry_code"].isin([str(value) for value in industry_codes])]
    positive_columns = ["quarterly_sales_per_store", "area_ha", "store_density"]
    source = source[source[positive_columns].notna().all(axis=1)]
    source = source[source[positive_columns].gt(0).all(axis=1)]
    sample = source.merge(
        _time_feature_frame(segments),
        on=["quarter_code", "trade_area_code", "industry_code"], how="inner",
        validate="one_to_one",
    )

    sample[TARGET_COLUMN] = np.log(sample["quarterly_sales_per_store"])
    sample["log_floating_population"] = np.log1p(sample["total_floating_population"])
    sample["log_resident_population"] = np.log1p(sample["total_resident_population"])
    sample["log_working_population"] = np.log1p(sample["total_working_population"])
    sample["log_household_count"] = np.log1p(sample["total_household_count"])
    sample["log_area_ha"] = np.log(sample["area_ha"])
    sample["log_facility_density"] = np.log1p(sample["facility_density"])
    sample["log_store_density"] = np.log1p(sample["store_density"])
    ordinals = sample["quarter_code"].map(quarter_to_ordinal)
    sample["time_index"] = ordinals - ordinals.min()
    sample["industry"] = sample["industry_name"].astype("string")
    sample["district"] = sample["district_name"].astype("string")
    sample["trade_area_type"] = sample["trade_area_type_name"].astype("string")
    sample["quarter"] = sample["quarter_code"].str[-1] + "Q"

    sample = sample.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[TARGET_COLUMN, *FEATURE_COLUMNS]
    )
    sample[CATEGORICAL_FEATURES] = sample[CATEGORICAL_FEATURES].astype(str)
    return sample.sort_values(
        ["quarter_code", "industry_code", "trade_area_code"]
    ).reset_index(drop=True)


def rolling_origin_splits(
    sample: pd.DataFrame,
    reference_quarter: str,
    fold_count: int = 3,
) -> list[tuple[str, pd.Index, pd.Index]]:
    """기준분기 이전 최근 분기를 검증점으로 하는 확장식 시간 분할을 만든다."""
    quarters = sorted(
        sample.loc[sample["quarter_code"].lt(reference_quarter), "quarter_code"].unique(),
        key=quarter_to_ordinal,
    )
    if len(quarters) <= fold_count:
        raise ValueError(f"롤링 검증에 필요한 과거 분기가 부족합니다: {len(quarters)}개")
    validation_quarters = quarters[-fold_count:]
    splits: list[tuple[str, pd.Index, pd.Index]] = []
    for validation_quarter in validation_quarters:
        train_index = sample.index[sample["quarter_code"].lt(validation_quarter)]
        validation_index = sample.index[sample["quarter_code"].eq(validation_quarter)]
        if train_index.empty or validation_index.empty:
            raise ValueError(f"롤링 검증 표본이 비었습니다: {validation_quarter}")
        if sample.loc[train_index, "quarter_code"].max() >= validation_quarter:
            raise ValueError(f"미래 분기가 학습 표본에 포함되었습니다: {validation_quarter}")
        splits.append((str(validation_quarter), train_index, validation_index))
    return splits


def median_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """원단위 중앙 절대 오차율을 계산한다."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if np.any(actual <= 0):
        raise ValueError("중앙 절대 오차율의 실제값은 양수여야 합니다.")
    return float(np.median(np.abs(actual - predicted) / actual))


def _metrics(y_log: pd.Series, prediction_log: np.ndarray) -> dict[str, float]:
    actual = np.exp(y_log.to_numpy(dtype=float))
    predicted = np.exp(np.asarray(prediction_log, dtype=float))
    return {
        "log_r2": float(r2_score(y_log, prediction_log)),
        "log_rmse": float(np.sqrt(mean_squared_error(y_log, prediction_log))),
        "median_absolute_percentage_error": median_absolute_percentage_error(actual, predicted),
    }


def _new_model(parameters: dict[str, Any]) -> ExplainableBoostingRegressor:
    return ExplainableBoostingRegressor(
        feature_names=FEATURE_COLUMNS,
        feature_types=FEATURE_TYPES,
        **parameters,
    )


def evaluate_candidates(
    sample: pd.DataFrame,
    reference_quarter: str,
    candidates: dict[str, dict[str, Any]],
    fixed_parameters: dict[str, Any],
    fold_count: int = 3,
    execution_id: str = "evaluation",
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """각 EBM 후보를 동일한 롤링 원점 폴드에서 학습·평가한다."""
    rows: list[dict[str, Any]] = []
    splits = rolling_origin_splits(sample, reference_quarter, fold_count)
    total_fits = len(candidates) * len(splits)
    fit_number = 0
    for candidate, variable_parameters in candidates.items():
        parameters = {**fixed_parameters, **variable_parameters}
        for validation_quarter, train_index, validation_index in splits:
            fit_number += 1
            train = sample.loc[train_index]
            validation = sample.loc[validation_index]
            model = _new_model(parameters)
            label = (
                f"[후보 검증 {fit_number}/{total_fits}] "
                f"후보 {candidate} · 검증분기 {validation_quarter}"
            )
            training_seconds = _fit_with_progress(
                model, train[FEATURE_COLUMNS], train[TARGET_COLUMN], label, progress
            )
            train_metrics = _metrics(
                train[TARGET_COLUMN], model.predict(train[FEATURE_COLUMNS])
            )
            validation_metrics = _metrics(
                validation[TARGET_COLUMN], model.predict(validation[FEATURE_COLUMNS])
            )
            rows.append({
                "execution_id": execution_id,
                "candidate": str(candidate),
                "validation_fold": str(validation_quarter),
                "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                "training_start_quarter": str(train["quarter_code"].min()),
                "training_end_quarter": str(train["quarter_code"].max()),
                "validation_quarter": str(validation_quarter),
                "training_rows": len(train),
                "validation_rows": len(validation),
                "train_log_r2": train_metrics["log_r2"],
                "validation_log_r2": validation_metrics["log_r2"],
                "validation_log_rmse": validation_metrics["log_rmse"],
                "validation_median_absolute_percentage_error": validation_metrics[
                    "median_absolute_percentage_error"
                ],
                "train_validation_r2_gap": train_metrics["log_r2"] - validation_metrics["log_r2"],
                "training_seconds": training_seconds,
            })
            if progress is not None:
                progress(
                    f"{label} 성능 · 로그 R² {validation_metrics['log_r2']:.3f} · "
                    f"RMSE {validation_metrics['log_rmse']:.3f} · "
                    "중앙 절대 오차율 "
                    f"{validation_metrics['median_absolute_percentage_error']:.1%}"
                )
    return pd.DataFrame(rows)


def select_candidate(metrics: pd.DataFrame, candidate_order: list[str]) -> str:
    """중앙 오차율, 로그 R², 학습-검증 격차, 설정 순서로 후보를 선택한다."""
    required = [
        "candidate", "validation_median_absolute_percentage_error",
        "validation_log_r2", "train_validation_r2_gap",
    ]
    missing = [column for column in required if column not in metrics]
    if missing or metrics.empty:
        raise ValueError(f"후보 선택 지표가 부족합니다: {', '.join(missing)}")
    summary = metrics.groupby("candidate", observed=True).agg(
        median_error=("validation_median_absolute_percentage_error", "median"),
        median_r2=("validation_log_r2", "median"),
        median_gap=("train_validation_r2_gap", lambda values: values.abs().median()),
    ).reset_index()
    order = {candidate: index for index, candidate in enumerate(candidate_order)}
    summary["simplicity_order"] = summary["candidate"].map(order).fillna(len(order))
    winner = summary.sort_values(
        ["median_error", "median_r2", "median_gap", "simplicity_order"],
        ascending=[True, False, True, True],
    ).iloc[0]
    return str(winner["candidate"])


def _global_explanations(
    model: ExplainableBoostingRegressor,
    training: pd.DataFrame,
    execution_id: str,
) -> pd.DataFrame:
    importance = np.asarray(model.term_importances(importance_type="avg_weight"), dtype=float)
    importance_total = importance.sum()
    term_scores = model.eval_terms(training[FEATURE_COLUMNS])
    rows: list[dict[str, Any]] = []
    for term_index, feature_indexes in enumerate(model.term_features_):
        term = str(model.term_names_[term_index])
        feature_name = FEATURE_COLUMNS[feature_indexes[0]] if len(feature_indexes) == 1 else None
        high_low_difference = np.nan
        if feature_name in NUMERIC_FEATURES:
            values = training[feature_name].astype(float)
            low_cut = values.quantile(0.20)
            high_cut = values.quantile(0.80)
            low_score = term_scores[values.le(low_cut).to_numpy(), term_index].mean()
            high_score = term_scores[values.ge(high_cut).to_numpy(), term_index].mean()
            high_low_difference = 100 * np.expm1(high_score - low_score)
        rows.append({
            "execution_id": execution_id,
            "term": term,
            "term_type": "main" if len(feature_indexes) == 1 else "interaction",
            "domain": FEATURE_GROUPS.get(feature_name, "interaction"),
            "role": "control" if feature_name in CONTROL_FEATURES else "analysis_factor",
            "importance": importance[term_index],
            "importance_share": 100 * importance[term_index] / importance_total if importance_total else np.nan,
            "high_20_low_20_prediction_difference": high_low_difference,
            "term_index": term_index,
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def _local_explanations(
    model: ExplainableBoostingRegressor,
    holdout: pd.DataFrame,
    execution_id: str,
) -> pd.DataFrame:
    scores = model.eval_terms(holdout[FEATURE_COLUMNS])
    identity = holdout[ID_COLUMNS].reset_index(drop=True)
    rows: list[pd.DataFrame] = []
    for term_index, feature_indexes in enumerate(model.term_features_):
        part = identity.copy()
        part.insert(0, "execution_id", execution_id)
        part.insert(1, "base_quarter", part["quarter_code"])
        part["term"] = str(model.term_names_[term_index])
        part["term_type"] = "main" if len(feature_indexes) == 1 else "interaction"
        part["log_contribution"] = scores[:, term_index]
        part["sales_ratio_contribution"] = 100 * np.expm1(scores[:, term_index])
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _predictions(
    model: ExplainableBoostingRegressor,
    holdout: pd.DataFrame,
    execution_id: str,
) -> pd.DataFrame:
    prediction_log = model.predict(holdout[FEATURE_COLUMNS])
    result = holdout[ID_COLUMNS].copy().reset_index(drop=True)
    result.insert(0, "execution_id", execution_id)
    result["actual_sales_per_store"] = holdout["quarterly_sales_per_store"].to_numpy()
    result["predicted_sales_per_store"] = np.exp(prediction_log)
    result["prediction_error_rate"] = (
        result["predicted_sales_per_store"] / result["actual_sales_per_store"] - 1
    )
    return result


def _atomic_joblib_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        suffix=".tmp", prefix=f".{path.name}.", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _execution_id(reference_quarter: str, config: dict[str, Any]) -> str:
    payload = json.dumps(config["model"], sort_keys=True, ensure_ascii=False).encode("utf-8")
    return f"ebm-{reference_quarter}-{hashlib.sha256(payload).hexdigest()[:12]}"


def train_ebm_pipeline(
    config_path: str | os.PathLike[str] = DEFAULT_CONFIG_FILE,
    reference_quarter: str | None = None,
    model_dir: str | os.PathLike[str] | None = None,
    mart_dir: str | os.PathLike[str] | None = None,
    progress: Callable[[str], None] | None = _print_progress,
) -> dict[str, Path]:
    """후보 롤링 검증, 홀드아웃 평가, 전체 재학습과 결과 저장을 수행한다."""
    pipeline_started = time.perf_counter()
    if progress is not None:
        progress("[1/7] 분석 설정을 불러옵니다.")
    config = load_analysis_config(config_path)
    analysis = config["analysis"]
    reference = str(reference_quarter or analysis["reference_quarter"])
    paths = {name: _project_path(value) for name, value in config["paths"].items()}
    destination = Path(model_dir).resolve() if model_dir else paths["model_dir"]
    mart_destination = Path(mart_dir).resolve() if mart_dir else paths["mart_dir"]
    if progress is not None:
        progress("[2/7] 전처리 CSV를 읽습니다.")
    commercial = _read_csv(paths["commercial_quarter"], [
        "quarter_code", "trade_area_code", "industry_code"
    ])
    segments = _read_csv(paths["segment_comparison"], [
        "quarter_code", "trade_area_code", "industry_code", "segment_code"
    ])
    if progress is not None:
        progress(
            f"[2/7] CSV 로드 완료 · 상권×업종 {len(commercial):,}행 · "
            f"구성항목 {len(segments):,}행"
        )
        progress("[3/7] 누수 변수를 제외한 EBM 표본을 생성합니다.")
    sample = prepare_model_sample(
        commercial, segments, [str(value) for value in analysis["industry_codes"]]
    )
    if progress is not None:
        progress(
            f"[3/7] 표본 생성 완료 · {len(sample):,}행 · "
            f"{sample['quarter_code'].min()}~{sample['quarter_code'].max()} · "
            f"특징 {len(FEATURE_COLUMNS)}개"
        )
    execution_id = _execution_id(reference, config)
    model_config = config["model"]
    fixed = dict(model_config["fixed_parameters"])
    candidates = {str(name): dict(values) for name, values in model_config["candidates"].items()}
    if progress is not None:
        fold_count = int(analysis["rolling_validation_folds"])
        progress(
            f"[4/7] 후보 {len(candidates)}개 × 롤링 폴드 {fold_count}개 "
            f"({len(candidates) * fold_count}회) 검증을 시작합니다."
        )
    metrics = evaluate_candidates(
        sample, reference, candidates, fixed,
        int(analysis["rolling_validation_folds"]), execution_id, progress,
    )
    winner = select_candidate(metrics, list(candidates))
    parameters = {**fixed, **candidates[winner]}
    if progress is not None:
        candidate_summary = metrics.groupby("candidate", observed=True).agg(
            median_error=("validation_median_absolute_percentage_error", "median"),
            median_r2=("validation_log_r2", "median"),
        )
        for candidate, row in candidate_summary.iterrows():
            progress(
                f"[4/7] 후보 {candidate} 요약 · 중앙 오차율 "
                f"{row['median_error']:.1%} · 중앙 로그 R² {row['median_r2']:.3f}"
            )
        progress(f"[4/7] 선택 후보: {winner}")

    training = sample[sample["quarter_code"].lt(reference)].copy()
    holdout = sample[sample["quarter_code"].eq(reference)].copy()
    if training.empty or holdout.empty:
        raise ValueError("기준분기 학습 또는 홀드아웃 표본이 없습니다.")
    evaluation_model = _new_model(parameters)
    evaluation_seconds = _fit_with_progress(
        evaluation_model,
        training[FEATURE_COLUMNS],
        training[TARGET_COLUMN],
        f"[5/7] 후보 {winner} 홀드아웃 평가 모델",
        progress,
    )
    train_metrics = _metrics(
        training[TARGET_COLUMN], evaluation_model.predict(training[FEATURE_COLUMNS])
    )
    holdout_metrics = _metrics(
        holdout[TARGET_COLUMN], evaluation_model.predict(holdout[FEATURE_COLUMNS])
    )
    if progress is not None:
        progress(
            f"[5/7] {reference} 홀드아웃 성능 · {len(holdout):,}행 · "
            f"로그 R² {holdout_metrics['log_r2']:.3f} · "
            f"RMSE {holdout_metrics['log_rmse']:.3f} · "
            f"중앙 절대 오차율 {holdout_metrics['median_absolute_percentage_error']:.1%}"
        )
    metrics = pd.concat([metrics, pd.DataFrame([{
        "execution_id": execution_id,
        "candidate": winner,
        "validation_fold": f"holdout_{reference}",
        "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
        "training_start_quarter": str(training["quarter_code"].min()),
        "training_end_quarter": str(training["quarter_code"].max()),
        "validation_quarter": reference,
        "training_rows": len(training),
        "validation_rows": len(holdout),
        "train_log_r2": train_metrics["log_r2"],
        "validation_log_r2": holdout_metrics["log_r2"],
        "validation_log_rmse": holdout_metrics["log_rmse"],
        "validation_median_absolute_percentage_error": holdout_metrics[
            "median_absolute_percentage_error"
        ],
        "train_validation_r2_gap": train_metrics["log_r2"] - holdout_metrics["log_r2"],
        "training_seconds": evaluation_seconds,
    }])], ignore_index=True)

    if progress is not None:
        progress("[6/7] 전역·지역 설명과 평가 모델을 저장합니다.")
    global_mart = _global_explanations(evaluation_model, training, execution_id)
    local_mart = _local_explanations(evaluation_model, holdout, execution_id)
    predictions = _predictions(evaluation_model, holdout, execution_id)
    evaluation_payload = {
        "execution_id": execution_id,
        "model_version": "1.0.0",
        "selected_candidate": winner,
        "parameters": parameters,
        "feature_columns": FEATURE_COLUMNS,
        "feature_types": FEATURE_TYPES,
        "evaluation_model": evaluation_model,
        "model_metrics": metrics,
        "model_global": global_mart,
        "model_local": local_mart,
        "predictions": predictions,
    }
    evaluation_path = destination / f"ebm_evaluation_{reference}.joblib"
    _atomic_joblib_dump(evaluation_payload, evaluation_path)

    final_model = _new_model(parameters)
    _fit_with_progress(
        final_model,
        sample[FEATURE_COLUMNS],
        sample[TARGET_COLUMN],
        f"[7/7] 후보 {winner} 전체 데이터 최종 모델",
        progress,
    )
    final_path = destination / f"ebm_final_{reference}.joblib"
    _atomic_joblib_dump(final_model, final_path)
    reloaded = joblib.load(final_path)
    probe = sample[FEATURE_COLUMNS].head(min(100, len(sample)))
    if not np.allclose(final_model.predict(probe), reloaded.predict(probe), rtol=0, atol=1e-12):
        raise RuntimeError("저장 후 다시 불러온 최종 모델의 예측이 일치하지 않습니다.")

    metadata = {
        "execution_id": execution_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_quarter": reference,
        "selected_candidate": winner,
        "feature_columns": FEATURE_COLUMNS,
        "feature_types": FEATURE_TYPES,
        "parameters": parameters,
        "sample_period": [str(sample["quarter_code"].min()), str(sample["quarter_code"].max())],
        "training_rows": len(training),
        "holdout_rows": len(holdout),
        "all_rows": len(sample),
        "holdout_metrics": holdout_metrics,
        "versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "interpret": interpret.__version__,
        },
    }
    metadata_path = destination / "model_metadata.json"
    _atomic_write_json(metadata, metadata_path)

    mart_frames = {
        "model_metrics": metrics,
        "model_global": global_mart,
        "model_local": local_mart,
    }
    output_paths = {
        "evaluation_model": evaluation_path, "final_model": final_path, "metadata": metadata_path,
    }
    for name, frame in mart_frames.items():
        validate_mart(frame, name)
        path = mart_destination / config["mart_files"][name]
        _atomic_write_csv(frame, path)
        output_paths[name] = path
    if progress is not None:
        progress(
            f"[완료] 모델 학습과 저장을 마쳤습니다 · 총 경과 "
            f"{_format_elapsed(time.perf_counter() - pipeline_started)}"
        )
    return output_paths


def main() -> None:
    """EBM 학습 CLI 진입점."""
    parser = argparse.ArgumentParser(description="서울 상권 EBM 후보를 검증하고 모델을 저장합니다.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_FILE))
    parser.add_argument("--reference-quarter")
    parser.add_argument("--model-dir")
    parser.add_argument("--mart-dir")
    parser.add_argument("--quiet", action="store_true", help="진행상황 출력을 생략합니다.")
    args = parser.parse_args()
    outputs = train_ebm_pipeline(
        args.config, args.reference_quarter, args.model_dir, args.mart_dir,
        None if args.quiet else _print_progress,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
