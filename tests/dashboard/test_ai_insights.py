from __future__ import annotations

import json
import unittest

import pandas as pd
import requests

from seoul_commerce.dashboard.ai_insights import (
    AIInsight,
    AIInsightError,
    build_report_context,
    generate_ai_insight,
)
from seoul_commerce.dashboard.report import (
    CompetitionView,
    ContextView,
    CustomerView,
    GrowthView,
    ModelView,
    ReportView,
)


class _FakeResponse:
    def __init__(self, data: dict[str, object], status_code: int = 200) -> None:
        self._data = data
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError("request failed")
            error.response = self
            raise error

    def json(self) -> dict[str, object]:
        return self._data


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return self.response


def _sample_report() -> ReportView:
    growth = GrowthView(
        trend=pd.DataFrame(
            {
                "quarter_label": ["2025년 2분기", "2026년 1분기"],
                "total_sales_hundred_million": [9.0, 10.8],
                "monthly_sales_million": [30.0, 36.0],
                "total_store_count": [10, 10],
            }
        ),
        contributions=pd.DataFrame(
            {"component": ["점포 수", "점포당 거래건수"], "log_contribution": [2.0, 7.0], "scope": ["total", "both"]}
        ),
        growth_type_label="거래량 성장형",
        growth_type_color="green",
        summary_text="거래량 증가가 총매출 성장에 가장 크게 기여했습니다.",
        per_store_summary_text="점포당 매출도 증가했습니다.",
        current_total_sales_display="10.8억 원",
        current_monthly_sales_per_store_display="3,600만 원",
        total_sales_qoq_change=5.0,
        total_sales_yoy_change=12.0,
        sales_per_store_qoq_change=5.0,
        sales_per_store_yoy_change=10.0,
        total_sales_log_change=12.0,
        quarterly_trend_rate=3.0,
        sales_yoy_volatility=1.5,
        recent_growth_count=4,
        consecutive_growth_quarters=3,
    )
    customer = CustomerView(
        floating_yoy_change=8.0,
        store_yoy_change=2.0,
        demand_store_gap=6.0,
        distances=pd.DataFrame({"segment_label": ["연령"], "distance": [0.2]}),
        overrepresented=pd.DataFrame(
            {
                "segment_label": ["연령 · 30대"],
                "sales_floating_share_ratio": [2.0],
                "sales_share_percent": [40.0],
                "floating_share_percent": [20.0],
            }
        ),
        summary_text="유동인구가 점포 수보다 빠르게 증가했습니다.",
    )
    competition = CompetitionView(
        selected_group="높음",
        summary=pd.DataFrame(
            {
                "density_label": ["낮음", "높음"],
                "sample_count": [20, 20],
                "density_median": [1.5, 5.5],
                "total_sales_hundred_million": [1.0, 3.0],
                "sales_per_store_million": [10.0, 18.0],
            }
        ),
        summary_text="선택 상권은 점포 밀도 높음 그룹입니다.",
    )
    context = ContextView(
        metrics=pd.DataFrame(
            {
                "domain_label": ["경쟁"],
                "metric_label": ["점포 밀도"],
                "display_value": ["5.50개/ha"],
                "peer_percentile": [90.0],
            }
        ),
        top_metrics=("점포 밀도",),
        summary_text="점포 밀도 조건이 두드러집니다.",
    )
    model = ModelView(
        candidate="C",
        validation_log_r2=0.64,
        validation_mape=16.0,
        validation_rows=100,
        training_period="2024년 1분기~2025년 4분기 학습",
        actual_sales_display="1.08억 원",
        predicted_sales_display="1억 원",
        prediction_error_rate=8.0,
        global_terms=pd.DataFrame({"term_label": ["유동인구"], "importance_share": [0.6]}),
        local_terms=pd.DataFrame({"term_label": ["유동인구"], "sales_ratio_contribution": [0.12]}),
        summary_text="유동인구가 예측값을 높이는 주요 특성입니다.",
    )
    return ReportView(
        trade_area_code="A",
        trade_area_name="테스트상권",
        district_name="강남구",
        admin_dong_name="역삼동",
        industry_code="I",
        industry_name="한식음식점",
        quarter_label="2026년 1분기",
        report_available=True,
        availability_label="분석 가능",
        monthly_sales_display="3,600만 원",
        yoy_change=10.0,
        trend_change=20.0,
        trend_change_display="+20.0%",
        trend_direction="상승",
        peer_position_display="상위 14.6%",
        peer_context_display="한식음식점 · 발달상권 247곳 중 상위 14.6%",
        density_display="높음",
        store_count_display="10개",
        summary_text="매출은 높은 편이지만 경쟁 밀도도 높습니다.",
        tags=(),
        review_points=("임대료를 확인하세요.",),
        growth=growth,
        customer=customer,
        competition=competition,
        context=context,
        model=model,
    )


class AIInsightTests(unittest.TestCase):
    def test_report_context_contains_all_analysis_sections(self) -> None:
        context = build_report_context(_sample_report())

        self.assertEqual(context["selection"]["trade_area_name"], "테스트상권")
        self.assertIn("growth", context)
        self.assertIn("customers", context)
        self.assertIn("competition", context)
        self.assertIn("commercial_context", context)
        self.assertIn("prediction_model", context)

    def test_generate_ai_insight_uses_requested_model_and_schema(self) -> None:
        result = {
            "assessment": "주의 필요",
            "overall_conclusion": "성장세는 확인되지만 높은 경쟁 밀도를 함께 봐야 합니다.",
            "key_evidence": ["근거 1", "근거 2", "근거 3"],
            "opportunities": ["기회 1", "기회 2"],
            "risks": ["위험 1", "위험 2"],
            "recommended_actions": ["행동 1", "행동 2", "행동 3"],
            "data_limitations": ["추정매출 기반입니다."],
        }
        session = _FakeSession(
            _FakeResponse(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json.dumps(result, ensure_ascii=False)}
                            ],
                        }
                    ],
                }
            )
        )

        insight = generate_ai_insight({"sample": True}, "test-key", session=session)

        self.assertIsInstance(insight, AIInsight)
        self.assertEqual(insight.assessment, "주의 필요")
        self.assertEqual(len(session.calls), 1)
        request_json = session.calls[0]["json"]
        self.assertEqual(request_json["model"], "gpt-4o-mini")
        self.assertEqual(request_json["text"]["format"]["type"], "json_schema")
        self.assertEqual(
            session.calls[0]["headers"]["Authorization"],
            "Bearer test-key",
        )

    def test_authentication_error_has_safe_message(self) -> None:
        session = _FakeSession(_FakeResponse({}, status_code=401))

        with self.assertRaisesRegex(AIInsightError, "API 키"):
            generate_ai_insight({"sample": True}, "bad-key", session=session)


if __name__ == "__main__":
    unittest.main()
