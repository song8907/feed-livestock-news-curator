"""
llm_model_guard.py - 실행 중 '죽은 LLM 모델' 기록(issue_grouper/relevance_filter/llm_summarizer 공유).

OpenRouter에서 내려간 모델(404 등)을 배치마다 다시 호출하면 매 배치 수 초씩 낭비되고,
결국 전부 느린 최종 안전망(openrouter/free)으로 떨어져 단계별 마감에 걸림(요약 생략·안 묶음 처리).
영구 오류로 판단된 모델은 이번 실행 동안 체인에서 건너뜀. 429/5xx/타임아웃은 일시 오류로 보고 계속 시도.
"""

_PERMANENT_STATUS = {400, 401, 402, 403, 404}
_dead_models: dict[str, str] = {}


def is_dead(model_name: str) -> bool:
    return model_name in _dead_models


def report_failure(model_name: str, error: Exception, module: str) -> None:
    """HTTP 4xx(429 제외)면 이번 실행 동안 해당 모델 사용 중단(LM-01, 메일 조치필요 코드로 집계됨)."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    if status in _PERMANENT_STATUS and model_name not in _dead_models:
        _dead_models[model_name] = f"HTTP {status}"
        print(f"[{module}] 🔴 조치필요 [LM-01] - 모델('{model_name}') HTTP {status} - 서비스 종료/권한 문제로 판단, "
              f"이번 실행 동안 체인에서 제외 (OPENROUTER_MODEL* 변수 교체 필요)")


def live_chain(chain: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """죽은 모델을 뺀 체인. 전부 죽었으면 마지막(최종 안전망) 하나는 남겨 시도는 하게 함."""
    alive = [(role, m) for role, m in chain if m not in _dead_models]
    return alive or chain[-1:]


def extract_content(data: dict) -> str:
    """OpenRouter 응답 본문 추출. 추론형 모델은 content가 None으로 오기도 함 -> 명시적 오류로(다음 모델로 재시도)."""
    content = data["choices"][0]["message"].get("content")
    if not content or not content.strip():
        raise ValueError(f"빈 응답(content={content!r}, model={data.get('model', '?')})")
    return content.strip()
