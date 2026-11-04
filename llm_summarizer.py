"""
llm_summarizer.py
(A) 자체 요약 생성 + (A-1) 얇은 재료 fallback 담당 모듈.
(B) 그룹핑 보조는 issue_grouper.stage3_llm_assist가 담당.
OpenRouter 모델 체인은 판단형(그룹핑/필터/재분류)과 별개로 요약 전용을 씀
(판단형은 정해진 JSON만 뱉으면 되고, 요약은 한국어 문장력이 중요해 기준이 다름).
API 키 없거나 LLM 실패 시 요약 생략, 원문 제목만 노출로 fallback.
"""

import os
import re
import time

import requests

import llm_model_guard
import trafilatura
from trafilatura.settings import use_config as _trafilatura_use_config

import issue_grouper as _ig


# --- 요약 전용 OpenRouter 모델 체인 ---
# issue_grouper의 OPENROUTER_MODEL(판단형 - 그룹핑/필터/재분류용)과는 별개.
# 기본값은 openai/gpt-oss-20b:free - 한국어 문장 생성 품질 기준으로 선정.
LLM_MODEL_OPENROUTER_SUMMARY = os.environ.get("OPENROUTER_MODEL_SUMMARY") or "openai/gpt-oss-20b:free"
LLM_MODEL_OPENROUTER_SUMMARY_2 = os.environ.get("OPENROUTER_MODEL_SUMMARY_2") or ""
LLM_MODEL_OPENROUTER_SUMMARY_3 = os.environ.get("OPENROUTER_MODEL_SUMMARY_3") or ""

_LLM_MODEL_CHAIN_SUMMARY_ROLES: list[tuple[str, str]] = []
if LLM_MODEL_OPENROUTER_SUMMARY:
    _LLM_MODEL_CHAIN_SUMMARY_ROLES.append(("1순위", LLM_MODEL_OPENROUTER_SUMMARY))
if LLM_MODEL_OPENROUTER_SUMMARY_2:
    _LLM_MODEL_CHAIN_SUMMARY_ROLES.append(("2순위", LLM_MODEL_OPENROUTER_SUMMARY_2))
if LLM_MODEL_OPENROUTER_SUMMARY_3:
    _LLM_MODEL_CHAIN_SUMMARY_ROLES.append(("3순위", LLM_MODEL_OPENROUTER_SUMMARY_3))
if "openrouter/free" not in [m for _, m in _LLM_MODEL_CHAIN_SUMMARY_ROLES]:
    _LLM_MODEL_CHAIN_SUMMARY_ROLES.append(("최종 안전망", "openrouter/free"))

LLM_MODEL_CHAIN_OPENROUTER_SUMMARY = [m for _, m in _LLM_MODEL_CHAIN_SUMMARY_ROLES]  # 하위호환/로그 표시용


_SYSTEM_PROMPT = (
    "너는 사료·축산업 뉴스 큐레이션 서비스의 요약 작성자다. 주어진 이슈(같은 "
    "사건을 다루는 기사 제목들과 참고 정보)를 보고 한국어로 2~3문장의 자체 "
    "요약을 작성하라. 원문을 그대로 옮기지 말고 핵심 내용만 새로 요약한다. "
    "확실하지 않은 수치나 사실은 "
    "임의로 만들어내지 말고, 주어진 제목/참고 정보에 있는 내용만 사용한다. "
    "확실하지 않으면 요약 대신 애매함을 그대로 표현하라. 전문용어·질병명·"
    "정부기관명·제도명 등 고유명사는 임의로 한글화하지 말고 영문 원어가 "
    "있으면 괄호로 병기한다. "
    "그리고 맨 첫 줄에 'TITLE: '로 시작해서, 주어진 기사 제목을 그대로 "
    "베끼지 말고 이슈 내용을 종합한 새 한국어 헤드라인을 직접 작성해 "
    "적어라(원문 기사 제목이 외국어라도 한국어로 작성한다) - 뉴스 헤드라인 "
    "답게 간결하고 전문적인 문체로 쓴다. 그 다음 줄부터 요약 문장을 이어서 "
    "작성한다. 이 형식 외에 다른 설명은 출력하지 않는다."
)

# TITLE: 접두사로 요약과 함께 새 헤드라인도 받음(titles[0] 대신). 국내/해외 구분 없이 항상 시도.
_TITLE_PATTERN = re.compile(r"\s*TITLE:\s*(.+?)\s*\n(.*)", re.DOTALL)


def _split_generated_title(text: str) -> tuple[str | None, str]:
    """LLM 응답 첫 줄의 'TITLE: ...'를 분리해 (헤드라인, 나머지 요약)로 반환. 형식 안 지켰으면 (None, 원본)."""
    m = _TITLE_PATTERN.match(text)
    if not m:
        return None, text.strip()
    return m.group(1).strip(), m.group(2).strip()


# 대용량 그룹(기사 수 > 임계값)은 배치별 사실 추출 -> 최종 합성의 2단계로 처리.
_LARGE_GROUP_THRESHOLD = 15
_BATCH_SIZE = 6  # 배치당 기사 수

_BATCH_EXTRACT_SYSTEM_PROMPT = (
    "너는 뉴스 기사 여러 건에서 핵심 사실만 추출하는 역할이다. 주어진 기사 "
    "제목/참고 정보들을 보고, 겹치지 않는 핵심 사실만 불릿 3~5개로 뽑아라. "
    "해석이나 의견, 요약 문장을 쓰지 말고 사실 나열만 하고, 확실하지 않은 "
    "내용은 만들어내지 말라. 각 불릿은 한 줄로, 다른 설명 없이 불릿만 "
    "출력한다."
)

_BODY_EXCERPT_CHARS = 600  # 기사 1건당 본문 발췌 길이 상한

# --- 재료 부족 기사 본문 추가 수집 (trafilatura, 범용 추출) ---
_BODY_FETCH_TIMEOUT_SECONDS = 10
_BODY_FETCH_MIN_LENGTH = 200
_BODY_FETCH_MAX_ARTICLES_PER_GROUP = 20
_BODY_FETCH_GROUP_TIME_BUDGET_SECONDS = 60

_TRAFILATURA_CONFIG = _trafilatura_use_config()
_TRAFILATURA_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", str(_BODY_FETCH_TIMEOUT_SECONDS))


def _build_user_prompt(item: dict) -> str:
    """이슈 하나(titles/articles)를 요약용 프롬프트로 변환. 그룹 크기 <= _LARGE_GROUP_THRESHOLD일 때만 호출."""
    titles = item.get("titles", [])
    lines = ["다음은 같은 이슈를 다룬 기사 제목들이다:"]
    for title in titles:
        lines.append(f"- {title}")

    context_lines = []
    for article in item.get("articles", []):
        source = article.get("source", "?")
        body = article.get("body")
        if body:
            context_lines.append(f"[{source}] 본문 일부: {body[:_BODY_EXCERPT_CHARS]}")
        description = article.get("description")
        if description:
            context_lines.append(f"[{source}] 설명: {description}")

    if context_lines:
        lines.append("\n참고 정보(그대로 인용하지 말고 참고만 할 것):")
        lines.extend(context_lines)

    lines.append("\n위 내용을 바탕으로 한국어 2~3문장 자체 요약을 작성하라.")
    return "\n".join(lines)


def _request_openrouter(system_prompt: str, user_prompt: str, api_key: str,
                         session: requests.Session, model_name: str) -> tuple[str, dict]:
    """반환: (응답 텍스트, 원본 응답 dict).
    OpenRouter 무료 티어 분당 상한 대응 - issue_grouper의 스로틀 함수를 재사용(그룹핑/요약이 같은 카운터를 공유해 더 정확)."""
    _ig._throttle_openrouter_free_tier()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "X-Title": _ig._OPENROUTER_X_TITLE,
    }
    body = {
        "model": model_name,
        "temperature": 0.3,
        "reasoning": {"exclude": True},  # 추론형 모델의 "생각 과정"이 content에 섞여 요약 대신 혼잣말이 나오는 것 방지(OpenRouter 공식 옵션)
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = session.post(_ig.LLM_API_URL_OPENROUTER, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return llm_model_guard.extract_content(data), data


def _call_llm(system_prompt: str, user_prompt: str, api_key: str, session: requests.Session) -> str | None:
    """
    요약 전용 모델 체인으로 LLM 1회 호출. 실패 시 None.
    _LLM_MODEL_CHAIN_SUMMARY_ROLES 순서로 재시도(LS-01~05).
    """
    data = None
    try:
        chain = llm_model_guard.live_chain(_LLM_MODEL_CHAIN_SUMMARY_ROLES)
        role_codes = {"1순위": "LS-01", "2순위": "LS-02", "3순위": "LS-03", "최종 안전망": "LS-04"}
        last_error: Exception | None = None
        for idx, (role, model_name) in enumerate(chain):
            try:
                if idx > 0:
                    print(f"[llm_summarizer] 🟡 주의 - 요약 생성 {role} 모델('{model_name}')로 재시도 "
                          f"({idx + 1}/{len(chain)})")
                text, data = _request_openrouter(system_prompt, user_prompt, api_key, session, model_name)
                if _is_suspicious_summary(text):
                    raise ValueError(f"응답이 요약이 아니라 안전성 판정/추론 과정으로 의심됨: {text[:80]!r}...")
                return text
            except Exception as e:
                last_error = e
                llm_model_guard.report_failure(model_name, e, "llm_summarizer")
                code = role_codes[role]
                is_final = idx == len(chain) - 1
                level = "🔴 조치필요" if is_final else "🟡 주의"
                next_note = "더 시도할 모델 없음" if is_final else "다음 후보 모델로 재시도"
                print(f"[llm_summarizer] {level} [{code}] - 요약 생성 {role} 모델('{model_name}') "
                      f"호출 실패 - {next_note}: {type(e).__name__} - {e!r}")
        raise last_error
    except Exception as e:
        snippet = (" ".join(str(data).split())[:200] + "...") if data is not None else "(응답을 아예 못 받음 - 요청/인증 단계에서 실패)"
        print(f"[llm_summarizer] 🔴 조치필요 [LS-05] - LLM 호출 실패: {type(e).__name__} - {e!r} "
              f"| 실제 응답: {snippet}")
        return None


def _build_batch_extract_prompt(batch_articles: list[dict]) -> str:
    """대용량 그룹 배치 하나를 사실 추출용 프롬프트로 변환(_summarize_large_group 1단계)."""
    lines = ["다음은 하나의 이슈를 다룬 기사들 중 일부(배치)이다:"]
    for article in batch_articles:
        title = article.get("title") or "(제목 없음)"
        source = article.get("source", "?")
        lines.append(f"- [{source}] {title}")
        body = article.get("body")
        if body:
            lines.append(f"  본문 일부: {body[:_BODY_EXCERPT_CHARS]}")
        description = article.get("description")
        if description:
            lines.append(f"  설명: {description}")
    lines.append("\n위 기사들에서 겹치지 않는 핵심 사실만 불릿 3~5개로 뽑아라.")
    return "\n".join(lines)


def _build_final_synthesis_prompt(item: dict, batch_facts: list[str]) -> str:
    """배치별 사실 추출 결과를 모아 최종 요약+헤드라인 프롬프트로 변환."""
    titles = item.get("titles", [])
    rep_titles = titles[:5]
    lines = ["다음은 같은 이슈를 다룬 기사 제목 일부(참고용):"]
    for t in rep_titles:
        lines.append(f"- {t}")
    if len(titles) > len(rep_titles):
        lines.append(f"(총 {len(titles)}건 중 일부만 표시, 전체 내용은 아래 사실 요약에 반영됨)")

    lines.append(f"\n아래는 전체 기사(총 {len(item.get('articles', []))}건)를 배치로 나눠 뽑은 핵심 사실 목록이다:")
    for i, facts in enumerate(batch_facts, start=1):
        lines.append(f"[배치 {i}]\n{facts}")

    lines.append("\n위 내용을 종합해서 한국어 2~3문장 자체 요약을 작성하라.")
    return "\n".join(lines)


def _summarize_large_group(item_for_prompt: dict, api_key: str, session: requests.Session) -> str | None:
    """
    _LARGE_GROUP_THRESHOLD 초과 그룹 전용 2단계 요약: 배치별 사실 추출 후 최종 합성.
    배치 일부 실패해도 나머지로 진행(LS-06), 전부 실패하면 None(LS-07).
    """
    articles = item_for_prompt.get("articles", [])
    batches = [articles[i:i + _BATCH_SIZE] for i in range(0, len(articles), _BATCH_SIZE)]

    batch_facts = []
    for batch_idx, batch in enumerate(batches, start=1):
        prompt = _build_batch_extract_prompt(batch)
        text = _call_llm(_BATCH_EXTRACT_SYSTEM_PROMPT, prompt, api_key, session)
        if text:
            batch_facts.append(text)
        else:
            print(f"[llm_summarizer] 🟡 주의 [LS-06] - 대용량 그룹({len(articles)}건) 배치 {batch_idx}/{len(batches)} "
                  f"사실 추출 실패 - 이 배치는 제외하고 나머지로 계속 진행")

    if not batch_facts:
        print(f"[llm_summarizer] 🔴 조치필요 [LS-07] - 대용량 그룹({len(articles)}건) 배치 전부 실패 - 요약 생략")
        return None

    final_prompt = _build_final_synthesis_prompt(item_for_prompt, batch_facts)
    return _call_llm(_SYSTEM_PROMPT, final_prompt, api_key, session)


# --- 요약 생략 시에도 제목만은 최소 번역 ---
# 단독 기사가 재료 부족으로 요약 자체가 생략돼도, 네이버 외 소스(원문이 외국어일 수 있음)면 제목만 한국어로 번역 시도. 실패해도 원문 제목 fallback.
_TRANSLATE_ONLY_SYSTEM_PROMPT = (
    "너는 뉴스 기사 제목을 한국어로 번역하는 역할이다. 주어진 원문 제목을 "
    "자연스러운 한국어 뉴스 헤드라인 한 줄로 번역하라. 이미 한국어 제목이면 "
    "그대로 돌려준다. 번역한 제목 한 줄만 출력하고, 다른 설명이나 따옴표는 "
    "붙이지 않는다."
)


def _translate_title_only(title: str, api_key: str, session: requests.Session) -> str | None:
    """제목 한 줄만 번역. 실패 시 None(호출부가 원문 제목 fallback)."""
    text = _call_llm(_TRANSLATE_ONLY_SYSTEM_PROMPT, title, api_key, session)
    if not text:
        return None
    return text.strip() or None


_REASONING_LEAK_MARKERS = (
    "user safety",
    "let me think", "let's think", "i need to", "i should",
    "okay, so", "first, i", "the user wants", "i'll",
)


def _is_suspicious_summary(text: str) -> bool:
    """
    요약이 아니라 안전성 판정 텍스트나 추론형 모델의 "생각 과정"이 온 것으로 의심되는 경우 감지.
    reasoning: exclude=true로 대부분 막히지만, 일부 모델은 이 옵션을 완벽히 안 지킬 수 있어 최소한의 안전장치로 둠.
    """
    lower = text.lower()
    if any(marker in lower for marker in _REASONING_LEAK_MARKERS):
        return True
    # 정상 요약은 한국어인데, 추론문은 대개 영어라 한글 비율이 지나치게 낮으면 의심
    if len(text) >= 40:
        hangul_count = sum(1 for ch in text if "\uac00" <= ch <= "\ud7a3")
        if hangul_count / len(text) < 0.2:
            return True
    return False


def _fetch_body_via_trafilatura(url: str) -> str | None:
    """범용 본문 추출 시도. 실패 시 조용히 None(재료 부족 경로로 흡수)."""
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True, config=_TRAFILATURA_CONFIG)
    except Exception:
        return None
    if not downloaded:
        return None
    try:
        extracted = trafilatura.extract(downloaded, favor_precision=True, include_comments=False,
                                         include_tables=False)
    except Exception:
        return None
    return extracted or None


def summarize_issue(item: dict, session: requests.Session | None = None) -> dict:
    """
    이슈 하나에 (A)/(A-1) 요약 로직 적용. 원본 item은 변경 없이 얕은 복사본 반환.

    반환값 추가 필드:
      summary: LLM 생성 요약, 또는 None(생략 시)
      summary_skipped_reason: 생략 사유(정상 요약 시 None)
      generated_title: 요약 성공 시 새 헤드라인, 실패/생략 시 None(deploy.py가 titles[0]로 fallback). 재료 부족으로 요약이 생략돼도 제목만은 _translate_title_only로 번역돼 채워지는 경우가 있음.
    """
    result = dict(item)
    result["generated_title"] = None
    titles = item.get("titles", [])
    articles = item.get("articles", [])
    item_for_prompt = item

    # (A-1) 단독 기사(그룹 크기 1)면서 재료가 얇으면 요약 생략.
    # 그룹은 있는 재료로 요약 시도(스킵 안 함).
    # 네이버는 body만 재료로 인정(description은 짧아도 기준을 넘기기 쉬워 제외).
    def _article_has_substantial_material(article: dict) -> bool:
        if len(article.get("body") or "") >= _BODY_FETCH_MIN_LENGTH:
            return True
        if article.get("source") == "네이버":
            return False
        return len(article.get("description") or "") >= 50

    has_substantial_material = any(_article_has_substantial_material(a) for a in articles)

    # 재료 얇은 기사 각각(다른 기사가 충분해도 무관) 최대 20건까지 본문 추가 수집 시도.
    needing_indices = [i for i, a in enumerate(articles) if not _article_has_substantial_material(a)]
    targets = needing_indices[:_BODY_FETCH_MAX_ARTICLES_PER_GROUP]

    if targets:
        label = "단독 기사" if len(titles) == 1 else f"그룹({len(titles)}건)"
        print(f"[llm_summarizer] 🟡 주의 [LS-08] - {label} 재료 부족 기사 {len(targets)}건 "
              f"(최대 {_BODY_FETCH_MAX_ARTICLES_PER_GROUP}건 한도) - 본문 추가 수집 시도")

        enriched_articles = list(articles)
        attempted = 0
        succeeded = 0
        group_urls = item.get("urls", [])
        start_time = time.monotonic()

        for idx in targets:
            if time.monotonic() - start_time >= _BODY_FETCH_GROUP_TIME_BUDGET_SECONDS:
                remaining = len(targets) - attempted
                print(f"[llm_summarizer] 🟡 주의 - {label} 본문 보강 시간 예산"
                      f"({_BODY_FETCH_GROUP_TIME_BUDGET_SECONDS}초) 초과 - 남은 {remaining}건 "
                      f"건너뛰고 지금까지 확보한 재료로 진행")
                break

            article = articles[idx]
            url = article.get("url") or (group_urls[idx] if idx < len(group_urls) else None)
            if not url:
                continue

            attempted += 1
            fetched_body = _fetch_body_via_trafilatura(url)
            if fetched_body and len(fetched_body) >= _BODY_FETCH_MIN_LENGTH:
                succeeded += 1
                enriched_article = dict(article)
                enriched_article["body"] = fetched_body
                enriched_articles[idx] = enriched_article

        print(f"[llm_summarizer] 본문 추가 수집 결과 [LS-09] - {label} {attempted}건 시도, "
              f"{succeeded}건 성공")

        if succeeded > 0:
            item_for_prompt = dict(item)
            item_for_prompt["articles"] = enriched_articles
            has_substantial_material = any(
                _article_has_substantial_material(a) for a in enriched_articles
            )

    # api_key는 (A-1) 스킵 분기에서도 제목 번역에 필요해 미리 조회.
    key_env_var = "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_env_var)

    if len(titles) == 1 and not has_substantial_material:
        result["summary"] = None
        result["summary_skipped_reason"] = "수집 데이터 부족"
        # 요약은 생략하되, 소스가 네이버가 아니면 제목만 번역 시도.
        source = articles[0].get("source") if articles else None
        if api_key and titles and source != "네이버":
            if session is not None:
                translated = _translate_title_only(titles[0], api_key, session)
            else:
                with requests.Session() as temp_session:
                    translated = _translate_title_only(titles[0], api_key, temp_session)
            if translated:
                result["generated_title"] = translated
                print(f"[llm_summarizer] 요약은 생략하되 제목만 번역 완료 - "
                      f"'{titles[0]}' -> '{translated}'")
        return result

    if not api_key:
        result["summary"] = None
        result["summary_skipped_reason"] = "요약 기능 일시 불가"
        return result

    # 대용량 그룹은 배치 2단계로, 그 이하는 1회 호출.
    total_articles = len(item_for_prompt.get("articles", []))
    if total_articles > _LARGE_GROUP_THRESHOLD:
        print(f"[llm_summarizer] 대용량 그룹({total_articles}건, 임계값 {_LARGE_GROUP_THRESHOLD} 초과) "
              f"- 배치로 나눠 2단계 요약 진행")
        if session is not None:
            summary_text = _summarize_large_group(item_for_prompt, api_key, session)
        else:
            with requests.Session() as temp_session:
                summary_text = _summarize_large_group(item_for_prompt, api_key, temp_session)
    else:
        user_prompt = _build_user_prompt(item_for_prompt)
        if session is not None:
            summary_text = _call_llm(_SYSTEM_PROMPT, user_prompt, api_key, session)
        else:
            with requests.Session() as temp_session:
                summary_text = _call_llm(_SYSTEM_PROMPT, user_prompt, api_key, temp_session)

    if not summary_text:
        result["summary"] = None
        result["summary_skipped_reason"] = "요약 생성 실패"
        return result

    generated_title, summary_text = _split_generated_title(summary_text)

    result["summary"] = summary_text
    result["summary_skipped_reason"] = None
    result["generated_title"] = generated_title
    return result


def summarize_top_issues(ranked_items: list[dict], label: str = "",
                          deadline: float | None = None) -> list[dict]:
    """
    ranked_items 전체에 summarize_issue 적용. 항목마다 즉시 로그 출력.
    deadline 넘으면 남은 항목은 "시간 예산 초과" 사유로 채워서 반환(기존 summary_skipped_reason 경로로 그대로 흡수).
    """
    results = []
    total = len(ranked_items)
    with requests.Session() as session:
        for i, item in enumerate(ranked_items, start=1):
            titles = item.get("titles", [])
            rep_title = titles[0] if titles else "(제목 없음)"
            prefix = f"[llm_summarizer] {label} " if label else "[llm_summarizer] "

            if deadline is not None and time.monotonic() >= deadline:
                remaining = total - i + 1
                print(f"{prefix}🟡 주의 [LS-10] - 시간 예산(파이프라인 기준 마감) 소진 - "
                      f"남은 {remaining}건은 요약 생략하고 원문 제목만 노출")
                results.extend(
                    {**dict(remaining_item), "generated_title": None, "summary": None,
                     "summary_skipped_reason": "시간 관계상 생략"}
                    for remaining_item in ranked_items[i - 1:]
                )
                break

            print(f"{prefix}({i}/{total}) '{rep_title}' (그룹 {len(titles)}건) - 처리 중...")

            result = summarize_issue(item, session)

            if result.get("summary"):
                print(f"{prefix}({i}/{total}) 요약 완료")
            else:
                print(f"{prefix}({i}/{total}) 요약 생략 - {result.get('summary_skipped_reason', '사유 불명')}")

            results.append(result)
    return results


def print_summaries(label: str, summarized: list[dict]) -> None:
    """결과 콘솔 출력. 요약 유무와 무관하게 원문 링크는 항상 노출."""
    print(f"\n=== {label} - LLM 요약 ===")
    for i, item in enumerate(summarized, start=1):
        titles = item.get("titles", [])
        rep_title = titles[0] if titles else "(제목 없음)"
        print(f"{i}. {rep_title}")
        if item.get("summary"):
            print(f"   요약: {item['summary']}")
        else:
            print(f"   (요약 생략 - {item.get('summary_skipped_reason', '사유 불명')})")
        urls = item.get("urls", [])
        shown_urls = ", ".join(urls[:3])
        more_note = f" 외 {len(urls) - 3}건" if len(urls) > 3 else ""
        print(f"   원문 링크: {shown_urls}{more_note}")