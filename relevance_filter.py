"""
relevance_filter.py
관련성 필터 - 정규화 직후, 이슈 그룹핑 전에 실행. 키워드 매칭만으론 못 거르는 오매칭(동음이의어, 기관명 일부로만 등장, 각주성 언급 등)을 LLM으로 판단.
issue_grouper.py의 3차 LLM 보조와 같은 패턴(OpenRouter 모델 체인, 배치, 검증, fallback) 재사용. 기본값 방향은 반대 - 애매하면 통과(true)가 안전(그룹핑 3차는 안 묶음이 안전).
WATT는 업계 전문지라 이 필터 없이 자동 통과. 소스별 컨텍스트 양 차이 있음 (네이버=description, WATT=body, GDELT=제목만).
"""

import json
import os
import time

import requests

import llm_model_guard

LLM_PROVIDER = "openrouter"  # 로그 표시용 고정값(더 이상 스위치 아님, OpenRouter만 사용)

LLM_MODEL_OPENROUTER = os.environ.get("OPENROUTER_MODEL") or "openrouter/free"
LLM_API_URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"

LLM_MODEL_OPENROUTER_2 = os.environ.get("OPENROUTER_MODEL_2") or ""
LLM_MODEL_OPENROUTER_3 = os.environ.get("OPENROUTER_MODEL_3") or ""

_LLM_MODEL_CHAIN_OPENROUTER_ROLES: list[tuple[str, str]] = []
if LLM_MODEL_OPENROUTER:
    _LLM_MODEL_CHAIN_OPENROUTER_ROLES.append(("1순위", LLM_MODEL_OPENROUTER))
if LLM_MODEL_OPENROUTER_2:
    _LLM_MODEL_CHAIN_OPENROUTER_ROLES.append(("2순위", LLM_MODEL_OPENROUTER_2))
if LLM_MODEL_OPENROUTER_3:
    _LLM_MODEL_CHAIN_OPENROUTER_ROLES.append(("3순위", LLM_MODEL_OPENROUTER_3))
if "openrouter/free" not in [m for _, m in _LLM_MODEL_CHAIN_OPENROUTER_ROLES]:
    _LLM_MODEL_CHAIN_OPENROUTER_ROLES.append(("최종 안전망", "openrouter/free"))

_LLM_MODEL_ROLE_ERROR_CODE = {
    "1순위": "RF-01",
    "2순위": "RF-02",
    "3순위": "RF-03",
    "최종 안전망": "RF-04",
}

LLM_MODEL_CHAIN_OPENROUTER = [m for _, m in _LLM_MODEL_CHAIN_OPENROUTER_ROLES]

_OPENROUTER_X_TITLE = "feed-livestock-news-relevance-filter"

BATCH_SIZE = 40
SNIPPET_MAX_CHARS = 150


_SYSTEM_PROMPT = (
    # 영어로 작성 - 특히 작은 무료 모델일수록 학습 데이터가 영어 위주라 형식 지시(JSON만 출력 등) 준수율이 더 안정적인 경향이 있고, 판단 대상(기사 제목) 자체도 다국어라 지시문을 영어로 통일하는 게 더 자연스럽다는 판단.
    "You are a relevance classifier for a feed and livestock industry news "
    "curation system. For each article, decide whether it substantively "
    "covers the \"feed industry\" or \"livestock industry\" (animal "
    "husbandry, disease control, distribution, policy, etc.) as its actual "
    "topic.\n\n"
    "RELEVANT (true):\n"
    "- Articles whose core topic is livestock (cattle/pigs/chickens/ducks "
    "etc.) rearing, disease, disease control, livestock product "
    "(meat/eggs/milk etc.) production, pricing, trade, or livestock-related "
    "government policy/legislation\n"
    "- Articles whose core topic is feed ingredients (grains such as corn, "
    "soybean meal, wheat; international grain market conditions), feed "
    "additives (amino acids, probiotics, enzymes, antibiotic alternatives, "
    "etc.), or feed manufacturing/distribution (feed mills, premixes, "
    "compound feed, TMR, etc.)\n\n"
    "NOT RELEVANT (false) - the following are mismatch patterns that have "
    "actually been confirmed to repeat, so filter them out with particular "
    "care:\n"
    "1. Homonyms: the search term spells the same but refers to something "
    "different (e.g. a disease name that is actually a person's name/"
    "nickname, or \"feed\" referring to the pet food industry - pet food is "
    "out of scope for this project)\n"
    "2. Appears only as part of a proper noun: the search term is merely "
    "part of an institution name, bill name, etc. (e.g. embedded in the "
    "name of a National Assembly standing committee - the article's actual "
    "subject is unrelated political news)\n"
    "3. Footnote-level mention: the article's core topic is something else "
    "entirely (e.g. semiconductor exports, overall price index), and "
    "livestock-related content appears only briefly as one line among many "
    "statistics/items\n"
    "4. Grain not used for feed: articles about corn, wheat, soybeans, etc. "
    "where the grain is not for feed but for food, bioenergy (ethanol), or "
    "the general agricultural commodity market broadly, with no mention of "
    "feed or livestock at all (e.g. a corn price article that only "
    "discusses food-ingredient supply like popcorn/cereal or ethanol "
    "feedstock)\n\n"
    "Examples that MUST be kept as relevant (true) - these types have "
    "actually been mistakenly filtered out before, so pay special "
    "attention:\n"
    "- Articles about consumer prices/supply of livestock products "
    "(eggs/meat/milk etc.) are relevant even if the title doesn't contain "
    "words like \"livestock\" or \"animal husbandry\". This is true even "
    "for titles with a light tone using slang or buzzwords. For example, "
    "an article discussing the cause of an egg price surge using the "
    "buzzword \"rocket eggs\" is an article about egg (a livestock product) "
    "prices, so it is true - do not judge it false just because the tone "
    "is light or the word \"livestock\" doesn't appear.\n"
    "- Before marking such an article false, first confirm it doesn't "
    "actually match any of NOT RELEVANT criteria 1-4 (homonym / part of "
    "proper noun / footnote mention / grain not for feed) - if it clearly "
    "doesn't match any of them, it is automatically relevant.\n\n"
    "The \"category\" value provided alongside each article is just a "
    "reference hint from automatic dictionary-keyword matching, not a "
    "confirmed answer - make your final judgment based on the actual "
    "title/summary content.\n\n"
    "If the judgment is ambiguous, answer true (conservative default - it "
    "is safer to let a few irrelevant articles through than to wrongly "
    "filter out a relevant one).\n\n"
    "Titles may be in different languages (Korean/English/other languages "
    "mixed) - judge by the same criteria regardless of language.\n\n"
    "Output only a JSON array with no other explanation. Each element must "
    "be in the form {\"id\": number, \"relevant\": true|false}, and id must "
    "exactly match the number of the input article."
)


def _snippet(article: dict) -> str | None:
    """판정 근거 스니펫. 네이버=description, WATT=body, GDELT=None(본문 없음)."""
    text = article.get("description") or article.get("body")
    if not text:
        return None
    return text[:SNIPPET_MAX_CHARS]


def _build_user_prompt(batch: list[dict]) -> str:
    lines = ["Judge whether each of the following articles is relevant to feed/livestock industry news.\n"]
    for idx, article in enumerate(batch, start=1):
        title = article.get("title", "")
        category = article.get("category", "기타")
        snippet = _snippet(article)
        snippet_part = f'"{snippet}"' if snippet else "(none - judge from title only)"
        lines.append(
            f'{idx}. Title: "{title}" / Category: {category} / Summary: {snippet_part}'
        )
    lines.append(
        f'\nThere are {len(batch)} articles total. Include the number above as "id" in each '
        f'element and answer with a JSON array only (e.g. [{{"id": 1, "relevant": true}}, '
        f'{{"id": 2, "relevant": false}}, ...]). Do not omit any id or change the order.'
    )
    return "\n".join(lines)


def _snippet_for_log(text: str, limit: int = 200) -> str:
    """LLM 원본 응답을 로그용으로 잘라서 반환."""
    if not text:
        return "(빈 응답)"
    flat = " ".join(text.split())
    return flat[:limit] + ("..." if len(flat) > limit else "")


# --- OpenRouter 무료 티어 분당 상한 대응 ---
# issue_grouper.py와 동일(분당 20회 상한 대응). 모듈별 별도 카운터 - 순서대로만 실행돼서 문제없음.
_OPENROUTER_MIN_INTERVAL_SECONDS = 1.5  # 분당 20회 상한(3.0초 간격) 기준보다 빠름 - 담당자 요청으로 하향
_openrouter_last_request_at = 0.0


def _throttle_openrouter_free_tier() -> None:
    """직전 OpenRouter 요청과의 간격이 _OPENROUTER_MIN_INTERVAL_SECONDS 미만이면 그만큼 대기."""
    global _openrouter_last_request_at
    elapsed = time.monotonic() - _openrouter_last_request_at
    wait = _OPENROUTER_MIN_INTERVAL_SECONDS - elapsed
    if wait > 0:
        time.sleep(wait)
    _openrouter_last_request_at = time.monotonic()


def _request_openrouter(system_prompt: str, user_prompt: str, api_key: str,
                         session: requests.Session, model_name: str) -> str:
    _throttle_openrouter_free_tier()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "X-Title": _OPENROUTER_X_TITLE,
    }
    body = {
        "model": model_name,
        "temperature": 0,
        "reasoning": {"exclude": True},  # 추론형 모델의 "생각 과정"이 content에 섞여 JSON 파싱을 깨는 것 방지(OpenRouter 공식 옵션, 전 모델 지원)
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = session.post(LLM_API_URL_OPENROUTER, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return llm_model_guard.extract_content(data)


def _request_llm_text(system_prompt: str, user_prompt: str, api_key: str, session: requests.Session,
                       label: str, validate=None):
    """
    모델 체인(1~3순위 -> 최종 안전망) 순으로 재시도(RF-01~10). validate 콜백 넘기면 형식 이상도 재시도 대상으로 취급.
    """
    chain = llm_model_guard.live_chain(_LLM_MODEL_CHAIN_OPENROUTER_ROLES)
    last_error: Exception | None = None
    for idx, (role, model_name) in enumerate(chain):
        is_final = idx == len(chain) - 1
        try:
            if idx > 0:
                print(f"[relevance_filter] 🟡 주의 - {label} {role} 모델('{model_name}')로 재시도 "
                      f"({idx + 1}/{len(chain)})")
            text = _request_openrouter(system_prompt, user_prompt, api_key, session, model_name)
            return validate(text, is_final) if validate else text
        except Exception as e:
            last_error = e
            llm_model_guard.report_failure(model_name, e, "relevance_filter")
            code = _LLM_MODEL_ROLE_ERROR_CODE[role]
            level = "🔴 조치필요" if is_final else "🟡 주의"
            next_note = "더 시도할 모델 없음 - 이 배치 전체 안전 처리" if is_final else "다음 후보 모델로 재시도"
            print(f"[relevance_filter] {level} [{code}] - {label} {role} 모델('{model_name}') "
                  f"호출/응답 검증 실패 - {next_note}: {type(e).__name__} - {e!r}")
    raise last_error


def _call_llm(batch: list[dict], api_key: str, session: requests.Session) -> list[bool] | None:
    """
    batch 각각의 relevant 판정. 실패 시 None(filter_articles가 그 배치 전부 통과).
    id 기반 부분 복구, 애매하면 통과(true)가 기본값.
    """
    user_prompt = _build_user_prompt(batch)

    def _validate(text: str, is_final: bool) -> list[bool]:
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
        parsed = json.loads(cleaned.strip())

        if not isinstance(parsed, list) or not parsed:
            actual = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
            raise ValueError(f"리스트가 아니거나 비어있음(실제 {actual}) | 실제 응답: {_snippet_for_log(text)}")

        by_id: dict[int, bool] = {}
        for item in parsed:
            try:
                by_id[int(item["id"])] = bool(item["relevant"])
            except (KeyError, TypeError, ValueError):
                continue

        missing = [idx for idx in range(1, len(batch) + 1) if idx not in by_id]
        if missing and not is_final:
            raise ValueError(f"id {missing} 누락(기대 {len(batch)}건 중 {len(missing)}건) - 다음 모델로 재시도")

        results = [by_id.get(idx, True) for idx in range(1, len(batch) + 1)]
        if missing:
            print(f"[relevance_filter] 🟡 주의 [RF-05] - 관련성 필터 최종 안전망까지 갔지만 id {missing} "
                  f"여전히 누락(기대 {len(batch)}건 중 {len(missing)}건) - 그 항목들만 통과 처리, "
                  f"나머지 {len(batch) - len(missing)}건은 정상 판정 사용")
        return results

    try:
        return _request_llm_text(_SYSTEM_PROMPT, user_prompt, api_key, session, "관련성 필터", validate=_validate)
    except Exception as e:
        print(f"[relevance_filter] 🔴 조치필요 [RF-06] - LLM({LLM_PROVIDER}) 호출/파싱 실패 - 이 배치"
              f"({len(batch)}건) 전부 통과 처리: {type(e).__name__} - {e!r}")
        return None


# ---------------------------------------------------------------------------
# 카테고리 재분류
# ---------------------------------------------------------------------------
# keyword_tagger(사전 매칭)와 relevance_filter(LLM 관련성 판단)는 기준이 달라, 사전엔 안 걸려 category="기타"인데 관련성은 확정된 기사가 생길 수 있음(카테고리별 Top N에 영원히 못 들어감). 그 공백을 여기서 메움.
# 관련성 필터 스키마는 안 건드리고, category="기타"인 관련 기사만 별도 배치로 다시 LLM에 물음.

CATEGORY_RECLASSIFY_SYSTEM_PROMPT_TEMPLATE = (
    "You are a categorization assistant for a feed and livestock industry "
    "news curation system. Each article below was not matched by "
    "dictionary-based keyword tagging and is currently labeled \"기타\" "
    "(uncategorized), but it has already been confirmed relevant to the "
    "feed/livestock industry by a separate relevance check. Assign the "
    "single best-fitting category for each article from the following "
    "exact list. Respond with the Korean label written exactly as shown "
    "below, character for character - do not translate, abbreviate, or "
    "modify it in any way:\n\n"
    "{category_list}\n\n"
    "If none of the categories are a good fit, respond with \"기타\" "
    "(i.e. leave it uncategorized) rather than forcing a poor match.\n\n"
    "Output only a JSON array with no other explanation. Each element must "
    "be in the form {{\"id\": number, \"category\": \"<one label from the "
    "list above, or 기타>\"}}, and id must exactly match the number of the "
    "input article."
)


def _build_category_user_prompt(batch: list[dict]) -> str:
    lines = ["Assign the best-fitting category to each of the following articles.\n"]
    for idx, article in enumerate(batch, start=1):
        title = article.get("title", "")
        snippet = _snippet(article)
        snippet_part = f'"{snippet}"' if snippet else "(none - judge from title only)"
        lines.append(f'{idx}. Title: "{title}" / Summary: {snippet_part}')
    lines.append(
        f'\nThere are {len(batch)} articles total. Include the number above as "id" in each '
        f'element and answer with a JSON array only. Do not omit any id or change the order.'
    )
    return "\n".join(lines)


def _call_category_llm(batch: list[dict], api_key: str, session: requests.Session,
                        category_choices: list[str], system_prompt: str) -> list[str] | None:
    """
    _call_llm과 같은 패턴의 재분류용 버전. 응답은 category_choices 중 하나(또는 "기타") 문자열이어야 하며, 목록에 없는 값은 무시(기타 유지).
    """
    user_prompt = _build_category_user_prompt(batch)
    valid_choices = set(category_choices) | {"기타"}

    def _validate(text: str, is_final: bool) -> list[str]:
        cleaned = text
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            cleaned = cleaned[4:] if cleaned.startswith("json") else cleaned
        parsed = json.loads(cleaned.strip())

        if not isinstance(parsed, list) or not parsed:
            actual = len(parsed) if isinstance(parsed, list) else type(parsed).__name__
            raise ValueError(f"리스트가 아니거나 비어있음(실제 {actual}) | 실제 응답: {_snippet_for_log(text)}")

        by_id: dict[int, str] = {}
        for item in parsed:
            try:
                category = str(item["category"])
                idx = int(item["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if category not in valid_choices:
                continue
            by_id[idx] = category

        missing = [idx for idx in range(1, len(batch) + 1) if idx not in by_id]
        if missing and not is_final:
            raise ValueError(f"id {missing} 누락/형식 이상(기대 {len(batch)}건 중 {len(missing)}건) - 다음 모델로 재시도")

        results = [by_id.get(idx, "기타") for idx in range(1, len(batch) + 1)]
        if missing:
            print(f"[relevance_filter] 🟡 주의 [RF-07] - 카테고리 재분류 최종 안전망까지 갔지만 id {missing} "
                  f"여전히 누락(기대 {len(batch)}건 중 {len(missing)}건) - 그 항목들만 "
                  f"'기타' 유지, 나머지 {len(batch) - len(missing)}건은 정상 판정 사용")
        return results

    try:
        return _request_llm_text(system_prompt, user_prompt, api_key, session, "카테고리 재분류", validate=_validate)
    except Exception as e:
        print(f"[relevance_filter] 🔴 조치필요 [RF-08] - 카테고리 재분류 LLM({LLM_PROVIDER}) 호출/파싱 실패 - "
              f"이 배치({len(batch)}건) 전부 '기타' 유지: {type(e).__name__} - {e!r}")
        return None


# ---------------------------------------------------------------------------
# 그룹 단위 - 그룹 대표 기사(group[0]) 1건만 LLM 판단, 그룹 전체를 유닛으로 통과/제외·재분류.
# LLM 호출량이 원본 기사 수가 아니라 고유 이슈(그룹) 수 기준으로 줄어듦.
# ---------------------------------------------------------------------------

def filter_groups(groups: list[list[dict]], deadline: float | None = None) -> list[list[dict]]:
    """
    이슈 그룹 리스트 중 대표 기사가 "관련 없다"고 확정된 그룹만 통째로 제외.
    API 키 없거나 전부 실패하면 원본 그대로 반환(안전 기본값).
    WATT 소스가 대표인 그룹은 filter_articles와 동일하게 자동 통과.

    deadline: filter_articles와 동일한 파이프라인 기준 절대 마감.
    """
    if not groups:
        return groups

    watt_groups = [g for g in groups if g[0].get("source") not in ("네이버", "GDELT")]
    llm_target_groups = [g for g in groups if g[0].get("source") in ("네이버", "GDELT")]

    if watt_groups:
        print(f"[relevance_filter] WATT 소스가 대표인 그룹 {len(watt_groups)}개는 업계 전문지 특성상 "
              f"LLM 호출 없이 자동 통과")

    if not llm_target_groups:
        return watt_groups

    key_env_var = "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_env_var)
    if not api_key:
        print(f"[relevance_filter] 🔴 조치필요 [RF-09] - {key_env_var} 없음(LLM_PROVIDER={LLM_PROVIDER}) - "
              f"관련성 필터 생략, {len(llm_target_groups)}개 그룹(네이버/GDELT 대표) 전부 통과")
        return groups

    model_desc = " -> ".join(LLM_MODEL_CHAIN_OPENROUTER)
    print(f"[relevance_filter] 관련성 필터(그룹 단위) 시작 - provider={LLM_PROVIDER}, "
          f"model={model_desc}, 대상 {len(llm_target_groups)}개 그룹(대표 기사 1건씩, "
          f"WATT 대표 {len(watt_groups)}개 제외)")

    kept = list(watt_groups)
    dropped_samples = []
    total_batches = (len(llm_target_groups) + BATCH_SIZE - 1) // BATCH_SIZE
    with requests.Session() as session:
        for batch_num, i in enumerate(range(0, len(llm_target_groups), BATCH_SIZE), start=1):
            if deadline is not None and time.monotonic() >= deadline:
                remaining = llm_target_groups[i:]
                kept.extend(remaining)
                print(f"[relevance_filter] 🟡 주의 [RF-10] - 시간 예산(파이프라인 기준 마감) 소진 - "
                      f"남은 {len(remaining)}개 그룹은 필터링 없이 전부 통과 처리하고 중단")
                break
            batch_groups = llm_target_groups[i:i + BATCH_SIZE]
            batch_representatives = [g[0] for g in batch_groups]
            print(f"[relevance_filter] 그룹 배치 {batch_num}/{total_batches} 처리 중 ({len(batch_groups)}개 그룹)")
            results = _call_llm(batch_representatives, api_key, session)
            if results is None:
                kept.extend(batch_groups)
                continue
            for group, relevant in zip(batch_groups, results):
                if relevant:
                    kept.append(group)
                else:
                    dropped_samples.append(group[0].get("title", ""))

    dropped_count = len(groups) - len(kept)
    print(f"[relevance_filter] 관련성 필터(그룹 단위) 완료 - 전체 {len(groups)}개 그룹(WATT 자동통과 "
          f"{len(watt_groups)}개 포함) 중 {dropped_count}개 제외, {len(kept)}개 유지")
    if dropped_samples:
        sample_n = min(10, len(dropped_samples))
        print(f"[relevance_filter] 제외된 그룹 대표 제목 샘플 (최대 {sample_n}건):")
        for title in dropped_samples[:sample_n]:
            print(f"   - {title}")

    return kept


def recategorize_uncategorized_groups(groups: list[list[dict]], deadline: float | None = None) -> list[list[dict]]:
    """
    filter_groups() 통과했지만 대표 기사 category가 "기타"인 그룹만 골라 대표 기사로 LLM 재분류.
    재분류되면(기타가 아닌 카테고리로 확정되면) 그 그룹 안에서 category가 "기타"인 멤버 전원에게 같은 카테고리를 적용한다.
     - 이미 사전 매칭으로 다른 카테고리가 붙은 멤버는 그 신호를 존중해 안 건드림.
    API 키 없거나 전부 실패해도 원본 그대로 반환.

    deadline: recategorize_uncategorized와 동일한 파이프라인 기준 절대 마감.
    """
    targets = [g for g in groups if g[0].get("category") == "기타"]
    if not targets:
        return groups

    key_env_var = "OPENROUTER_API_KEY"
    api_key = os.environ.get(key_env_var)
    if not api_key:
        print(f"[relevance_filter] 🔴 조치필요 [RF-11] - {key_env_var} 없음(LLM_PROVIDER={LLM_PROVIDER}) - "
              f"카테고리 재분류 생략, {len(targets)}개 그룹 '기타' 그대로 유지")
        return groups

    import keyword_tagger
    category_choices = list(keyword_tagger.CATEGORY_KEYWORDS.keys())
    category_list_text = "\n".join(f"- {c}" for c in category_choices)
    system_prompt = CATEGORY_RECLASSIFY_SYSTEM_PROMPT_TEMPLATE.format(category_list=category_list_text)

    model_desc = " -> ".join(LLM_MODEL_CHAIN_OPENROUTER)
    print(f"[relevance_filter] 카테고리 재분류(그룹 단위) 시작 - provider={LLM_PROVIDER}, "
          f"model={model_desc}, 대상 {len(targets)}개 그룹(대표 기사가 '기타')")

    reclassified_count = 0
    total_batches = (len(targets) + BATCH_SIZE - 1) // BATCH_SIZE
    with requests.Session() as session:
        for batch_num, i in enumerate(range(0, len(targets), BATCH_SIZE), start=1):
            if deadline is not None and time.monotonic() >= deadline:
                remaining = len(targets) - i
                print(f"[relevance_filter] 🟡 주의 [RF-12] - 시간 예산(파이프라인 기준 마감) 소진 - "
                      f"남은 {remaining}개 그룹은 재분류 없이 '기타' 유지하고 중단")
                break
            batch_groups = targets[i:i + BATCH_SIZE]
            batch_representatives = [g[0] for g in batch_groups]
            print(f"[relevance_filter] 카테고리 재분류(그룹) 배치 {batch_num}/{total_batches} "
                  f"처리 중 ({len(batch_groups)}개 그룹)")
            results = _call_category_llm(batch_representatives, api_key, session, category_choices, system_prompt)
            if results is None:
                continue
            for group, new_category in zip(batch_groups, results):
                if new_category != "기타":
                    for article in group:
                        if article.get("category") == "기타":
                            article["category"] = new_category
                    reclassified_count += 1

    print(f"[relevance_filter] 카테고리 재분류(그룹 단위) 완료 - {len(targets)}개 그룹 중 "
          f"{reclassified_count}개 재분류됨, {len(targets) - reclassified_count}개 '기타' 유지")

    return groups