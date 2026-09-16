"""
gdelt_collector.py
GDELT DOC 2.0 API(gdeltdoc)로 해외 언급 데이터 수집.
tuple(articles, timeline)반환 - timeline은 시계열 수집 제거로 항상 빈 dict.
GDELT 429 대응: 전역 쿨다운, 키워드 단위 외부 재시도, UA 헤더 주입, OR 결합 요청.
"""

import json
import os
import re as _re
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

import requests
import requests.sessions
import requests.utils as _requests_utils

import keyword_source
from gdeltdoc import GdeltDoc, Filters
from gdeltdoc.errors import RateLimitError

# --- 요청 타임아웃 ---
# gdeltdoc은 requests.get()을 timeout 없이 호출함 -> GDELT가 응답을 매달면 무한 대기,
# 파이썬 쪽 deadline 체크가 영영 안 돌아서 job 360분 강제 취소로 이어짐.
# gdeltdoc.api_client 모듈 안의 requests 참조만 교체(네이버 등 다른 모듈엔 영향 없음).
REQUEST_CONNECT_TIMEOUT_SECONDS = 30
REQUEST_READ_TIMEOUT_SECONDS = 120

try:
    import gdeltdoc.api_client as _gdelt_api_client

    class _RequestsWithTimeout:
        """requests 모듈 대리 객체 - get()에만 기본 timeout 주입, 나머지 속성은 원본 그대로."""

        def __getattr__(self, name):
            return getattr(requests, name)

        @staticmethod
        def get(*args, **kwargs):
            kwargs.setdefault("timeout", (REQUEST_CONNECT_TIMEOUT_SECONDS, REQUEST_READ_TIMEOUT_SECONDS))
            return requests.get(*args, **kwargs)

    if getattr(_gdelt_api_client, "requests", None) is requests:
        _gdelt_api_client.requests = _RequestsWithTimeout()
    else:
        print("[gdelt] 🟡 주의 - gdeltdoc 내부 구조가 예상과 달라 요청 timeout 주입 실패 "
              "(gdeltdoc 버전 확인 필요)")
except ImportError:
    print("[gdelt] 🟡 주의 - gdeltdoc.api_client 모듈 없음 - 요청 timeout 주입 실패 (gdeltdoc 버전 확인 필요)")

# UA 미기재 시 429 잦음(gdeltdoc 이슈#22) - requests 기본 헤더 전역 오버라이드.
_GDELT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

if not getattr(requests.utils, "_gdelt_ua_patched", False):
    _original_default_headers = _requests_utils.default_headers

    def _default_headers_with_gdelt_ua():
        headers = _original_default_headers()
        headers["User-Agent"] = _GDELT_USER_AGENT
        return headers

    requests.utils.default_headers = _default_headers_with_gdelt_ua
    requests.sessions.default_headers = _default_headers_with_gdelt_ua
    requests.utils._gdelt_ua_patched = True


# fallback 키워드(영문). 구글 시트 우선.
# keyword_tagger.py의 CATEGORY_KEYWORDS "en" 항목 전체를 카테고리별로 반영.
KEYWORDS_EN = [
    # 질병명
    "African swine fever",
    "ASF",
    "avian influenza",
    "bird flu",
    "bovine spongiform encephalopathy",
    "bovine tuberculosis",
    "brucellosis",
    "BSE",
    "classical swine fever",
    "FMD",
    "foot-and-mouth disease",
    "lumpy skin disease",
    "Newcastle disease",
    "PED",
    "porcine epidemic diarrhea",
    "porcine reproductive and respiratory syndrome",
    # 시장/가격 용어
    "compound feed",
    "corn futures",
    "farm-gate price",
    "feed cost",
    "feed ingredients",
    "feed price",
    "feed self-sufficiency rate",
    "formula feed",
    "global grain market",
    "grain price",
    "grain self-sufficiency rate",
    "livestock exports",
    "livestock imports",
    "livestock supply and demand",
    "mixed feed",
    "SBM",
    "soybean futures",
    "soybean meal",
    "wheat futures",
    # 정부·제도 용어
    "Animal and Plant Quarantine Agency",
    "animal welfare certification",
    "antibiotic-free livestock products",
    "biosecurity",
    "disease control",
    "disease surveillance",
    "livestock manure",
    "livestock movement restriction",
    "livestock traceability system",
    "Ministry of Agriculture, Food and Rural Affairs",
    "Protection Zone",
    "quarantine",
    "standstill order",
    "Surveillance Zone",
    # 축종별 용어
    "beef cattle",
    "broiler",
    "dairy cattle",
    "dairy farming",
    "egg supply",
    "Korean native cattle",
    "laying hens",
    "livestock industry",
    "pig farming",
    "poultry industry",
    "swine industry",
    # 사료업계 특화 용어
    "animal feed",
    "feed manufacturer",
    "feed mill",
    "feed producer",
    "grain elevator",
    "livestock feed",
    "nutrition company",
    "premix",
    "roughage",
    "Total Mixed Ration",
    # 사료첨가제/항생제 규제
    "AMR",
    "antibiotic growth promoter",
    "antibiotic-free feed",
    "antimicrobial resistance",
    "Control of Livestock and Fish Feed Act",
    "feed additive",
    "feed enzyme",
    "feed supplement",
    "methionine",
    "phytase",
    "reduction of antibiotic use",
    "veterinary drugs",
    # 무역/관세 이슈
    "livestock import tariff",
    # 가금 계열화/수직계열화
    "contract farmer",
    "contract grower",
    "guarantee of production cost",
    "poultry vertical integration",
    "vertical integrator",
    # 스마트팜/축산 기술
    "livestock automation",
    "livestock big data",
    "precision livestock farming",
    "smart farming",
    "smart livestock barn",
]

# --- 학습형 스킵 목록: ValueError 누적 시 다음 실행부터 자동 스킵 ---
SKIP_STATE_PATH = "state/gdelt_skip_keywords.json"
SKIP_STATE_FAILURE_THRESHOLD = 2

_value_error_keywords_this_run: list[str] = []


def _load_skip_state(path: str = SKIP_STATE_PATH) -> dict:
    """학습된 스킵 상태 읽기. 없음/파싱 실패/구조 이상이면 빈 dict."""
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            print(f"[gdelt] 🟡 주의 [GD-01] - 학습된 스킵 상태 파일 구조 이상(dict 아님, 타입: "
                  f"{type(state).__name__}) - 빈 상태로 다시 시작: {path}")
            return {}
        return state
    except (OSError, json.JSONDecodeError) as e:
        print(f"[gdelt] 학습된 스킵 상태 파일 읽기 실패(처음 실행이거나 파일 없음 - "
              f"정상, 빈 상태로 시작): {path} - {type(e).__name__} - {e!r}")
        return {}


def _save_skip_state(state: dict, path: str = SKIP_STATE_PATH) -> None:
    """저장 실패해도 예외 안 던짐."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[gdelt] 학습된 스킵 상태 저장 완료 -> {path}")
    except OSError as e:
        print(f"[gdelt] 🟡 주의 [GD-02] - 학습된 스킵 상태 저장 실패(이번 실행 결과엔 영향 없음, "
              f"다음 실행에서 다시 학습 시도됨): {path} - {type(e).__name__} - {e!r}")


def _update_skip_state_after_run() -> None:
    """collect() 끝에서 호출 - 이번 실행 ValueError 발생 키워드들의 fail_count 갱신+저장."""
    if not _value_error_keywords_this_run:
        return

    from collections import Counter
    occurrence_counts = Counter(_value_error_keywords_this_run)

    state = _load_skip_state()
    now_str = datetime.now(timezone.utc).isoformat()
    for keyword, occurrences in occurrence_counts.items():
        entry = state.get(keyword)
        if not isinstance(entry, dict):
            entry = {"fail_count": 0}
        entry["fail_count"] = entry.get("fail_count", 0) + occurrences
        entry["reason"] = "GDELT API가 'phrase too short' 등으로 쿼리 자체를 거부함 (자동 학습됨)"
        entry["last_seen"] = now_str
        state[keyword] = entry
        if occurrences > 1:
            print(f"[gdelt] '{keyword}' - 이번 실행에서 ValueError {occurrences}회 발생 확인 "
                  f"(재시도 라운드 반복 실패) - fail_count {occurrences}만큼 증가")
        if entry["fail_count"] >= SKIP_STATE_FAILURE_THRESHOLD:
            print(f"[gdelt] 🟡 주의 [GD-03] - '{keyword}' - 누적 ValueError {entry['fail_count']}회 확인됨 "
                  f"(임계값 {SKIP_STATE_FAILURE_THRESHOLD}) - 다음 실행부터 자동 스킵 대상으로 등록")

    _save_skip_state(state)


# --- 학습형 크라우딩 목록: 상한 근처였던 키워드는 다음부터 배치 없이 개별 요청 ---
CROWDING_STATE_PATH = "state/gdelt_crowding_keywords.json"
CROWDING_STATE_LEARN_THRESHOLD = 2

_crowded_keywords_this_run: list[str] = []


def _load_crowding_state(path: str = CROWDING_STATE_PATH) -> dict:
    """학습된 크라우딩 상태 읽기. _load_skip_state와 동일한 안전 처리."""
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            print(f"[gdelt] 🟡 주의 - 학습된 크라우딩 상태 파일 구조 이상(dict 아님, 타입: "
                  f"{type(state).__name__}) - 빈 상태로 다시 시작: {path}")
            return {}
        return state
    except (OSError, json.JSONDecodeError) as e:
        print(f"[gdelt] 학습된 크라우딩 상태 파일 읽기 실패(처음 실행이거나 파일 없음 - "
              f"정상, 빈 상태로 시작): {path} - {type(e).__name__} - {e!r}")
        return {}


def _save_crowding_state(state: dict, path: str = CROWDING_STATE_PATH) -> None:
    """저장 실패해도 예외 안 던짐."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[gdelt] 학습된 크라우딩 상태 저장 완료 -> {path}")
    except OSError as e:
        print(f"[gdelt] 🟡 주의 - 학습된 크라우딩 상태 저장 실패(이번 실행 결과엔 영향 없음, "
              f"다음 실행에서 다시 학습 시도됨): {path} - {type(e).__name__} - {e!r}")


def _update_crowding_state_after_run() -> None:
    """collect() 끝에서 호출 - 이번 실행 상한 근처 키워드들의 crowd_count 갱신+저장."""
    if not _crowded_keywords_this_run:
        return

    state = _load_crowding_state()
    now_str = datetime.now(timezone.utc).isoformat()
    for keyword in dict.fromkeys(_crowded_keywords_this_run):
        entry = state.get(keyword)
        if not isinstance(entry, dict):
            entry = {"crowd_count": 0}
        entry["crowd_count"] = entry.get("crowd_count", 0) + 1
        entry["last_seen"] = now_str
        state[keyword] = entry
        if entry["crowd_count"] >= CROWDING_STATE_LEARN_THRESHOLD:
            print(f"[gdelt] '{keyword}' - {entry['crowd_count']}회 연속 상한 근처 확인됨 "
                  f"(임계값 {CROWDING_STATE_LEARN_THRESHOLD}) - 다음 실행부터 배치 없이 바로 개별 요청으로 처리")

    _save_crowding_state(state)

# --- 키워드 오매칭(false positive) 필터 - 실제 확인된 것만 등록 ---
FALSE_POSITIVE_FILTERS = {
    "foot-and-mouth disease": ["hand, foot and mouth", "hand foot and mouth"],
}

# GDELT 제목은 구두점 앞에도 공백이 들어감(예: "hand , foot") - 비교 전 정규화 필요.
_SPACE_BEFORE_PUNCT = _re.compile(r"\s+([,.;:!?])")


def _normalize_spacing(text: str) -> str:
    """구두점 앞 공백 제거(비교용). "hand , foot" -> "hand, foot"."""
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def _is_false_positive(keyword: str, title: str) -> bool:
    """등록된 제외 패턴이 제목에 포함되면 True."""
    patterns = FALSE_POSITIVE_FILTERS.get(keyword, [])
    if not patterns or not title:
        return False
    title_normalized = _normalize_spacing(title.lower())
    return any(_normalize_spacing(p.lower()) in title_normalized for p in patterns)

DAYS_BACK = 7
TIMESPAN = f"{DAYS_BACK}d"

TIMELINE_TIMESPAN = "8d"  # 시계열 전용 기간(현재 시계열 수집 자체는 제거된 상태, 사용 안 함)

MAX_RECORDS = 250  # article_search 1회 호출 최대 반환 건수(API 자체 한계)

TIME_BUDGET_SECONDS = 4 * 60 * 60  # 4시간 - deadline 인자 없으면 자기 시작부터 이 값 적용(하위호환/단독 실행용)

# --- 적응형 배치 수집: 상한 근처면 크라우더 포함 배치 전체를 개별 재요청 ---
BATCH_SIZE = 5  # 잠정값
CROWDING_CAP_TRIGGER_RATIO = 1.0  # 정확히 상한(250)일 때만 크라우딩 검사
CROWDING_SHARE_THRESHOLD = 0.4  # 한 키워드가 이 비율 이상 차지하면 크라우딩 판정

REQUEST_INTERVAL = 15.0  # 키워드/배치 사이 요청 간격(초)

MAX_RETRIES = 4  # 429 재시도: 60->120->240->480초(누적 900초)
BACKOFF_BASE_SECONDS = 60

NETWORK_ERROR_MAX_RETRIES = 2  # 네트워크 에러는 429보다 짧게 재시도
NETWORK_ERROR_WAIT_SECONDS = 10

OUTER_RETRY_PASSES = 2  # 키워드 단위 외부 재시도 라운드 수(총 시도 = 1+이 값)
OUTER_RETRY_WAIT_SECONDS = 90


# --- 시간 예산(deadline) 전역 공유 ---
# collect()가 시작할 때 설정. _call_with_retry 내부 대기(쿨다운/백오프)와 라운드 간 대기도
# 이 마감을 넘기지 않게 함 - 배치 사이에서만 체크하면 429 백오프 한 번에 15분씩 넘어가 버림.
_active_deadline: float | None = None


class BudgetExceededError(Exception):
    """시간 예산 마감에 도달해 요청/대기를 중단함(ValueError 아님 - 학습형 스킵 대상 아님)."""


def _seconds_left() -> float:
    if _active_deadline is None:
        return float("inf")
    return _active_deadline - time.monotonic()


def _sleep_within_budget(seconds: float) -> bool:
    """마감을 넘지 않는 만큼만 대기. 마감까지 전부 못 기다리면 남은 만큼 자고 False 반환."""
    left = _seconds_left()
    if left <= 0:
        return False
    if seconds <= left:
        time.sleep(seconds)
        return True
    time.sleep(left)
    return False


# --- 전역(프로세스 공유) 쿨다운: 429 시 이후 모든 호출이 같은 차단 구간 공유 ---
_cooldown_until = 0.0
_cooldown_lock = threading.Lock()


def _wait_for_cooldown():
    """쿨다운 중이면 대기, 아니면 즉시 반환."""
    with _cooldown_lock:
        remaining = _cooldown_until - time.time()
    if remaining > 0:
        if remaining >= _seconds_left():
            raise BudgetExceededError(
                f"전역 쿨다운 {remaining:.0f}초가 시간 예산 마감({max(_seconds_left(), 0):.0f}초 남음)을 넘김")
        print(f"[gdelt] 전역 쿨다운 중 - {remaining:.0f}초 남음, 대기")
        time.sleep(remaining)


def _trigger_cooldown(seconds: float):
    """전역 쿨다운 설정(연장만 함, 단축 안 됨)."""
    global _cooldown_until
    with _cooldown_lock:
        new_until = time.time() + seconds
        if new_until > _cooldown_until:
            _cooldown_until = new_until


def _parse_retry_after(response) -> float | None:
    """RateLimitError 응답의 Retry-After 헤더 파싱(초 또는 HTTP-date). 없으면 None."""
    if response is None:
        return None
    value = response.headers.get("Retry-After")
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        pass

    try:
        from email.utils import parsedate_to_datetime
        retry_dt = parsedate_to_datetime(value)
        if retry_dt.tzinfo is None:
            retry_dt = retry_dt.replace(tzinfo=timezone.utc)
        seconds = (retry_dt - datetime.now(timezone.utc)).total_seconds()
        return max(seconds, 0.0)
    except (TypeError, ValueError):
        return None


def _call_with_retry(func, *args, label: str = "", **kwargs):
    """
    RateLimitError(길게)와 네트워크 에러(짧게)를 다른 정책으로 재시도.
    429는 전역 쿨다운을 걸어 이후 모든 호출이 공유. Retry-After 헤더 있으면 그 값 우선, 없으면 지수 백오프.
    """
    rate_limit_attempt = 0
    network_attempt = 0

    while True:
        if _seconds_left() <= 0:
            raise BudgetExceededError(f"{label} - 시간 예산 마감 도달, 요청 생략")
        _wait_for_cooldown()
        try:
            return func(*args, **kwargs)
        except RateLimitError as e:
            if rate_limit_attempt >= MAX_RETRIES:
                raise
            server_wait = _parse_retry_after(getattr(e, "response", None))
            wait = server_wait if server_wait is not None else BACKOFF_BASE_SECONDS * (2 ** rate_limit_attempt)
            if wait >= _seconds_left():
                raise BudgetExceededError(
                    f"{label} - 429 백오프 {wait:.0f}초가 시간 예산 마감"
                    f"({max(_seconds_left(), 0):.0f}초 남음)을 넘김 - 재시도 중단") from e
            rate_limit_attempt += 1
            now_str = datetime.now(timezone.utc).isoformat(timespec="seconds")
            print(f"[gdelt] {now_str} - {label} - 429 rate limit - {wait:.0f}초 전역 쿨다운 설정 "
                  f"({rate_limit_attempt}/{MAX_RETRIES})")
            _trigger_cooldown(wait)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            # Timeout = ConnectTimeout + ReadTimeout(응답 매달림)
            if network_attempt >= NETWORK_ERROR_MAX_RETRIES:
                raise
            network_attempt += 1
            print(f"[gdelt] {label} - 접속/응답 실패({type(e).__name__}) - "
                  f"{NETWORK_ERROR_WAIT_SECONDS}초 대기 후 재시도 "
                  f"({network_attempt}/{NETWORK_ERROR_MAX_RETRIES})")
            if not _sleep_within_budget(NETWORK_ERROR_WAIT_SECONDS):
                raise BudgetExceededError(f"{label} - 네트워크 재시도 대기 중 시간 예산 마감 도달") from e


def _parse_seendate(raw: str) -> datetime:
    """seendate -> datetime. "YYYYMMDDHHMMSS"/ISO 8601 순으로 시도."""
    raw = raw.strip()

    for fmt in ("%Y%m%d%H%M%S", "%Y%m%dT%H%M%SZ"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        pass

    raise ValueError(f"seendate 파싱 실패, 형식 확인 필요: {raw!r}")


def _is_recent(dt: datetime, days: int) -> bool:
    cutoff = datetime.now(dt.tzinfo) - timedelta(days=days)
    return dt >= cutoff


def _extract_domain(url: str) -> str:
    """URL 도메인만 추출(press 필드용)."""
    if not url:
        return ""
    domain = urlparse(url).netloc
    return domain.replace("www.", "")


def _collect_articles_for_keyword(gd: "GdeltDoc", keyword: str) -> tuple[bool, list[dict], str | None]:
    """
    키워드 1개 article_search. 반환: (성공 여부, 기사 리스트, 실패 사유).
    정식 실행 경로에서는 _collect_articles_for_keywords(OR 결합)로 대체됨 -
    이 함수는 진단/개별 폴백용으로 남겨둠.
    """
    f = Filters(keyword=keyword, timespan=TIMESPAN, num_records=MAX_RECORDS)
    keyword_articles = []
    false_positive_count = 0

    try:
        articles_df = _call_with_retry(gd.article_search, f, label=f"{keyword} / article_search")

        if articles_df is not None and not articles_df.empty:
            for _, row in articles_df.iterrows():
                if _is_false_positive(keyword, str(row["title"])):
                    false_positive_count += 1
                    continue

                try:
                    published_at = _parse_seendate(str(row["seendate"]))
                except ValueError as e:
                    print(f"[gdelt] 🟡 주의 [GD-04] - '{keyword}' 기사 스킵 - {type(e).__name__} - {e!r}")
                    continue

                if not _is_recent(published_at, DAYS_BACK):
                    continue

                keyword_articles.append({
                    "source": "GDELT",
                    "title": row["title"],
                    "url": row["url"],
                    "published_at": published_at.isoformat(),
                    "category": None,
                    "body": None,
                    "press": _extract_domain(row["url"]),
                    "language": row.get("language", ""),
                    "sourcecountry": row.get("sourcecountry", ""),
                })

        fp_note = f" (오매칭 필터로 {false_positive_count}건 제외)" if false_positive_count else ""
        print(f"[gdelt] '{keyword}' article_search -> 최근 {DAYS_BACK}일 이내 "
              f"{len(keyword_articles)}건 수집 완료{fp_note}")
        return True, keyword_articles, None

    except BudgetExceededError as e:
        print(f"[gdelt] 🟡 주의 - '{keyword}' 시간 예산 마감으로 중단: {e}")
        return False, [], f"{type(e).__name__}: {e}"

    except ValueError as e:
        # 쿼리 자체 거부(확정적 실패) - 학습형 스킵 목록 대상으로 기록.
        print(f"[gdelt] 🟡 주의 [GD-05] - '{keyword}' article_search 실패(쿼리 자체 거부로 추정 - "
              f"{type(e).__name__}: {e})")
        _value_error_keywords_this_run.append(keyword)
        return False, [], f"{type(e).__name__}: {e}"

    except Exception as e:
        print(f"[gdelt] 🟡 주의 [GD-06] - '{keyword}' article_search 실패: {type(e).__name__} - {e!r}")
        return False, [], f"{type(e).__name__}: {e}"


def _collect_articles_for_keywords(gd: "GdeltDoc", keywords: list[str]) -> tuple[bool, list[dict]]:
    """
    여러 키워드를 OR 쿼리 하나로 묶어 article_search 1회 호출(요청 횟수 절약).
    MAX_RECORDS 상한이 "키워드 조합 전체"에 적용됨에 주의(개별 요청과 다름).
    오매칭 필터는 keywords 전체의 패턴을 모아 확인(OR 매칭이라 어느 키워드가
    트리거했는지 API가 알려주지 않음).
    """
    label = " OR ".join(keywords)
    f = Filters(keyword=keywords, timespan=TIMESPAN, num_records=MAX_RECORDS)
    combined_articles = []
    false_positive_count = 0

    try:
        articles_df = _call_with_retry(gd.article_search, f, label=f"{label} / article_search")

        if articles_df is not None and not articles_df.empty:
            for _, row in articles_df.iterrows():
                title = str(row["title"])
                if any(_is_false_positive(kw, title) for kw in keywords):
                    false_positive_count += 1
                    continue

                try:
                    published_at = _parse_seendate(str(row["seendate"]))
                except ValueError as e:
                    print(f"[gdelt] 🟡 주의 [GD-07] - '{label}' 기사 스킵 - {type(e).__name__} - {e!r}")
                    continue

                if not _is_recent(published_at, DAYS_BACK):
                    continue

                combined_articles.append({
                    "source": "GDELT",
                    "title": row["title"],
                    "url": row["url"],
                    "published_at": published_at.isoformat(),
                    "category": None,
                    "body": None,
                    "press": _extract_domain(row["url"]),
                    "language": row.get("language", ""),
                    "sourcecountry": row.get("sourcecountry", ""),
                })

        fp_note = f" (오매칭 필터로 {false_positive_count}건 제외)" if false_positive_count else ""
        print(f"[gdelt] '{label}' article_search -> 최근 {DAYS_BACK}일 이내 "
              f"{len(combined_articles)}건 수집 완료{fp_note}")

        # 키워드별 매칭 현황(제목 기준 근사치, 본문 매칭은 집계 안 됨)
        print(f"[gdelt] 키워드별 매칭 현황(제목 기준 근사치 - 본문에만 있는 매칭은 " 
              f"집계 안 됨, 하나의 기사가 여러 키워드에 동시 매칭될 수 있어 합계가 "
              f"전체 건수와 다를 수 있음):")
        for kw in keywords:
            kw_lower = kw.lower()
            count = sum(1 for a in combined_articles if kw_lower in a["title"].lower())
            print(f"  - '{kw}': {count}건")
        return True, combined_articles

    except BudgetExceededError as e:
        print(f"[gdelt] 🟡 주의 - '{label}' 시간 예산 마감으로 중단: {e}")
        return False, []

    except ValueError as e:
        # 쿼리 자체 거부는 재시도해도 100% 같은 이유로 실패(OR 결합에 문제
        # 키워드가 섞여있는 한) - 재시도 대신 즉시 키워드별 개별 요청으로 전환.
        print(f"[gdelt] 🟡 주의 [GD-08] - '{label}' article_search 실패(쿼리 자체 거부로 추정 - "
              f"{type(e).__name__}: {e}) - 재시도 대신 키워드별 개별 요청으로 즉시 전환")
        return _collect_articles_individually(gd, keywords)

    except Exception as e:
        print(f"[gdelt] 🟡 주의 [GD-09] - '{label}' article_search 실패: {type(e).__name__} - {e!r}")
        return False, []


def _collect_articles_individually(gd: "GdeltDoc", keywords: list[str]) -> tuple[bool, list[dict]]:
    """
    결합 쿼리가 확정적 실패(재시도로 안 풀림)로 끝났을 때의 격리 폴백.
    키워드를 하나씩 따로 요청해 문제 키워드만 실패로 남기고 나머지는 살림.
    """
    all_articles = []
    any_success = False
    for keyword in keywords:
        if _seconds_left() <= 0:
            break
        success, keyword_articles, _reason = _collect_articles_for_keyword(gd, keyword)
        if success:
            any_success = True
            all_articles.extend(keyword_articles)
        _sleep_within_budget(REQUEST_INTERVAL)
    return any_success, all_articles


def _detect_crowded_keywords(articles: list[dict], keywords: list[str]) -> list[str]:
    """
    배치(OR 결합) 결과에서 특정 키워드가 과도하게 차지해 다른 키워드가
    상한에 밀렸을 가능성을 감지. 반환: 크라우딩 원인 키워드 목록(비어있지
    않으면 호출부가 크라우더 포함 배치 전체를 개별 재요청 대상으로 삼음).
    """
    total = len(articles)
    if total < MAX_RECORDS * CROWDING_CAP_TRIGGER_RATIO:
        return []

    crowders = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        count = sum(1 for a in articles if keyword_lower in a["title"].lower())
        if count / total >= CROWDING_SHARE_THRESHOLD:
            crowders.append(keyword)
    return crowders


def _handle_batch_crowding(batch: list[str], batch_articles: list[dict]) -> list[str]:
    """
    배치 결과의 크라우딩 여부를 판단, 트리거 여부와 무관하게 항상 판정
    결과를 로그로 남긴다. 크라우더 자신도 나머지와 함께 배치 전체를 개별
    재요청 대상으로 편입(250건 상한을 다른 키워드와 나눠 쓰다 잘렸을 수
    있으므로). 반환: 개별 재요청 대상 키워드 리스트(없으면 빈 리스트).
    """
    crowders = _detect_crowded_keywords(batch_articles, batch)
    if crowders:
        print(f"[gdelt] 🟡 주의 [GD-10] - 배치 {batch} 내 크라우딩 감지({crowders}가 결과의 "
              f"{int(CROWDING_SHARE_THRESHOLD * 100)}% 이상 차지 추정) - "
              f"크라우더 포함 배치 전체를 개별 재요청 대상으로 편입")
        return list(batch)
    if len(batch_articles) >= MAX_RECORDS * CROWDING_CAP_TRIGGER_RATIO:
        print(f"[gdelt] 🟡 주의 [GD-11] - 배치 {batch} 결과가 상한 근처까지 참({len(batch_articles)}건) - "
              f"골고루 밀렸을 위험 있어 배치 전체를 개별 재요청 대상으로 편입")
        return list(batch)
    print(f"[gdelt] 배치 {batch} - 크라우딩/상한 근접 없음, 개별 재요청 없이 결과 확정"
          f"({len(batch_articles)}건)")
    return []


def _collect_timeline_for_keyword(gd: "GdeltDoc", keyword: str) -> dict | None:
    """키워드 1개 timelinevol/timelinevolraw 수집. collect()에서 더 이상 호출 안 함(시계열 수집 제거됨)."""
    f = Filters(keyword=keyword, timespan=TIMELINE_TIMESPAN, num_records=MAX_RECORDS)
    try:
        vol_df = _call_with_retry(gd.timeline_search, "timelinevol", f,
                                   label=f"{keyword} / timelinevol")
        time.sleep(REQUEST_INTERVAL)
        volraw_df = _call_with_retry(gd.timeline_search, "timelinevolraw", f,
                                      label=f"{keyword} / timelinevolraw")
        print(f"[gdelt] '{keyword}' 시계열 수집 완료")
        return {
            "vol": vol_df.to_dict("records") if vol_df is not None else [],
            "volraw": volraw_df.to_dict("records") if volraw_df is not None else [],
        }
    except Exception as e:
        print(f"[gdelt] '{keyword}' 시계열 수집 실패: {type(e).__name__} - {e!r}")
        return None


def collect(keywords: list[str] | None = None, deadline: float | None = None) -> tuple[list[dict], dict]:
    """
    KEYWORDS_EN(또는 인자로 넘긴 keywords)을 대상으로 GDELT에서 기사 수집. 진입점.
    timeline은 시계열 수집 제거로 항상 빈 dict.

    deadline: time.monotonic() 기준 절대 마감(파이프라인 시작 기준 체크포인트).
    None이면 지금부터 TIME_BUDGET_SECONDS 후로 자체 계산(단독 실행용).

    적응형 배치 수집: ① BATCH_SIZE씩 묶어 OR 결합 요청 ② 크라우딩 감지되면
    크라우더 포함 배치 전체를 개별 재요청 ③ 배치 요청 자체 실패(429 등)는
    같은 배치로 외부 재시도, 최종 실패해야 개별 전환. deadline 초과 시 남은
    키워드는 이번 실행 건너뜀.
    """
    global _active_deadline
    if deadline is None:
        deadline = time.monotonic() + TIME_BUDGET_SECONDS
    _active_deadline = deadline

    gd = GdeltDoc()
    target_keywords = keywords if keywords is not None else keyword_source.get_keywords("en", KEYWORDS_EN)

    _value_error_keywords_this_run.clear()
    _crowded_keywords_this_run.clear()
    skip_state = _load_skip_state()
    crowding_state = _load_crowding_state()

    active_keywords = []
    known_crowders = []
    for keyword in target_keywords:
        learned_entry = skip_state.get(keyword)
        if isinstance(learned_entry, dict) and learned_entry.get("fail_count", 0) >= SKIP_STATE_FAILURE_THRESHOLD:
            print(f"[gdelt] '{keyword}' 스킵 - 학습형 스킵 목록 등재됨 "
                  f"({learned_entry.get('fail_count')}회 연속 ValueError 확인, "
                  f"{SKIP_STATE_PATH} 참고)")
            continue
        crowding_entry = crowding_state.get(keyword)
        if isinstance(crowding_entry, dict) and crowding_entry.get("crowd_count", 0) >= CROWDING_STATE_LEARN_THRESHOLD:
            print(f"[gdelt] '{keyword}' - 학습된 크라우딩 키워드({crowding_entry.get('crowd_count')}회 "
                  f"연속 상한 근처 확인, {CROWDING_STATE_PATH} 참고) - 배치 없이 바로 개별 요청으로 처리")
            known_crowders.append(keyword)
            continue
        active_keywords.append(keyword)

    all_articles = []
    timeline_by_keyword = {}

    if not active_keywords and not known_crowders:
        return all_articles, timeline_by_keyword

    def _over_budget() -> bool:
        return time.monotonic() >= deadline

    budget_exceeded = False
    skipped_due_to_budget: list[str] = []

    # --- 1단계: 배치 단위 수집 + 크라우딩 감지 ---
    batches = [active_keywords[i:i + BATCH_SIZE] for i in range(0, len(active_keywords), BATCH_SIZE)]
    pending_individual: list[str] = list(known_crowders)
    pending_batches: list[list[str]] = []

    for batch_idx, batch in enumerate(batches):
        if _over_budget():
            remaining = [kw for b in batches[batch_idx:] for kw in b]
            skipped_due_to_budget.extend(remaining)
            budget_exceeded = True
            print(f"[gdelt] 🟡 주의 - 시간 예산(파이프라인 기준 마감 도달) 소진 - "
                  f"남은 배치 {len(batches) - batch_idx}개(키워드 {len(remaining)}개)는 "
                  f"이번 실행에서 건너뜀. 다음 실행에서 다시 시도됨.")
            break

        if len(batch) == 1:
            pending_individual.append(batch[0])
            continue

        success, batch_articles = _collect_articles_for_keywords(gd, batch)
        if not success:
            # 실패했다고 개별 전환(요청 5배)하면 429를 악화시키므로 같은 배치로 재시도
            print(f"[gdelt] 배치 {batch} 요청 실패 - 같은 배치로 재시도 예정 "
                  f"(개별 전환 아님)")
            pending_batches.append(batch)
        else:
            all_articles.extend(batch_articles)
            pending_individual.extend(_handle_batch_crowding(batch, batch_articles))
        _sleep_within_budget(REQUEST_INTERVAL)

    # --- 1-보조단계: 배치 요청 자체가 실패한 것들을 배치 그대로 재시도 ---
    if budget_exceeded:
        skipped_due_to_budget.extend(kw for b in pending_batches for kw in b)
        pending_batches = []

    batch_round = pending_batches
    for round_num in range(1, OUTER_RETRY_PASSES + 1):
        if not batch_round:
            break
        if _over_budget():
            remaining = [kw for b in batch_round for kw in b]
            skipped_due_to_budget.extend(remaining)
            budget_exceeded = True
            print(f"[gdelt] 🟡 주의 - 시간 예산 소진 - 배치 재시도 중단"
                  f"(키워드 {len(remaining)}개는 이번 실행에서 건너뜀)")
            batch_round = []
            break
        print(f"[gdelt] --- 배치 재시도 라운드 {round_num}/{OUTER_RETRY_PASSES} - "
              f"이전 라운드 실패 배치 {len(batch_round)}개 ---")
        print(f"[gdelt] 라운드 간 안전 대기 {OUTER_RETRY_WAIT_SECONDS}초")
        _sleep_within_budget(OUTER_RETRY_WAIT_SECONDS)

        still_failed_batches = []
        for batch_idx2, batch in enumerate(batch_round):
            if _over_budget():
                remaining = [kw for b in batch_round[batch_idx2:] for kw in b]
                skipped_due_to_budget.extend(remaining)
                budget_exceeded = True
                print(f"[gdelt] 🟡 주의 - 시간 예산 소진 - 배치 재시도 라운드 도중 중단"
                      f"(키워드 {len(remaining)}개는 이번 실행에서 건너뜀)")
                still_failed_batches = []
                break
            success, batch_articles = _collect_articles_for_keywords(gd, batch)
            if success:
                all_articles.extend(batch_articles)
                pending_individual.extend(_handle_batch_crowding(batch, batch_articles))
            else:
                still_failed_batches.append(batch)
            _sleep_within_budget(REQUEST_INTERVAL)
        batch_round = still_failed_batches

    if batch_round:
        # 배치로 더 시도할 수단이 없어 키워드 단위로 쪼갬
        print(f"[gdelt] 🟡 주의 [GD-12] - 배치 재시도 {OUTER_RETRY_PASSES}회 소진 - 개별 요청 전환: {batch_round}")
        for batch in batch_round:
            pending_individual.extend(batch)

    # --- 2단계: 개별 보충 요청 - 실패한 것만 외부 재시도 라운드 ---
    if budget_exceeded:
        skipped_due_to_budget.extend(pending_individual)
        pending_individual = []

    round_keywords = list(dict.fromkeys(pending_individual))
    failed_keywords: list[str] = []
    failure_reasons: dict[str, str] = {}

    for round_num in range(OUTER_RETRY_PASSES + 1):
        if round_num > 0:
            if not failed_keywords:
                break
            print(f"[gdelt] --- 기사 수집 외부 재시도 라운드 {round_num}/{OUTER_RETRY_PASSES} - "
                  f"이전 라운드 실패 키워드 {len(failed_keywords)}개: {failed_keywords} ---")
            if _seconds_left() <= 0:
                skipped_due_to_budget.extend(failed_keywords)
                print(f"[gdelt] 🟡 주의 - 시간 예산 소진 - 개별 재시도 라운드 생략"
                      f"(키워드 {len(failed_keywords)}개는 이번 실행에서 건너뜀)")
                failed_keywords = []
                break
            print(f"[gdelt] 라운드 간 안전 대기 {OUTER_RETRY_WAIT_SECONDS}초")
            _sleep_within_budget(OUTER_RETRY_WAIT_SECONDS)
            round_keywords = failed_keywords

        if not round_keywords:
            break

        failed_keywords = []

        for kw_idx, keyword in enumerate(round_keywords):
            if _over_budget():
                remaining = round_keywords[kw_idx:]
                skipped_due_to_budget.extend(remaining)
                print(f"[gdelt] 🟡 주의 - 시간 예산 소진 - 개별 요청 중단"
                      f"(키워드 {len(remaining)}개는 이번 실행에서 건너뜀)")
                failed_keywords = []
                round_keywords = []
                break

            success, keyword_articles, reason = _collect_articles_for_keyword(gd, keyword)
            if success:
                all_articles.extend(keyword_articles)
                failure_reasons.pop(keyword, None)
                if len(keyword_articles) >= MAX_RECORDS * CROWDING_CAP_TRIGGER_RATIO:
                    _crowded_keywords_this_run.append(keyword)
            else:
                failed_keywords.append(keyword)
                failure_reasons[keyword] = reason or "사유 불명"
            _sleep_within_budget(REQUEST_INTERVAL)

        if not round_keywords:
            break

    if failed_keywords:
        detail = ", ".join(f"{kw} ({failure_reasons.get(kw, '사유 불명')})" for kw in failed_keywords)
        print(f"[gdelt] 🟡 주의 [GD-13] - 최종 실패 키워드 (총 {OUTER_RETRY_PASSES + 1}회 시도 후에도 실패, "
              f"기사 0건으로 처리됨): {detail}")

    if skipped_due_to_budget:
        unique_skipped = list(dict.fromkeys(skipped_due_to_budget))
        print(f"[gdelt] 🟡 주의 - 시간 예산 초과로 이번 실행에서 아예 시도 못 한 키워드 "
              f"{len(unique_skipped)}개(실패로 기록되지 않음, 다음 실행에서 처음부터 재시도됨): "
              f"{unique_skipped}")

    _update_skip_state_after_run()
    _update_crowding_state_after_run()

    _active_deadline = None
    return all_articles, timeline_by_keyword
