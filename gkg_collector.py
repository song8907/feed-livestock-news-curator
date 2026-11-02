"""
gkg_collector.py - GDELT GKG 2.1 원본 파일(data.gdeltproject.org, 15분 단위 zip)에서 제목 키워드 매칭으로 기사 수집.

DOC API(api.gdeltproject.org)가 429/{}로 막힐 때 gdelt_collector가 대체 경로로 호출함.
- 수집 구간의 15분 슬롯 파일을 전부 받아 <PAGE_TITLE> 기준으로 키워드 매칭(본문 매칭은 불가)
- 영문 GKG만 사용(번역 GKG는 제목이 원문 언어라 영문 키워드에 거의 안 걸림)
- 반환 스키마는 gdelt_collector와 동일(source="GDELT")
"""

import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Callable
from urllib.parse import urlparse

import requests

BASE_URL = "http://data.gdeltproject.org/gdeltv2"
SLOT = timedelta(minutes=15)
MAX_WORKERS = 4
DOWNLOAD_RETRIES = 2

# GKG 2.1 컬럼: 1 DATE(YYYYMMDDHHMMSS), 3 SourceCommonName, 4 DocumentIdentifier(URL), 26 Extras
COL_DATE, COL_SOURCE, COL_URL, COL_EXTRAS = 1, 3, 4, 26
_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.S)


def _keyword_pattern(keyword: str) -> re.Pattern:
    """단어 경계 매칭. 'AI'·'LLM'처럼 짧은 대문자 약어는 대소문자 구분(일반 단어 속 오매칭 방지)."""
    escaped = r"\s+".join(re.escape(part) for part in keyword.split())
    flags = 0 if (keyword.isupper() and len(keyword) <= 4) else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags)


def _slot_times(window_start: datetime, window_end: datetime) -> list[datetime]:
    """구간 [start, end)에 걸치는 15분 슬롯 파일 타임스탬프(GKG 행의 DATE = 파일 타임스탬프)."""
    t = window_start.replace(minute=(window_start.minute // 15) * 15, second=0, microsecond=0)
    if t < window_start:
        t += SLOT
    slots = []
    while t < window_end:
        slots.append(t)
        t += SLOT
    return slots


def _download_slot(ts: datetime) -> tuple[datetime, bytes | None, str | None]:
    url = f"{BASE_URL}/{ts.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"
    last_error = None
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                return ts, resp.content, None
            if resp.status_code == 404:
                return ts, None, "404(슬롯 파일 없음)"
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = f"{type(e).__name__}"
        time.sleep(3 * (attempt + 1))
    return ts, None, last_error


def _iter_rows(content: bytes):
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    for line in text.splitlines():
        cols = line.split("\t")
        if len(cols) <= COL_EXTRAS:
            continue
        m = _TITLE_RE.search(cols[COL_EXTRAS])
        if not m:
            continue
        title = unescape(m.group(1)).strip()
        if title:
            yield cols[COL_DATE], cols[COL_SOURCE], cols[COL_URL], title


def collect(keywords: list[str], window_start: datetime, window_end: datetime,
            is_false_positive: Callable[[str, str], bool] | None = None) -> list[dict]:
    """구간 내 GKG 슬롯 전체에서 제목에 키워드가 들어간 기사 수집(URL 기준 중복 제거)."""
    if not keywords:
        return []

    patterns = [(kw, _keyword_pattern(kw)) for kw in keywords]
    slots = _slot_times(window_start, window_end)
    t0 = time.monotonic()
    print(f"[gkg] 대체 수집 시작 - 키워드 {len(keywords)}개, 슬롯 {len(slots)}개 "
          f"({window_start.isoformat()} ~ {window_end.isoformat()})")

    articles: dict[str, dict] = {}
    per_keyword = {kw: 0 for kw in keywords}
    failed_slots: list[str] = []
    total_rows = 0
    total_bytes = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for ts, content, error in pool.map(_download_slot, slots):
            if content is None:
                failed_slots.append(f"{ts.strftime('%m%d %H%M')}({error})")
                continue
            total_bytes += len(content)
            try:
                rows = list(_iter_rows(content))
            except (zipfile.BadZipFile, IndexError) as e:
                failed_slots.append(f"{ts.strftime('%m%d %H%M')}({type(e).__name__})")
                continue
            total_rows += len(rows)

            for raw_date, source, url, title in rows:
                if url in articles:
                    continue
                try:
                    published_at = datetime.strptime(raw_date, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                if not (window_start <= published_at < window_end):
                    continue
                matched = [kw for kw, p in patterns if p.search(title)
                           and not (is_false_positive and is_false_positive(kw, title))]
                if not matched:
                    continue
                for kw in matched:
                    per_keyword[kw] += 1
                articles[url] = {
                    "source": "GDELT",
                    "title": title,
                    "url": url,
                    "published_at": published_at.isoformat(),
                    "category": None,
                    "body": None,
                    "press": source or urlparse(url).netloc.replace("www.", ""),
                    "language": "English",
                    "sourcecountry": "",
                }

    elapsed = time.monotonic() - t0
    print(f"[gkg] 슬롯 {len(slots) - len(failed_slots)}/{len(slots)}개 처리, {total_bytes / 1e6:.0f}MB, "
          f"제목 {total_rows}행 스캔, {elapsed:.0f}초 -> 매칭 {len(articles)}건")
    print("[gkg] 키워드별 매칭: " + ", ".join(f"'{kw}' {n}" for kw, n in per_keyword.items()))
    if failed_slots:
        level = "🔴 조치필요 [GK-02]" if len(failed_slots) > len(slots) // 2 else "🟡 주의 [GK-01]"
        print(f"[gkg] {level} - 받지 못한 슬롯 {len(failed_slots)}개: {failed_slots[:10]}")
    return list(articles.values())
