"""
naver_collector.py
네이버 뉴스 검색 API 수집 모듈(수집 레이어).
"""

import re
import html
import os
import time
import requests
from urllib.parse import urlparse
from datetime import datetime, timedelta
from dotenv import load_dotenv

import keyword_source

load_dotenv()

# NAVER Developers Center -> NAVER API HUB(NCP API Gateway) 이관(2026-07). HUB 키는 옛 주소에서 401.
NAVER_API_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
DAYS_BACK = 7  # 최근 N일 이내 기사만

# fallback 키워드. 구글 시트(KEYWORD_SHEET_CSV_URL) 우선, 실패 시 이 리스트 사용.
# 시트 변경 시 수동 동기화 필요(자동 아님).
# 시트 실패 시 fallback - 시트 정상 시 마지막 로드 목록(국문 66개)을 그대로 옮김.
KEYWORDS = [
    "광우병",
    "구제역",
    "뉴캣슬병",
    "돼지생식기호흡기증후군",
    "돼지열병",
    "돼지유행성설사병",
    "럼피스킨병",
    "브루셀라병",
    "아프리카돼지열병",
    "우결핵",
    "조류독감",
    "곡물가격",
    "곡물자급률",
    "국제곡물가",
    "대두박",
    "배합사료",
    "사료 원료",
    "사료가격",
    "사료값",
    "사료자급률",
    "산지가격",
    "축산물 수출입",
    "가축 이동제한",
    "가축분뇨",
    "가축전염병예방법",
    "농림축산검역본부",
    "동물복지 인증",
    "무항생제 축산물",
    "방역대",
    "축산물 이력제",
    "가금 전반",
    "계란 수급",
    "낙농",
    "산란계",
    "양돈",
    "육계",
    "육우",
    "젖소",
    "축산업 전반",
    "곡물엘리베이터",
    "배합사료업체",
    "사료공장",
    "영양첨가업체",
    "조사료",
    "프리믹스",
    "동물용의약품",
    "무항생제 사료",
    "사료관리법",
    "사료보충제",
    "사료첨가제",
    "사료효소",
    "피타아제",
    "항생제 사용 저감",
    "항생제 성장촉진제",
    "축산물 할당관세",
    "계약사육",
    "계열주체",
    "계열화 기업",
    "계열화사업법",
    "생산비 보장",
    "축산업 계열화",
    "스마트축사",
    "자동급여시스템",
    "정밀축산",
    "축산 빅데이터",
    "축산 자동화",
]


def _is_recent(published_at: str, days: int) -> bool:
    """published_at이 오늘 기준 최근 N일 이내인지 확인."""
    pub_dt = datetime.fromisoformat(published_at)
    cutoff = datetime.now(pub_dt.tzinfo) - timedelta(days=days)
    return pub_dt >= cutoff


MAX_START = 1000  # 네이버 API가 허용하는 start 최댓값


def _phrase_present(article: dict, keyword: str) -> bool:
    """네이버 API는 공백 구분 검색어를 단어별 AND로 처리해 오탐 가능 - 여러 단어 키워드는 문구 인접 여부 재확인. 단어 1개는 항상 통과."""
    if " " not in keyword.strip():
        return True
    combined = f"{article.get('title', '')} {article.get('description', '')}"
    combined_normalized = " ".join(combined.split())
    keyword_normalized = " ".join(keyword.split())
    return keyword_normalized in combined_normalized


def collect() -> list[dict]:
    """KEYWORDS(또는 시트 활성 키워드)를 돌며 네이버 뉴스 전체 수집. 진입점."""
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("[naver] 🔴 조치필요 [NV-01] - NAVER_CLIENT_ID/NAVER_CLIENT_SECRET 없음 - 네이버 수집 불가")
        raise RuntimeError("NAVER_CLIENT_ID/NAVER_CLIENT_SECRET 없음")

    target_keywords = keyword_source.get_keywords("ko", KEYWORDS)

    all_results = []
    with requests.Session() as session:
        for keyword in target_keywords:
            keyword_results = []
            start = 1
            try:
                while start <= MAX_START:
                    page = search_naver_news(keyword, client_id, client_secret, start=start, session=session)

                    if not page:
                        break

                    recent_in_page = [r for r in page if _is_recent(r["published_at"], DAYS_BACK)]
                    phrase_ok = [r for r in recent_in_page if _phrase_present(r, keyword)]
                    filtered_out = len(recent_in_page) - len(phrase_ok)
                    if filtered_out:
                        print(f"[naver] '{keyword}' - 문구 인접성 필터로 {filtered_out}건 제외"
                              f"(AND 매칭 오탐 방지)")
                    keyword_results.extend(phrase_ok)

                    oldest_in_page = page[-1]
                    if not _is_recent(oldest_in_page["published_at"], DAYS_BACK):
                        break

                    if len(page) < 100:
                        break

                    start += 100
                    time.sleep(0.2)
            except requests.exceptions.RequestException as e:
                print(f"[naver] 🔴 조치필요 [NV-02] - '{keyword}' 수집 중 오류 발생(지금까지 모은 "
                      f"{len(keyword_results)}건은 보존하고 다음 키워드로 진행): {type(e).__name__} - {e!r}")

            all_results.extend(keyword_results)
            print(f"[naver] '{keyword}' -> 최근 {DAYS_BACK}일 이내 {len(keyword_results)}건 "
                  f"({start if start <= MAX_START else MAX_START}건째까지 확인)")

            time.sleep(0.2)

    return all_results


def search_naver_news(keyword: str, client_id: str, client_secret: str, start: int = 1,
                       session: requests.Session | None = None) -> list[dict]:
    """네이버 뉴스 검색 API 호출 -> 공통 스키마로 변환해 반환."""
    headers = {
        "X-NCP-APIGW-API-KEY-ID": client_id,
        "X-NCP-APIGW-API-KEY": client_secret,
    }

    params = {
        "query": keyword,
        "display": 100,
        "start": start,
        "sort": "date",
    }

    requester = session if session is not None else requests
    response = requester.get(NAVER_API_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    results = []
    for item in data.get("items", []):
        results.append({
            "source": "네이버",
            "title": _strip_html_tags(item["title"]),
            "url": item["originallink"] or item["link"],
            "published_at": _parse_pub_date(item["pubDate"]),
            "category": None,
            "body": None,
            "description": _strip_html_tags(item["description"]),
            "press": _extract_press(item["originallink"]),
        })
    return results


def _parse_pub_date(pub_date_str: str) -> str:
    """RFC 822 형식("Mon, 13 Jul 2026 09:00:00 +0900") -> ISO 8601 변환."""
    dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
    return dt.isoformat()


def _extract_press(originallink: str) -> str:
    """originallink 도메인을 언론사 식별자로 사용. 서브도메인 통합은 안 함."""
    if not originallink:
        return ""
    domain = urlparse(originallink).netloc
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _strip_html_tags(text: str) -> str:
    """title/description의 <b> 강조 태그 제거 + HTML 엔티티 복원."""
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return text