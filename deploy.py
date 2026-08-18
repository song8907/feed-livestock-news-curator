"""
deploy.py - 배포 레이어([12]단계).
Gmail SMTP로 주간 큐레이션 결과를 HTML 이메일(+PDF 첨부) 발송.
인증정보는 GitHub Secrets(SMTP_USER/SMTP_APP_PASSWORD/EMAIL_RECIPIENTS)에서 읽음.
발송 실패해도 예외 안 던지고 로그만 남기고 조용히 실패(파이프라인 안 죽음).
"""

import html
import os
import smtplib
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# 이메일 디자인 토큰. 국내=파랑, 해외=보라(증감표의 초록=증가와 겹치지 않게).
ACCENT_DOMESTIC = "#1a73e8"
ACCENT_DOMESTIC_TINT = "#e8f0fe"
ACCENT_INTL = "#7c3aed"
ACCENT_INTL_TINT = "#f3ebfd"

HEADER_BG = f"linear-gradient(90deg, {ACCENT_DOMESTIC} 0%, {ACCENT_INTL} 100%)"  # 왼쪽 국내색 -> 오른쪽 해외색


def _escape(value) -> str:
    return html.escape(str(value))


def _format_week_label_kr(week_label: str) -> str:
    """"2026-31"(ISO 연도-주차) -> "2026년 7월 4주차". 월요일 기준 몇 번째 주인지 계산.
    형식이 예상과 다르면 원본 그대로 반환."""
    try:
        monday = datetime.strptime(f"{week_label}-1", "%G-%V-%u")
    except ValueError:
        return week_label
    week_of_month = ((monday.day - 1) // 7) + 1
    return f"{monday.year}년 {monday.month}월 {week_of_month}주차"


def _format_week_label_kr_short(week_label: str) -> str:
    """"2026-31"(ISO 연도-주차) -> "7월 4주차". 연도 없이 월/주차만 필요한 곳(그래프 x축 등)에 사용.
    형식이 예상과 다르면 원본 그대로 반환."""
    try:
        monday = datetime.strptime(f"{week_label}-1", "%G-%V-%u")
    except ValueError:
        return week_label
    week_of_month = ((monday.day - 1) // 7) + 1
    return f"{monday.month}월 {week_of_month}주차"


def _format_issue_html(item: dict, rank: int | None = None, accent: str = ACCENT_DOMESTIC) -> str:
    """
    이슈 하나 분량의 HTML 카드.
    대표 제목은 item["generated_title"](LLM 생성 헤드라인) 우선, 없으면 titles[0] fallback.
    rank가 있으면 원형 순위 배지 표시.
    """
    titles = item.get("titles", [])
    fallback_title = titles[0] if titles else "(제목 없음)"
    display_title = item.get("generated_title") or fallback_title
    rank_html = ""
    if rank is not None:
        rank_html = (f'<span style="display:inline-block; min-width:20px; height:20px; line-height:20px; '
                     f'text-align:center; border-radius:50%; background:{accent}; color:#fff; '
                     f'font-size:11px; font-weight:bold; margin-right:6px;">{rank}</span>')

    extra_html = ""
    if len(titles) > 1:
        extra_html = (f'<p style="margin:2px 0 4px 0; font-size:11px; color:#aaa;">'
                      f'(총 {len(titles)}건 기사를 종합)</p>')

    cross_html = ""
    if item.get("cross_axis_partner"):
        cross_html = (f'<p style="margin:2px 0 4px 0; font-size:12px; color:{accent};">'
                      f'🔗 반대 축에서도 다뤄짐: {_escape(item["cross_axis_partner"])}</p>')

    if item.get("summary"):
        body_html = f'<p style="margin:4px 0 8px 0; color:#333; font-size:13px; line-height:1.5;">{_escape(item["summary"])}</p>'
    else:
        reason = item.get("summary_skipped_reason", "사유 불명")
        body_html = (f'<p style="margin:4px 0 8px 0; color:#999; font-size:12px; font-style:italic;">'
                     f'(요약 생략 - {_escape(reason)})</p>')

    urls = item.get("urls", [])
    shown = urls[:3]
    more = f" 외 {len(urls) - 3}건" if len(urls) > 3 else ""
    links_html = ""
    if shown:
        link_tags = ", ".join(f'<a href="{_escape(u)}" style="color:{accent}; text-decoration:none;">원문</a>' for u in shown)
        links_html = f'<p style="margin:0; font-size:12px; color:#888;">원문 링크: {link_tags}{more}</p>'

    return f"""
    <div style="margin-bottom:12px; padding:12px 14px; border:1px solid #eee; border-radius:8px; background:#fff;">
      <p style="margin:0; font-weight:bold; font-size:14px; color:#111;">{rank_html}{_escape(display_title)}</p>
      {extra_html}
      {cross_html}
      {body_html}
      {links_html}
    </div>
    """


def _axis_label_html(title: str, accent: str) -> str:
    """국내/해외 컬러바 라벨. 좌우 2단 배치에서 "왼쪽=국내, 오른쪽=해외"를 명시."""
    return (f'<h3 style="font-size:16px; color:#222; margin:20px 0 10px 0; '
            f'padding-left:10px; border-left:4px solid {accent};">{_escape(title)}</h3>')


def _format_category_comparison_axis_html(axis_data: dict[str, dict] | None, accent: str) -> str:
    """카테고리별 지난주 대비 증감 표(축 하나 분량). 2단 레이아웃에서 좌우 배치용. 가나다순("기타"만 맨 뒤)."""
    if not axis_data:
        return '<p style="font-size:13px; color:#999; margin:4px 0;">(비교할 지난주 데이터 없음)</p>'
    rows = []
    for category in sorted(axis_data.keys(), key=lambda c: (c == "기타", c)):
        values = axis_data[category]
        delta = values["delta"]
        sign = "+" if delta >= 0 else ""
        color = "#1a7f37" if delta > 0 else ("#c0392b" if delta < 0 else "#888")
        rows.append(
            f'<tr>'
            f'<td style="padding:6px 8px; border-bottom:1px solid #f0f0f0; font-size:13px;">{_escape(category)}</td>'
            f'<td style="padding:6px 8px; border-bottom:1px solid #f0f0f0; font-size:13px; text-align:right;">{values["this_week"]}건</td>'
            f'<td style="padding:6px 8px; border-bottom:1px solid #f0f0f0; font-size:13px; text-align:right; color:#999;">{values["last_week"]}건</td>'
            f'<td style="padding:6px 8px; border-bottom:1px solid #f0f0f0; font-size:13px; text-align:right; '
            f'color:{color}; font-weight:bold;">{sign}{delta}</td>'
            f'</tr>'
        )
    return (
        f'<table style="width:100%; border-collapse:collapse;">'
        f'<tr style="background:#fafafa;">'
        f'<th style="text-align:left; padding:6px 8px; font-size:11px; color:#888;">카테고리</th>'
        f'<th style="text-align:right; padding:6px 8px; font-size:11px; color:#888;">이번 주</th>'
        f'<th style="text-align:right; padding:6px 8px; font-size:11px; color:#888;">지난주</th>'
        f'<th style="text-align:right; padding:6px 8px; font-size:11px; color:#888;">증감</th>'
        f'</tr>{"".join(rows)}</table>'
    )


def _empty_cell_html(text: str = "(해당 없음)") -> str:
    return f'<p style="color:#999; font-size:13px;">{_escape(text)}</p>'


def _format_section_html_aligned(left_label: str, left_items: list[dict],
                                  right_label: str, right_items: list[dict],
                                  left_accent: str = ACCENT_DOMESTIC, right_accent: str = ACCENT_INTL) -> str:
    """
    국내/해외를 순위별로 같은 행에 정렬(한쪽이 짧아도 그쪽 칸만 비고 반대쪽까지
    위로 안 쏠림). 한쪽이 0건이면 "이번 주 이슈 없음" 1줄만 표시.
    """
    left_header = _axis_label_html(left_label, left_accent)
    right_header = _axis_label_html(right_label, right_accent)

    left_cards = [_format_issue_html(item, rank=i, accent=left_accent) for i, item in enumerate(left_items, start=1)]
    right_cards = [_format_issue_html(item, rank=i, accent=right_accent) for i, item in enumerate(right_items, start=1)]

    if not left_cards:
        left_cards = [_empty_cell_html("(이번 주 이슈 없음)")]
    if not right_cards:
        right_cards = [_empty_cell_html("(이번 주 이슈 없음)")]

    rows = [f'<tr><td width="50%" valign="top" style="padding-right:14px;">{left_header}</td>'
            f'<td width="50%" valign="top" style="padding-left:14px;">{right_header}</td></tr>']
    for i in range(max(len(left_cards), len(right_cards))):
        left_cell = left_cards[i] if i < len(left_cards) else ""
        right_cell = right_cards[i] if i < len(right_cards) else ""
        rows.append(f'<tr><td width="50%" valign="top" style="padding-right:14px;">{left_cell}</td>'
                    f'<td width="50%" valign="top" style="padding-left:14px;">{right_cell}</td></tr>')

    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="width:100%; border-collapse:collapse; table-layout:fixed;">'
            + "".join(rows) + '</table>')


def _format_category_html_aligned(domestic_by_category: dict[str, list[dict]],
                                   international_by_category: dict[str, list[dict]],
                                   left_label: str = "국내", right_label: str = "해외",
                                   left_accent: str = ACCENT_DOMESTIC, right_accent: str = ACCENT_INTL,
                                   left_tint: str = ACCENT_DOMESTIC_TINT, right_tint: str = ACCENT_INTL_TINT) -> str:
    """
    같은 카테고리가 국내/해외 양쪽에서 같은 행에 오도록 카테고리 단위로 행을
    맞춤(한쪽에만 있는 카테고리도 반대쪽에 빈 칸으로 자리 유지). 가나다순 고정
    ("기타"만 예외로 맨 뒤).
    """
    seen = set(domestic_by_category) | set(international_by_category)
    categories = sorted(c for c in seen if c != "기타")
    if "기타" in seen:
        categories.append("기타")

    if not categories:
        return ""

    def _tag(text: str, accent: str, tint: str) -> str:
        return (f'<span style="display:inline-block; padding:4px 14px; border-radius:14px; '
                f'background:{tint}; color:{accent}; font-size:15px; font-weight:bold; '
                f'margin:14px 0 8px 0;">{_escape(text)}</span>')

    rows = []
    for category in categories:
        left_items = domestic_by_category.get(category, [])
        right_items = international_by_category.get(category, [])

        rows.append(f'<tr><td width="50%" valign="top" style="padding-right:14px;">{_tag(category, left_accent, left_tint)}</td>'
                    f'<td width="50%" valign="top" style="padding-left:14px;">{_tag(category, right_accent, right_tint)}</td></tr>')

        left_cards = [_format_issue_html(item, rank=i, accent=left_accent) for i, item in enumerate(left_items, start=1)]
        right_cards = [_format_issue_html(item, rank=i, accent=right_accent) for i, item in enumerate(right_items, start=1)]
        if not left_cards:
            left_cards = [_empty_cell_html()]
        if not right_cards:
            right_cards = [_empty_cell_html()]

        for i in range(max(len(left_cards), len(right_cards))):
            left_cell = left_cards[i] if i < len(left_cards) else ""
            right_cell = right_cards[i] if i < len(right_cards) else ""
            rows.append(f'<tr><td width="50%" valign="top" style="padding-right:14px;">{left_cell}</td>'
                        f'<td width="50%" valign="top" style="padding-left:14px;">{right_cell}</td></tr>')

    return ('<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
            'style="width:100%; border-collapse:collapse; table-layout:fixed;">'
            + "".join(rows) + '</table>')


def _two_column_table(left_html: str, right_html: str) -> str:
    """국내/해외 두 블록을 좌우로 배치. flexbox/grid 대신 table 사용(구형 Outlook 호환)."""
    return f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="width:100%; border-collapse:collapse; table-layout:fixed;">
      <tr>
        <td width="50%" valign="top" style="padding-right:14px;">{left_html}</td>
        <td width="50%" valign="top" style="padding-left:14px;">{right_html}</td>
      </tr>
    </table>
    """


def render_email_html(week_label: str, domestic_summarized: list[dict], international_summarized: list[dict],
                       domestic_by_category: dict[str, list[dict]],
                       international_by_category: dict[str, list[dict]],
                       failed_sources: list[str],
                       category_comparison: dict[str, dict[str, dict]] | None = None,
                       trend_chart_pngs: dict[str, bytes | None] | None = None,
                       embed_images_as: str = "cid",
                       error_codes: list[str] | None = None) -> tuple[str, dict[str, bytes]]:
    """
    scored.json 데이터로 이메일 본문 HTML 생성. 폭 1000px, 옅은 회색 배경 위
    흰색 콘텐츠 카드, 국내/해외 좌우 2단 레이아웃.

    trend_chart_pngs: {"국내": PNG bytes|None, "해외": PNG bytes|None} -
    render_category_trend_chart()로 호출부가 미리 만들어서 넘김(PDF용/이메일용
    두 번 호출해도 matplotlib 렌더링이 중복 안 되게).

    embed_images_as: "cid"(기본, 실제 이메일 발송용) 또는 "data_uri"(PDF 변환용).
    Gmail이 base64 인라인 이미지를 막아서 실제 발송은 "cid"로 참조만 걸고,
    진짜 이미지는 send_email()이 Content-ID 붙은 MIME 인라인 첨부로 붙인다
    (반환값 inline_images). Playwright는 cid: 스킴을 못 읽으므로 PDF 변환
    시엔 "data_uri"로 호출.

    error_codes: main.py가 로그에서 뽑아낸 🔴 조치필요 코드 목록. PDF에는
    안 넣음(embed_images_as=="cid"일 때만 렌더링) - 코드만 작게 표시.

    반환: (html_content, inline_images) - inline_images는 embed_images_as="cid"일
    때만 채워짐({cid_name: PNG bytes}), "data_uri"면 항상 빈 dict.
    """
    inline_images: dict[str, bytes] = {}

    header_html = f"""
    <div style="background:{HEADER_BG}; padding:26px 32px; border-radius:10px 10px 0 0;">
      <p style="margin:0; font-size:11px; letter-spacing:2px; color:#9fc0ff; font-weight:bold;">NEWSLETTER</p>
      <h1 style="margin:6px 0 0 0; font-size:22px; color:#fff; font-weight:bold;">사료·축산업 뉴스 큐레이션</h1>
      <p style="margin:5px 0 0 0; font-size:13px; color:#c9dcff;">{_escape(_format_week_label_kr(week_label))}</p>
    </div>
    """

    section_header = lambda text: f'<h2 style="font-size:18px; color:#111; margin:26px 0 12px 0;">{_escape(text)}</h2>'

    parts = [
        '<div style="font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Arial, sans-serif; '
        'background:#f2f4f7; padding:24px 0; word-break:keep-all; overflow-wrap:break-word;">',
        '<div style="max-width:1000px; margin:0 auto; background:#fff; border-radius:10px; '
        'overflow:hidden; border:1px solid #e5e5e5;">',
        header_html,
        '<div style="padding:24px 32px; color:#333;">',
    ]

    if category_comparison:
        parts.append(section_header("카테고리별 지난주 대비 증감"))
        parts.append(_two_column_table(
            _axis_label_html("국내", ACCENT_DOMESTIC) + _format_category_comparison_axis_html(category_comparison.get("국내"), ACCENT_DOMESTIC),
            _axis_label_html("해외", ACCENT_INTL) + _format_category_comparison_axis_html(category_comparison.get("해외"), ACCENT_INTL),
        ))

    if trend_chart_pngs and (trend_chart_pngs.get("국내") or trend_chart_pngs.get("해외")):
        def _chart_img_tag(png_bytes: bytes | None, cid_name: str) -> str:
            if not png_bytes:
                return '<p style="font-size:13px; color:#999;">(그래프 생성 실패)</p>'
            if embed_images_as == "cid":
                inline_images[cid_name] = png_bytes
                src = f"cid:{cid_name}"
            else:
                src = _png_to_data_uri(png_bytes)
            return f'<img src="{src}" width="100%" style="max-width:100%; display:block;">'

        parts.append(section_header("카테고리별 최근 추이"))
        parts.append(_two_column_table(
            _axis_label_html("국내", ACCENT_DOMESTIC) + _chart_img_tag(trend_chart_pngs.get("국내"), "domestic_trend_chart"),
            _axis_label_html("해외", ACCENT_INTL) + _chart_img_tag(trend_chart_pngs.get("해외"), "international_trend_chart"),
        ))

    parts.append(section_header("주간 Top 이슈"))
    parts.append(_format_section_html_aligned(
        "국내", domestic_summarized, "해외", international_summarized,
        left_accent=ACCENT_DOMESTIC, right_accent=ACCENT_INTL,
    ))

    parts.append(section_header("카테고리별 Top N"))
    parts.append(_format_category_html_aligned(
        domestic_by_category, international_by_category,
        left_accent=ACCENT_DOMESTIC, right_accent=ACCENT_INTL,
        left_tint=ACCENT_DOMESTIC_TINT, right_tint=ACCENT_INTL_TINT,
    ))

    if failed_sources:
        parts.append(
            f'<p style="margin-top:24px; font-size:12px; color:#c0392b;">'
            f'참고 - 이번 실행에서 실패한 소스: {_escape(", ".join(failed_sources))}</p>'
        )

    parts.append(
        '<p style="margin-top:28px; padding-top:16px; border-top:1px solid #eee; '
        'font-size:11px; color:#aaa; text-align:center;">'
        '이 메일은 사료·축산업 뉴스 큐레이션 시스템이 매주 자동으로 발송합니다.<br>'
        'AI가 자동으로 생성한 요약·헤드라인이 포함되어 있어 실제 내용과 다를 수 있습니다. '
        '정확한 내용은 원문 링크를 확인해주세요.</p>'
    )

    if error_codes and embed_images_as == "cid":
        # PDF에는 안 넣음(실제 이메일에만). 코드만, 안내문보다 더 흐리게.
        parts.append(
            f'<p style="margin-top:6px; font-size:9px; color:#ccc; text-align:center;">'
            f'{_escape(", ".join(error_codes))}</p>'
        )

    parts.append('</div>')
    parts.append('</div>')
    parts.append('</div>')
    return "".join(parts), inline_images


def render_category_trend_chart(trend_entries: list[dict], axis: str) -> bytes | None:
    """
    category_aggregator.load_weekly_trend() 결과로 카테고리별 최근 N주 추이 선그래프 PNG 생성.
    "기타"는 항상 압도적으로 커서 나머지 카테고리 흐름이 안 보이게 되므로 그래프에서는 제외.
    카테고리 순서는 가나다순 고정 - 국내/해외 두 그래프에서 같은 카테고리가 항상 같은 색이 되게 함.

    데이터가 2주 미만이면(추이라고 부를 게 없음) None. 폰트/matplotlib 관련 실패도 예외 없이 None만 반환 - 그래프 하나 없다고 이메일 발송 전체가
    막히면 안 됨.
    """
    if not trend_entries or len(trend_entries) < 2:
        return None

    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from keyword_tagger import CATEGORY_KEYWORDS

        # 이름만 지정(rcParams["font.family"]="NanumGothic")하면 matplotlib 폰트 캐시 타이밍 문제로 못 찾을 수 있어 파일 경로로 직접 등록.
        _font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        if os.path.exists(_font_path):
            fm.fontManager.addfont(_font_path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
        else:
            print(f"[deploy] 🟡 주의 [DP-01] - 나눔고딕 폰트 파일 없음({_font_path}) - "
                  f"run-pipline.yml의 fonts-nanum 설치 스텝 확인 필요, 이번엔 한글이 깨질 수 있음")
        plt.rcParams["axes.unicode_minus"] = False

        categories = sorted(CATEGORY_KEYWORDS.keys())
        week_labels = [_format_week_label_kr_short(e["week_label"]) for e in trend_entries]
        colors = plt.cm.tab10.colors

        fig, ax_plot = plt.subplots(figsize=(9, 4), dpi=120)
        plotted_any = False
        for i, category in enumerate(categories):
            values = [e.get(axis, {}).get(category, 0) for e in trend_entries]
            if not any(values):
                continue  # 기간 내내 0건인 카테고리는 범례만 차지하니 생략
            ax_plot.plot(week_labels, values, marker="o", label=category,
                         color=colors[i % len(colors)], linewidth=2)
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            return None

        ax_plot.set_title(f"{axis} 카테고리별 최근 {len(trend_entries)}주 추이 ('기타' 제외)", fontsize=11)
        ax_plot.set_ylabel("건수")
        ax_plot.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, frameon=False)
        ax_plot.spines["top"].set_visible(False)
        ax_plot.spines["right"].set_visible(False)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        print(f"[deploy] 🟡 주의 [DP-02] - {axis} 카테고리 트렌드 그래프 생성 실패 - 이메일에서 이 그래프만 생략: "
              f"{type(e).__name__} - {e!r}")
        return None


def _png_to_data_uri(png_bytes: bytes) -> str:
    """PNG bytes를 이메일 HTML의 <img>에 바로 박아넣을 수 있는 base64 data URI로 변환."""
    import base64
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_email_pdf(html_content: str) -> bytes | None:
    """
    render_email_html()이 만든 HTML을 PDF로 변환. Playwright/Chromium 사용(WATT_collector.py와 같은 브라우저, 별도 의존성 불필요).

    링크는 href 그대로 살아서 PDF에서도 클릭 가능. 여백 0 + scale=0.79로 콘텐츠 폭(1000px)을 A4 폭(≈793px)에 맞춤(여백만 없애면 우측이 잘림).
    실패해도 예외 없이 None만 반환 - HTML 본문 발송이 우선, PDF는 부가 기능.
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(html_content, wait_until="networkidle")
                pdf_bytes = page.pdf(
                    format="A4", print_background=True,
                    margin={"top": "0mm", "bottom": "0mm", "left": "0mm", "right": "0mm"},
                    scale=0.79,
                )
            finally:
                browser.close()
        return pdf_bytes
    except Exception as e:
        print(f"[deploy] 🟡 주의 [DP-03] - PDF 변환 실패 - 이메일은 HTML 본문만 발송(첨부 없이): "
              f"{type(e).__name__} - {e!r}")
        return None


def send_email(html_content: str, subject: str, recipients: list[str],
               smtp_user: str, smtp_app_password: str,
               pdf_attachment: bytes | None = None, pdf_filename: str = "weekly.pdf",
               inline_images: dict[str, bytes] | None = None) -> bool:
    """
    Gmail SMTP(587, STARTTLS)로 발송. 성공 True, 실패해도 예외 없이 False.

    구조: mixed(PDF 첨부) > related(인라인 이미지) > alternative(HTML 본문).
    inline_images({cid_name: PNG bytes})를 넘기면 HTML의 <img src="cid:cid_name">가 가리키는 이미지를 Content-ID 붙은 MIME 파트로 같이 넣는다(render_email_html의 embed_images_as="cid" 모드와 짝 맞춰 사용).
    """
    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = ", ".join(recipients)

    related = MIMEMultipart("related")
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(html_content, "html", "utf-8"))
    related.attach(body)

    for cid_name, png_bytes in (inline_images or {}).items():
        img_part = MIMEImage(png_bytes, _subtype="png")
        img_part.add_header("Content-ID", f"<{cid_name}>")
        img_part.add_header("Content-Disposition", "inline", filename=f"{cid_name}.png")
        related.attach(img_part)

    msg.attach(related)

    if pdf_attachment:
        pdf_part = MIMEApplication(pdf_attachment, _subtype="pdf")
        pdf_part.add_header("Content-Disposition", "attachment", filename=pdf_filename)
        msg.attach(pdf_part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_app_password)
            server.sendmail(smtp_user, recipients, msg.as_string())
    except (smtplib.SMTPException, OSError) as e:
        print(f"[deploy] 🔴 조치필요 [DP-04] - 이메일 발송 실패: {type(e).__name__} - {e!r}")
        return False

    attach_note = f", PDF 첨부 포함({len(pdf_attachment):,} bytes)" if pdf_attachment else " (PDF 첨부 없음)"
    inline_note = f", 인라인 이미지 {len(inline_images)}개" if inline_images else ""
    print(f"[deploy] 이메일 발송 완료 -> {', '.join(recipients)}{attach_note}{inline_note}")
    return True


def send_weekly_email(week_label: str, domestic_summarized: list[dict], international_summarized: list[dict],
                       domestic_by_category: dict[str, list[dict]],
                       international_by_category: dict[str, list[dict]],
                       failed_sources: list[str],
                       category_comparison: dict[str, dict[str, dict]] | None = None,
                       weekly_trend: list[dict] | None = None,
                       error_codes: list[str] | None = None) -> bool:
    """main.py 호출 진입점. 인증정보 없으면 안전하게 생략."""
    smtp_user = os.environ.get("SMTP_USER")
    smtp_app_password = os.environ.get("SMTP_APP_PASSWORD")
    recipients_raw = os.environ.get("EMAIL_RECIPIENTS")

    if not smtp_user or not smtp_app_password:
        print("[deploy] 🔴 조치필요 [DP-05] - SMTP_USER/SMTP_APP_PASSWORD 없음 - 이메일 발송 생략")
        return False
    if not recipients_raw:
        print("[deploy] 🔴 조치필요 [DP-06] - EMAIL_RECIPIENTS 없음 - 이메일 발송 생략")
        return False

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]
    if not recipients:
        print("[deploy] 🔴 조치필요 [DP-07] - EMAIL_RECIPIENTS가 비어있음(콤마만 있거나 공백) - 이메일 발송 생략")
        return False

    # 트렌드 차트는 여기서 딱 한 번만 렌더링(matplotlib 호출 비용) - 아래 cid용/data_uri용 두 HTML이 같은 PNG bytes를 재사용한다.
    trend_chart_pngs = None
    if weekly_trend and len(weekly_trend) >= 2:
        trend_chart_pngs = {
            "국내": render_category_trend_chart(weekly_trend, "국내"),
            "해외": render_category_trend_chart(weekly_trend, "해외"),
        }

    html_content, inline_images = render_email_html(
        week_label, domestic_summarized, international_summarized,
        domestic_by_category, international_by_category, failed_sources,
        category_comparison, trend_chart_pngs, embed_images_as="cid",
        error_codes=error_codes)

    subject = f"[사료·축산뉴스] {_format_week_label_kr(week_label)} 주간 큐레이션"

    # PDF는 Playwright(Chromium)가 렌더링하는데 cid: 스킴을 못 읽으므로 data_uri 버전을 따로 만들어서 넘김 - 실제 발송 본문(cid)과는 별개 HTML.
    pdf_html_content, _ = render_email_html(
        week_label, domestic_summarized, international_summarized,
        domestic_by_category, international_by_category, failed_sources,
        category_comparison, trend_chart_pngs, embed_images_as="data_uri")
    pdf_bytes = render_email_pdf(pdf_html_content)
    pdf_filename = f"feed_livestock_news_{week_label}.pdf"  # 파일명은 ASCII로(한글 파일명 인코딩 이슈 회피)

    return send_email(html_content, subject, recipients, smtp_user, smtp_app_password,
                       pdf_attachment=pdf_bytes, pdf_filename=pdf_filename,
                       inline_images=inline_images)
