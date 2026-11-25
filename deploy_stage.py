"""
deploy_stage.py - job3(배포) 진입점: [11] 저장 ~ [12] 이메일 발송.

job2(process)는 순위가 정해진 직후부터 요약이 하나 끝날 때마다 중간 결과를
DEPLOY_CHECKPOINT_PATH에 덮어써 둔다(run-pipline.yml이 artifact로 넘김).
이 job은 process가 정상 종료했든 시간 초과로 잘렸든 항상 실행되어, 남아 있는
가장 최근 체크포인트로 저장·발송한다 - process가 6시간 제한에 걸려도 메일은 나감
(2026-11-22 회차: process 6h0m 강제 종료로 메일·저장 모두 누락된 일 재발 방지).

의존성이 가벼운 모듈(storage, deploy)만 import - 이 job은 PyTorch/임베딩 모델을 설치하지 않음.
main.run()(로컬 단일 실행)도 save_and_deploy()를 그대로 재사용한다.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import deploy
import storage

DEPLOY_CHECKPOINT_PATH = "deploy_checkpoint.pkl"
DEPLOY_ARTICLES_PATH = "deploy_articles.pkl"
COLLECTED_ARTIFACT_PATH = "collected_articles.json"


def _atomic_pickle(obj, path: str) -> None:
    """쓰는 도중 job이 잘려도 이전 체크포인트가 깨지지 않게 임시 파일에 쓰고 교체."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def write_checkpoint(state: dict) -> None:
    try:
        _atomic_pickle(state, DEPLOY_CHECKPOINT_PATH)
    except Exception as e:  # 체크포인트 실패가 본 처리를 막으면 안 됨
        print(f"[deploy_stage] 🟡 주의 [DS-01] - 중간 결과 저장 실패(처리는 계속): {type(e).__name__} - {e!r}")


def write_articles(articles: list[dict]) -> None:
    try:
        _atomic_pickle(articles, DEPLOY_ARTICLES_PATH)
    except Exception as e:
        print(f"[deploy_stage] 🟡 주의 [DS-01] - 기사 목록 저장 실패(처리는 계속): {type(e).__name__} - {e!r}")


def save_and_deploy(state: dict, articles: list[dict]) -> None:
    """[11] 저장 + [12] 배포. state는 process가 남긴 체크포인트 형식."""
    failed_sources = state.get("failed_sources") or []

    print("\n=== [11] 저장 ===")
    # 수동 실행(workflow_dispatch)은 저장 생략 - 지난주 대비 증감 비교 기준 오염 방지
    is_manual_run = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    saved_dir = None
    if is_manual_run:
        print("[main] 수동 실행(workflow_dispatch)이라 저장을 건너뜁니다 - "
              "'지난주 대비 증감' 비교 기준 오염 방지(콘솔에 출력된 이번 실행 결과는 "
              "그대로 확인 가능, data/에는 안 남음)")
    elif not state.get("ranked"):
        print("[deploy_stage] 순위 결과가 없어(처리 단계가 순위 확정 전에 멈춤) 저장을 건너뜁니다 - "
              "지난주 대비 비교 기준 오염 방지")
    else:
        try:
            saved_dir = storage.save_week(articles, state["domestic"], state["international"],
                                           state["domestic_cat"], state["international_cat"],
                                           state.get("gdelt_timeline") or {}, failed_sources,
                                           state.get("category_distribution") or {},
                                           state.get("category_comparison"))
        except Exception as e:
            print(f"[main] 🔴 조치필요 [MN-14] - 저장 단계에서 예상 못 한 오류 발생(콘솔 로그의 결과는 그대로 유효함): "
                  f"{type(e).__name__} - {e!r}")

    print("\n=== [12] 배포 ===")
    try:
        week_label = os.path.basename(saved_dir) if saved_dir else datetime.now(timezone.utc).strftime("%G-%V")
        deploy.send_weekly_email(week_label, state["domestic"], state["international"],
                                  state["domestic_cat"], state["international_cat"],
                                  failed_sources, state.get("category_comparison"),
                                  state.get("weekly_trend") or [], state.get("error_codes") or [])
    except Exception as e:
        print(f"[main] 🔴 조치필요 [MN-15] - 배포 단계에서 예상 못 한 오류 발생(콘솔 로그의 결과는 그대로 유효함): "
              f"{type(e).__name__} - {e!r}")

    if failed_sources:
        saved_dir_note = f"{saved_dir}/scored.json에도" if saved_dir else "(data/ 파일에는 안 남았지만)"
        print(f"\n[main] 🔴 조치필요 [MN-16] - 이번 실행 실패 소스: {failed_sources} "
              f"({saved_dir_note} failed_sources로 같이 저장됨)")


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _fallback_state() -> dict:
    """체크포인트가 아예 없을 때(순위 확정 전에 process가 멈춤) - 수집 결과의 실패 소스만 살려 빈 메일 상태 구성."""
    failed_sources: list[str] = []
    gdelt_timeline: dict = {}
    try:
        with open(COLLECTED_ARTIFACT_PATH, encoding="utf-8") as f:
            collected = json.load(f)
        failed_sources = list(collected.get("failed_sources") or [])
        gdelt_timeline = collected.get("gdelt_timeline") or {}
    except Exception as e:
        print(f"[deploy_stage] 🟡 주의 [DS-02] - 수집 결과 파일도 읽지 못함: {type(e).__name__} - {e!r}")
    return {
        "ranked": False, "complete": False,
        "domestic": [], "international": [], "domestic_cat": {}, "international_cat": {},
        "failed_sources": failed_sources, "gdelt_timeline": gdelt_timeline,
        "category_distribution": {}, "category_comparison": None, "weekly_trend": [], "error_codes": [],
    }


def run() -> None:
    print("=== [job3] 배포 단계 시작 ===")
    if os.path.exists(DEPLOY_CHECKPOINT_PATH):
        state = _load_pickle(DEPLOY_CHECKPOINT_PATH)
        print(f"[deploy_stage] 중간 결과 로드 - 단계: {state.get('stage')}, 완료 여부: {state.get('complete')}")
    else:
        print("[deploy_stage] 🔴 조치필요 [DS-03] - process 단계의 중간 결과가 없음(순위 확정 전에 멈춤) - "
              "기사 없이 실패 안내만 발송")
        state = _fallback_state()
        state["error_codes"] = ["DS-03"]

    articles: list[dict] = []
    if os.path.exists(DEPLOY_ARTICLES_PATH):
        try:
            articles = _load_pickle(DEPLOY_ARTICLES_PATH)
        except Exception as e:
            print(f"[deploy_stage] 🟡 주의 [DS-02] - 기사 목록 로드 실패: {type(e).__name__} - {e!r}")

    failed_sources = list(state.get("failed_sources") or [])
    error_codes = list(state.get("error_codes") or [])
    if not state.get("complete"):
        note = ("처리 단계 시간 초과(일부 요약 생략)" if state.get("ranked")
                else "처리 단계 시간 초과(이번 주 순위 없음)")
        print(f"[deploy_stage] 🔴 조치필요 [DS-04] - {note} - process가 끝까지 못 감, 남은 중간 결과로 발송")
        failed_sources.append(note)
        error_codes.append("DS-04")
    state["failed_sources"] = failed_sources
    state["error_codes"] = list(dict.fromkeys(error_codes))

    save_and_deploy(state, articles)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[deploy_stage] 🔴 조치필요 [DS-05] - 배포 단계 전체 실패: {type(e).__name__} - {e!r}")
        sys.exit(1)
