from __future__ import annotations

from pathlib import Path


def find_repo_root() -> Path:
    """
    현재 테스트 파일이 tests/ 또는 tests/tests/ 아래에 있어도
    app.py가 있는 저장소 루트를 안정적으로 찾습니다.
    """
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "app.py").exists():
            return parent

    raise FileNotFoundError("저장소 루트의 app.py를 찾을 수 없습니다.")


def test_app_syntax():
    root = find_repo_root()
    app_path = root / "app.py"
    source = app_path.read_text(encoding="utf-8")
    compile(source, str(app_path), "exec")


def test_v21_features_present():
    root = find_repo_root()
    source = (root / "app.py").read_text(encoding="utf-8")

    # v2.1 또는 v3 이상에서 유지되어야 하는 핵심 기능 검사
    expected_any = [
        "relative_probability",
        "relative_probability_pct",
        "build_backtest_report",
        "walk_forward",
        "st.session_state",
    ]

    assert any(token in source for token in expected_any), (
        "app.py에서 Lotto64 백테스트/상대점수 기능을 찾지 못했습니다."
    )
