from pathlib import Path

def test_app_syntax():
    source=(Path(__file__).resolve().parents[1]/"app.py").read_text(encoding="utf-8")
    compile(source,"app.py","exec")

def test_v21_features_present():
    source=(Path(__file__).resolve().parents[1]/"app.py").read_text(encoding="utf-8")
    assert "normalized_probability_table" in source
    assert "backtest_report_payload" in source
    assert "walk_forward_report.json" in source
    assert "st.session_state" in source
