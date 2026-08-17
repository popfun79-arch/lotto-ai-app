from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_data_diagnosis_uses_analysis_window():
    text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'a.metric("분석 시작 회차", int(analysis["round"].min()))' in text
    assert 'b.metric("분석 최신 회차", int(analysis["round"].max()))' in text
    assert 'c.metric("분석 회차 수", len(analysis))' in text
    assert 'st.dataframe(analysis.tail(20)' in text
