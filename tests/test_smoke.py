from pathlib import Path

def test_all_python_syntax():
    root = Path(__file__).resolve().parents[1]
    for path in root.rglob("*.py"):
        if ".venv" in str(path):
            continue
        compile(path.read_text(encoding="utf-8"), str(path), "exec")

def test_required_files():
    root = Path(__file__).resolve().parents[1]
    for relative in [
        "app.py",
        "requirements.txt",
        "README.md",
        "lotto64/config.py",
        ".github/workflows/ci.yml",
        ".streamlit/config.toml",
    ]:
        assert (root / relative).exists()
