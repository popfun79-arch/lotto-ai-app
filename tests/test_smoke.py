from pathlib import Path

def test_app_syntax():
    root=Path(__file__).resolve().parents[1]
    source=(root/"app.py").read_text(encoding="utf-8")
    compile(source,str(root/"app.py"),"exec")

def test_required_files():
    root=Path(__file__).resolve().parents[1]
    for name in ["app.py","requirements.txt","README.md"]:
        assert (root/name).exists()
