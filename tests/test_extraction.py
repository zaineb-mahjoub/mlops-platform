import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml"))
from extraction import extract_text

def test_txt_extraction(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world")
    assert extract_text(str(f)) == "Hello world"

def test_unsupported_extension_raises(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("a,b,c")
    try:
        extract_text(str(f))
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    