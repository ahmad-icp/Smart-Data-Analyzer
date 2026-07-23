from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_runtime_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Smart Data Analyzer Pro"
    assert len(app.tabs) >= 6
