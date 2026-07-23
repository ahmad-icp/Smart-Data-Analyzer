from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_runtime_exceptions():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Smart Data Analyzer Pro"
    assert len(app.tabs) >= 6


def test_all_workspaces_render_with_a_loaded_dataset():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=30)
    dataset = pd.DataFrame(
        {
            "age": [20, 25, 30, 35, 40, 45] * 5,
            "income": [30, 35, 40, 50, 55, 65] * 5,
            "group": ["A", "A", "A", "B", "B", "B"] * 5,
            "target": [0, 0, 0, 1, 1, 1] * 5,
        }
    )
    app.session_state["df"] = dataset.copy()
    app.session_state["original_df"] = dataset.copy()
    app.run()
    assert not app.exception
    assert any(header.value == "Statistical Analysis Lab" for header in app.header)
    assert len(app.metric) >= 5
