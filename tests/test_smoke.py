import pathlib

def test_repo_layout():
    assert pathlib.Path("configs/config.yaml").exists()
    assert pathlib.Path("src/train.py").exists()
