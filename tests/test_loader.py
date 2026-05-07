from repo_aware_ai.loader import load_repo_files


def test_loader_picks_up_code_files(tiny_repo):
    files = load_repo_files(tiny_repo)
    paths = {f.path for f in files}

    assert "src/alpha.py" in paths
    assert "src/beta.js" in paths
    assert "README.md" in paths


def test_loader_skips_ignored_dirs(tiny_repo):
    files = load_repo_files(tiny_repo)
    paths = {f.path for f in files}
    assert not any(p.startswith("node_modules/") for p in paths)


def test_loader_uses_posix_separators(tiny_repo):
    files = load_repo_files(tiny_repo)
    for f in files:
        assert "\\" not in f.path
