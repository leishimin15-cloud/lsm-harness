from lsm_harness.config import Settings
from lsm_harness.doctor import checks


def test_doctor_checks_are_local_and_key_aware(tmp_path):
    results = checks(Settings(api_key="configured", home=tmp_path))
    assert all(ok for _, ok, _ in results)
    names = {name for name, _, _ in results}
    assert "SQLite FTS5 trigram" in names
    assert "Model API Key" in names


def test_cli_module_imports_prompt_toolkit():
    from lsm_harness.gateway import cli

    assert cli.PromptSession is not None
