from pathlib import Path
from pipeline import validate_row, transform_good_row, run_pipeline


def make_sample_row():
    return {
        "ID": "10",
        "Title": "SQL Injection on Login Page",
        "Category": "Web Application Security",
        "Attack Type": "SQL Injection",
        "Scenario Description": "An attacker enters a crafted SQL string into the login form to bypass authentication.",
        "Tools Used": "sqlmap, Burp Suite",
        "Attack Steps ": "Step 1: Intercept login POST. Step 2: Inject payload.",
        "Target Type": "Web Server",
        "Vulnerability": "Unsanitized SQL query",
        "MITRE Technique": "T1190",
        "Impact": "Unauthorized access",
        "Detection Method": "WAF logs",
        "Solution": "Parameterized queries",
        "Tags": "sqli, web",
        "Source": "OWASP",
        "Unnamed: 15": "",
    }


def test_valid_row_passes():
    seen = set()
    ok, reason, cleaned = validate_row(make_sample_row(), seen_keys=seen)
    assert ok is True
    assert reason == "ok"
    out = transform_good_row(cleaned)
    assert out["ID"] == 10
    assert out["tool_count"] == 2


def test_missing_required_field_quarantined():
    row = make_sample_row()
    row["Scenario Description"] = "   "
    ok, reason, _ = validate_row(row)
    assert ok is False
    assert reason == "missing:Scenario Description"


def test_bad_id_quarantined():
    row = make_sample_row()
    row["ID"] = "abc"
    ok, reason, _ = validate_row(row)
    assert ok is False
    assert reason == "bad_number:ID"


def test_shifted_unnamed15_quarantined():
    row = make_sample_row()
    row["Unnamed: 15"] = "Educational Simulation"
    ok, reason, _ = validate_row(row)
    assert ok is False
    assert reason == "column_shift:Unnamed_15"


def test_duplicate_row_quarantined():
    seen = set()
    r1 = make_sample_row()
    r2 = make_sample_row()
    r2["ID"] = "11"
    ok1, _, _ = validate_row(r1, seen_keys=seen)
    ok2, reason2, _ = validate_row(r2, seen_keys=seen)
    assert ok1 is True
    assert ok2 is False
    assert reason2 == "duplicate_scenario"
