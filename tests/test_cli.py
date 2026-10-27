import json

from streamwatch.cli import main


def test_detect_csv_column(tmp_path, capsys):
    path = tmp_path / "m.csv"
    rows = ["ts,latency"] + [f"{i},{10 + (i % 3) * 0.1}" for i in range(100)]
    rows[80] = "79,50"
    path.write_text("\n".join(rows))
    assert main(["detect", str(path), "--column", "latency"]) == 0
    out = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [o["index"] for o in out] == [79]


def test_detect_stdin(monkeypatch, capsys):
    import io
    data = "\n".join(["1.0"] * 30 + ["9.0"])
    monkeypatch.setattr("sys.stdin", io.StringIO(data))
    main(["detect", "--detector", "ewma"])
    assert json.loads(capsys.readouterr().out)["index"] == 30


def test_bench_prints_table(capsys):
    main(["bench", "--n", "600"])
    out = capsys.readouterr().out
    assert "iforest" in out and "seasonal-bursts" in out
