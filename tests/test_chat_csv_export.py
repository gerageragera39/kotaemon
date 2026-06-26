import csv
import json

from ktem.utils import chat_export


def test_export_chat_csv_writes_question_answer_and_context(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_export, "CHAT_OUTPUT_DIR", tmp_path)

    csv_path = chat_export.export_chat_csv(
        convo_id="conversation:1",
        messages=[["What is D3B?", "D3B is a study program."]],
        retrieval_history=["<h5>Evidence</h5><p>Context chunk &amp; source.</p>"],
        plot_history=[{"kind": "plot"}],
        selected={"files": ["source.pdf"]},
    )

    assert csv_path == tmp_path / "chat_conversation_1.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["question"] == "What is D3B?"
    assert rows[0]["answer"] == "D3B is a study program."
    assert "Context chunk & source." in rows[0]["context"]
    assert json.loads(rows[0]["contexts"]) == [rows[0]["context"]]
    assert json.loads(rows[0]["plot_data"]) == {"kind": "plot"}
    assert json.loads(rows[0]["selected_sources"]) == {"files": ["source.pdf"]}
