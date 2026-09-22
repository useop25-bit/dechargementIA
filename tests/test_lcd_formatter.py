"""
Example test — run with:  python -m pytest tests/
"""

from pathlib import Path

from srcs.pipeline.lcd_formatter import format_lcd_csv
from srcs.pipeline.decision import decide
from srcs.utils.geometry import BBox, ContainerMatch, SegmentationInstance

EXAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "lcd" / "example_lcd.csv"

def test_format_lcd_csv_keeps_only_expected_fields():
    records = format_lcd_csv(EXAMPLE_CSV)

    assert len(records) == 5
    for record in records:
        assert set(record.keys()) == {
            "reference_number",
            "container_number",
            "description",
            "quantity",
            "weight_kg",
        }

    first = records[0]
    assert first["reference_number"] == "REF-00231"
    assert first["container_number"] == "MSCU1234567"
    assert first["quantity"] == 12
    assert first["weight_kg"] == 340.5

def test_format_lcd_csv_skips_rows_missing_ids(tmp_path):
    csv_with_bad_row = tmp_path / "bad.csv"
    csv_with_bad_row.write_text(
        "reference_number,container_number,description,quantity,weight_kg\n"
        "REF-1,CONT-1,Widget,1,10.0\n"
        ",CONT-2,Missing ref,1,10.0\n"
    )

    records = format_lcd_csv(csv_with_bad_row)
    assert len(records) == 1
    assert records[0]["reference_number"] == "REF-1"

def test_leftmost_strategy_selects_leftmost_takeable_instance(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "DECISION_STRATEGY", "leftmost")
    instances = [
        SegmentationInstance(0, "container", 0.9, BBox(200, 10, 80, 80)),
        SegmentationInstance(1, "container", 0.9, BBox(20, 10, 80, 80)),
    ]
    matches = [ContainerMatch(0, None, None, 0.0), ContainerMatch(1, None, None, 0.0)]

    decisions = decide(instances, matches, [])

    assert len(decisions) == 1
    assert decisions[0].selection_zone.x == 20