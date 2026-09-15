"""
Example test — run with:  python -m pytest tests/

# TODO: add equivalent tests for pipeline/fusion.py and pipeline/decision.py
# once the fusion strategy (see pipeline/fusion.py #TODO) is finalized —
# those are the modules most worth locking down with tests since they hold
# the actual business logic. Build test fixtures using utils/geometry.py
# dataclasses directly (no need to run real models), e.g.:
#
#   instance = SegmentationInstance(instance_id=0, class_name="container",
#                                    confidence=0.9, bbox=BBox(0, 0, 200, 100))
#   barcode = BarcodeDetection(value="MSCU1234567", symbology="CODE128",
#                               confidence=0.8, bbox=BBox(10, 10, 30, 20))
#   matches = fusion.fuse([instance], [barcode], lcd_records, containers_db)
#   assert matches[0].container_number == "MSCU1234567"
"""

from pathlib import Path

from pipeline.lcd_formatter import format_lcd_csv

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
