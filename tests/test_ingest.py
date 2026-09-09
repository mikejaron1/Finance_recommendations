"""Tests for reading a user's own documents into their profile.

The bar for this feature is that it never silently changes a number the user
entered themselves, so most of these tests are about *proposals* and
*conflicts* rather than about parsing accuracy.
"""

from __future__ import annotations

import io

import pytest

from finrec import ingest
from finrec.ingest import Finding
from finrec.profile import Profile

CSV = """date,description,amount
2024-01-05,RENT PAYMENT,-2400
2024-01-07,WHOLE FOODS,-180.22
2024-01-12,SHELL GAS,-60
2024-02-05,RENT PAYMENT,-2400
2024-02-09,WHOLE FOODS,-210.10
2024-02-14,NETFLIX,-15.99
2024-03-05,RENT PAYMENT,-2400
2024-03-11,WHOLE FOODS,-190
2024-03-20,UNITED AIRLINES,-540
"""

STATEMENT = """Chase Total Checking      $12,450.10
Roth IRA                  $88,200.00
401(k) Traditional        $210,500
Credit Card ending 4321   $4,120.55
Card APR 22.99%
Mortgage balance          $412,000
Mortgage rate 6.125%
Student loan (Navient) 28,400
Showing 25 transactions
"""


def fields(findings) -> dict:
    return {f.field: f.value for f in findings}


class TestCsvExtraction:
    def test_derives_monthly_spending(self):
        found, summary = ingest.extract_from_csv(io.StringIO(CSV), "chase.csv")
        assert summary["months"] == 3
        assert fields(found)["monthly_spending"] == pytest.approx(2798.77, abs=0.05)

    def test_essential_spending_excludes_discretionary(self):
        found, _ = ingest.extract_from_csv(io.StringIO(CSV), "chase.csv")
        essential = fields(found)["monthly_essential_spending"]
        assert essential == pytest.approx(2400.0, abs=1.0)
        assert essential < fields(found)["monthly_spending"]

    def test_confidence_rises_with_more_history(self):
        one_month = "date,description,amount\n2024-01-05,RENT,-2400\n"
        short, _ = ingest.extract_from_csv(io.StringIO(one_month), "a.csv")
        long, _ = ingest.extract_from_csv(io.StringIO(CSV), "b.csv")
        assert short[0].confidence < long[0].confidence

    def test_empty_file_yields_nothing(self):
        found, _ = ingest.extract_from_csv(io.StringIO("date,description,amount\n"), "e.csv")
        assert found == []

    def test_records_the_filename_as_provenance(self):
        found, _ = ingest.extract_from_csv(io.StringIO(CSV), "amex-2024.csv")
        assert all(f.source == "amex-2024.csv" for f in found)


class TestTextExtraction:
    def test_reads_labelled_balances(self):
        got = fields(ingest.extract_from_text(STATEMENT))
        assert got["cash"] == pytest.approx(12450.10)
        assert got["roth_balance"] == pytest.approx(88200.0)
        assert got["mortgage_balance"] == pytest.approx(412000.0)
        assert got["student_loans"] == pytest.approx(28400.0)

    def test_ignores_digits_that_belong_to_the_label(self):
        """'401(k) Traditional $210,500' must not read as $401."""
        got = fields(ingest.extract_from_text(STATEMENT))
        assert got["traditional_401k"] == pytest.approx(210500.0)

    def test_ignores_account_number_fragments(self):
        """'Credit Card ending 4321  $4,120.55' must not read as $4,321."""
        got = fields(ingest.extract_from_text(STATEMENT))
        assert got["credit_card_debt"] == pytest.approx(4120.55)

    def test_rates_are_stored_as_fractions(self):
        got = fields(ingest.extract_from_text(STATEMENT))
        assert got["credit_card_rate"] == pytest.approx(0.2299)
        assert got["mortgage_rate"] == pytest.approx(0.06125)

    def test_percent_line_is_not_also_read_as_dollars(self):
        found = ingest.extract_from_text("Card APR 22.99%")
        assert [f.field for f in found] == ["credit_card_rate"]

    def test_skips_unlabelled_numbers(self):
        assert ingest.extract_from_text("$45,000\n$12,000") == []

    def test_skips_row_counts(self):
        assert "cash" not in fields(ingest.extract_from_text("Showing 25 cash transactions"))

    def test_rejects_implausible_rates(self):
        assert ingest.extract_from_text("Mortgage rate 91%") == []

    def test_preserves_multiple_accounts_for_explicit_aggregation(self):
        text = "Roth IRA $50,000\nRoth 401k rollover $60,000"
        found = fields(ingest.extract_from_text(text))
        assert found["roth_balance"] in (50000.0, 60000.0)
        assert len([f for f in ingest.extract_from_text(text) if f.field == "roth_balance"]) == 2

    def test_evidence_is_captured_for_review(self):
        found = ingest.extract_from_text("Roth IRA $88,200.00")
        assert "Roth IRA" in found[0].evidence

    def test_blank_text_is_safe(self):
        assert ingest.extract_from_text("") == []


class TestConflicts:
    def test_untouched_field_is_new(self):
        rows = ingest.describe_conflicts(Profile(mortgage_balance=0),
                                         [Finding("mortgage_balance", 400000, 0.9, "s")])
        assert rows[0]["status"] == "new"

    def test_matching_value_is_not_a_conflict(self):
        rows = ingest.describe_conflicts(Profile(cash=50000),
                                         [Finding("cash", 50000, 0.9, "s")])
        assert rows[0]["status"] == "same"

    def test_differing_value_is_a_conflict(self):
        rows = ingest.describe_conflicts(Profile(cash=50000),
                                         [Finding("cash", 12450, 0.9, "s")])
        assert rows[0]["status"] == "conflict"
        assert rows[0]["delta"] == pytest.approx(-37550)

    def test_rates_are_compared_in_their_own_units(self):
        """A dollar-sized tolerance would call every rate a match."""
        rows = ingest.describe_conflicts(Profile(credit_card_rate=0.18),
                                         [Finding("credit_card_rate", 0.2299, 0.8, "s")])
        assert rows[0]["status"] == "conflict"

    def test_rounding_noise_is_not_a_conflict(self):
        rows = ingest.describe_conflicts(Profile(credit_card_rate=0.22),
                                         [Finding("credit_card_rate", 0.2201, 0.8, "s")])
        assert rows[0]["status"] == "same"


class TestApplying:
    def test_nothing_is_applied_without_explicit_acceptance(self):
        before = Profile(cash=5000)
        after = ingest.apply_findings(before, [Finding("cash", 99999, 0.9, "s")])
        assert after.cash == 5000

    def test_accepted_fields_are_applied(self):
        after = ingest.apply_findings(Profile(cash=5000),
                                      [Finding("cash", 12450, 0.9, "s")], {"cash"})
        assert after.cash == 12450

    def test_unaccepted_fields_are_left_alone(self):
        findings = [Finding("cash", 12450, 0.9, "s"), Finding("roth_balance", 88200, 0.9, "s")]
        after = ingest.apply_findings(Profile(cash=5000, roth_balance=1000), findings, {"cash"})
        assert after.cash == 12450
        assert after.roth_balance == 1000

    def test_does_not_mutate_the_original(self):
        before = Profile(cash=5000)
        ingest.apply_findings(before, [Finding("cash", 12450, 0.9, "s")], {"cash"})
        assert before.cash == 5000

    def test_unknown_accepted_fields_are_rejected(self):
        with pytest.raises(ValueError, match="Unsupported"):
            ingest.apply_findings(Profile(), [Finding("not_a_field", 1, 0.9, "s")],
                                   {"not_a_field"})

    def test_income_is_reconciled_after_applying(self):
        after = ingest.apply_findings(Profile(salary=100000, bonus=0, stock_comp=0),
                                      [Finding("bonus", 25000, 0.9, "s")], {"bonus"})
        assert after.gross_income >= 125000


class TestPresentation:
    def test_labels_are_human_readable(self):
        assert Finding("credit_card_rate", 0.22, 0.8, "s").label == "Credit card APR"
        assert "_" not in Finding("traditional_401k", 1, 0.8, "s").label

    def test_rate_fields_are_flagged(self):
        assert Finding("mortgage_rate", 0.06, 0.9, "s").is_rate
        assert not Finding("cash", 1000, 0.9, "s").is_rate

    def test_confidence_has_a_word_form(self):
        assert Finding("cash", 1, 0.9, "s").confidence_label == "high"
        assert Finding("cash", 1, 0.6, "s").confidence_label == "medium"
        assert Finding("cash", 1, 0.3, "s").confidence_label == "low"

    def test_serialises_for_the_api(self):
        payload = Finding("cash", 1000, 0.9, "chase.csv", "line").to_dict()
        assert payload["label"] == "Cash & savings"
        assert payload["source"] == "chase.csv"


class TestOcr:
    def test_availability_check_does_not_raise(self):
        assert isinstance(ingest.ocr_available(), bool)

    def test_missing_engine_gives_actionable_error(self, monkeypatch):
        monkeypatch.setattr(ingest, "_tesseract_path", lambda: None)
        monkeypatch.setattr(ingest, "_has_pytesseract", lambda: False)
        with pytest.raises(RuntimeError, match="tesseract"):
            ingest.ocr_image(b"not an image")

    @pytest.mark.skipif(not ingest.ocr_available(), reason="no OCR engine installed")
    def test_reads_a_rendered_screenshot(self):
        pil = pytest.importorskip("PIL")
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGB", (560, 150), "white")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
        except Exception:  # pragma: no cover - font availability varies
            font = ImageFont.load_default()
        for i, line in enumerate(["Chase Checking $12,450.10", "Roth IRA $88,200.00",
                                  "Credit Card APR 22.99%"]):
            draw.text((14, 14 + i * 42), line, fill="black", font=font)

        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        found, text = ingest.extract_from_image(buffer.getvalue(), "shot.png")

        assert "Roth" in text
        got = fields(found)
        assert got["roth_balance"] == pytest.approx(88200.0)
        assert got["credit_card_rate"] == pytest.approx(0.2299)
        assert pil is not None
