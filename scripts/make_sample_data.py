"""Generate data/sample_transactions.csv — synthetic but realistic spending data.

Run: python3 scripts/make_sample_data.py
The output is committed so the dashboard's "use sample data" option works out of the box.
No real financial data is used or stored.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(20240617)

OUT = Path(__file__).resolve().parent.parent / "data" / "sample_transactions.csv"

# (description, typical amount, monthly frequency, jitter)
RECURRING = [
    ("WELLS FARGO HM MORTGAGE PMT", 4298.06, 1, 0.0),
    ("HOA DUES - OAKRIDGE HOMEOWNERS", 285.00, 1, 0.0),
    ("PG&E ENERGY STATEMENT", 210.00, 1, 0.35),
    ("EBMUD WATER SERVICE", 96.00, 1, 0.30),
    ("COMCAST XFINITY INTERNET", 89.99, 1, 0.0),
    ("VERIZON WIRELESS AUTOPAY", 142.30, 1, 0.05),
    ("GEICO AUTO INSURANCE", 168.00, 1, 0.0),
    ("NETFLIX.COM", 22.99, 1, 0.0),
    ("SPOTIFY USA", 19.99, 1, 0.0),
    ("HULU + DISNEY BUNDLE", 18.99, 1, 0.0),
    ("HBO MAX SUBSCRIPTION", 16.99, 1, 0.0),
    ("APPLE.COM/BILL ICLOUD", 9.99, 1, 0.0),
    ("YOUTUBE PREMIUM", 13.99, 1, 0.0),
    ("ADOBE CREATIVE CLOUD", 59.99, 1, 0.0),
    ("EQUINOX FITNESS SF", 215.00, 1, 0.0),
    ("CLASSPASS INC", 79.00, 1, 0.0),
    ("BRIGHT HORIZONS PRESCHOOL", 1850.00, 1, 0.0),
    ("CHEWY.COM PET SUPPLIES", 74.50, 1, 0.4),
    ("TRANSFER TO SAVINGS - VANGUARD", 2500.00, 1, 0.0),
    ("NELNET STUDENT LOAN PMT", 412.00, 1, 0.0),
    ("TOYOTA FINANCIAL AUTO LOAN", 528.00, 1, 0.0),
]

FREQUENT = [
    ("SAFEWAY #1842", 128.00, 5, 0.5),
    ("TRADER JOES #221", 74.00, 3, 0.45),
    ("COSTCO WHOLESALE #1103", 246.00, 1, 0.5),
    ("WHOLE FOODS MARKET", 92.00, 2, 0.5),
    ("DOORDASH*ORDER", 46.00, 6, 0.55),
    ("UBEREATS", 39.00, 3, 0.5),
    ("STARBUCKS STORE 09122", 7.85, 12, 0.35),
    ("PHILZ COFFEE", 6.40, 4, 0.3),
    ("BLUE BOTTLE COFFEE", 8.20, 3, 0.3),
    ("CHIPOTLE 2244", 17.50, 3, 0.3),
    ("SUSHI RAN RESTAURANT", 118.00, 1, 0.5),
    ("BARREL HOUSE BREWERY", 62.00, 2, 0.5),
    ("NAPOLI PIZZA KITCHEN", 44.00, 2, 0.4),
    ("AMAZON.COM*MKTPLACE", 68.00, 8, 0.7),
    ("TARGET T-1189", 94.00, 2, 0.6),
    ("SHELL OIL 574812", 62.00, 4, 0.3),
    ("CHEVRON 0093471", 58.00, 2, 0.3),
    ("UBER TRIP", 26.00, 4, 0.6),
    ("LYFT RIDE", 22.00, 2, 0.6),
    ("BART CLIPPER RELOAD", 40.00, 2, 0.0),
    ("CVS/PHARMACY #4471", 38.00, 2, 0.6),
    ("PETCO ANIMAL SUPPLIES", 55.00, 1, 0.5),
    ("HOME DEPOT #6612", 145.00, 1, 0.8),
]

OCCASIONAL = [
    ("UNITED AIRLINES 0162", 640.00, 0.25, 0.6),
    ("MARRIOTT BONVOY SAN DIEGO", 480.00, 0.2, 0.5),
    ("AIRBNB * HMQ4X2", 720.00, 0.15, 0.5),
    ("EXPEDIA TRAVEL", 385.00, 0.12, 0.5),
    ("KAISER PERMANENTE MEDICAL", 220.00, 0.4, 0.6),
    ("BRIGHT SMILE DENTAL", 190.00, 0.3, 0.4),
    ("IKEA EMERYVILLE", 320.00, 0.2, 0.7),
    ("WAYFAIR ORDER", 275.00, 0.15, 0.7),
    ("ETSY.COM PURCHASE", 58.00, 0.5, 0.6),
    ("BANANA REPUBLIC SHOP", 130.00, 0.5, 0.6),
    ("FRANCHISE TAX BOARD PMT", 1400.00, 0.1, 0.0),
]

INCOME = [
    ("DIRECT DEPOSIT PAYROLL", 9800.00, 2, 0.0),
    ("DIRECT DEPOSIT PAYROLL - SPOUSE", 6200.00, 2, 0.0),
]


def jittered(amount: float, jitter: float) -> float:
    if jitter <= 0:
        return round(amount, 2)
    return round(max(1.0, amount * random.uniform(1 - jitter, 1 + jitter)), 2)


def main() -> None:
    end = date.today().replace(day=1)
    start = end - timedelta(days=365)
    start = start.replace(day=1)
    rows: list[tuple[str, str, float]] = []

    month_start = start
    while month_start < end:
        next_month = (month_start.replace(day=28) + timedelta(days=4)).replace(day=1)
        days_in_month = (next_month - month_start).days

        def add(desc: str, amount: float, count: float, jitter: float, sign: int = -1) -> None:
            whole = int(count)
            if random.random() < (count - whole):
                whole += 1
            for _ in range(whole):
                when = month_start + timedelta(days=random.randint(1, days_in_month) - 1)
                if when >= end:
                    continue
                rows.append((when.isoformat(), desc, sign * jittered(amount, jitter)))

        for desc, amt, freq, jit in RECURRING + FREQUENT + OCCASIONAL:
            add(desc, amt, freq, jit)
        for desc, amt, freq, jit in INCOME:
            add(desc, amt, freq, jit, sign=1)

        month_start = next_month

    rows.sort(key=lambda r: r[0])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Date", "Description", "Amount"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} transactions to {OUT}")


if __name__ == "__main__":
    main()
