"""Location-driven auto-fill — so the user types a place, not a spreadsheet.

The design goal is that entering *salary + location* is enough to produce a
usable plan. Everything that can be derived from a location is derived here:
property tax rates, state income tax, typical rents, insurance premiums and
the local price-to-rent ratio.

Two tiers, in order of preference:

1. **Bundled reference data** (this module). Always available, works offline,
   deterministic, and therefore testable. Sourced from Tax Foundation /
   Census ACS / ATTOM style aggregates; every value carries an ``as_of`` year.
2. **Live providers** (:mod:`finrec.providers`). Optional network lookups for
   things that genuinely move week to week, e.g. mortgage rates. They are
   cached and fail soft — a network outage degrades to tier 1, never to an
   error.

Nothing here is authoritative for filing taxes. It exists to remove blank
form fields, and every auto-filled value stays editable by the user.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "LocationData",
    "STATE_NAMES",
    "STATE_PROPERTY_TAX",
    "STATE_INSURANCE_ANNUAL",
    "STATE_PRICE_TO_RENT",
    "STATE_MEDIAN_HOME_PRICE",
    "METROS",
    "resolve_location",
    "lookup_location",
    "estimate_rent",
    "estimate_home_price",
    "autofill_fields",
]

REFERENCE_YEAR = 2025

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming",
}

# Effective property tax as a share of market value (owner-occupied), by state.
# These are *effective* rates, not statutory millage -- the distinction matters
# enormously in states with assessment caps (CA Prop 13) or homestead
# exemptions (FL, TX), where the statutory rate badly overstates what a typical
# owner actually pays.
STATE_PROPERTY_TAX: dict[str, float] = {
    "AL": 0.0041, "AK": 0.0104, "AZ": 0.0063, "AR": 0.0062, "CA": 0.0071,
    "CO": 0.0051, "CT": 0.0179, "DE": 0.0058, "DC": 0.0062, "FL": 0.0079,
    "GA": 0.0090, "HI": 0.0029, "ID": 0.0056, "IL": 0.0195, "IN": 0.0079,
    "IA": 0.0143, "KS": 0.0128, "KY": 0.0080, "LA": 0.0056, "ME": 0.0117,
    "MD": 0.0099, "MA": 0.0110, "MI": 0.0126, "MN": 0.0102, "MS": 0.0075,
    "MO": 0.0091, "MT": 0.0069, "NE": 0.0146, "NV": 0.0048, "NH": 0.0177,
    "NJ": 0.0221, "NM": 0.0067, "NY": 0.0154, "NC": 0.0070, "ND": 0.0093,
    "OH": 0.0141, "OK": 0.0083, "OR": 0.0089, "PA": 0.0141, "RI": 0.0129,
    "SC": 0.0049, "SD": 0.0110, "TN": 0.0056, "TX": 0.0147, "UT": 0.0055,
    "VT": 0.0166, "VA": 0.0080, "WA": 0.0083, "WV": 0.0056, "WI": 0.0138,
    "WY": 0.0055,
}

# Average residential electricity price, $/kWh. This is what decides whether
# solar pays: the same panels are a fine investment in California and a poor
# one in Washington, purely because of what the utility charges.
STATE_ELECTRICITY_RATE: dict[str, float] = {
    "AL": 0.16, "AK": 0.25, "AZ": 0.15, "AR": 0.13, "CA": 0.32,
    "CO": 0.15, "CT": 0.29, "DE": 0.16, "DC": 0.17, "FL": 0.16,
    "GA": 0.14, "HI": 0.43, "ID": 0.11, "IL": 0.16, "IN": 0.15,
    "IA": 0.14, "KS": 0.14, "KY": 0.13, "LA": 0.13, "ME": 0.25,
    "MD": 0.18, "MA": 0.31, "MI": 0.19, "MN": 0.15, "MS": 0.14,
    "MO": 0.13, "MT": 0.13, "NE": 0.12, "NV": 0.17, "NH": 0.27,
    "NJ": 0.20, "NM": 0.15, "NY": 0.25, "NC": 0.14, "ND": 0.11,
    "OH": 0.16, "OK": 0.12, "OR": 0.13, "PA": 0.19, "RI": 0.29,
    "SC": 0.15, "SD": 0.13, "TN": 0.13, "TX": 0.15, "UT": 0.11,
    "VT": 0.22, "VA": 0.15, "WA": 0.11, "WV": 0.15, "WI": 0.17,
    "WY": 0.12,
}

# Annual sunshine, expressed the way a solar model needs it: kWh produced per
# kW of installed capacity per year.
STATE_SOLAR_PRODUCTION: dict[str, float] = {
    "AZ": 1_700, "NM": 1_680, "NV": 1_650, "CA": 1_550, "TX": 1_500,
    "CO": 1_500, "UT": 1_480, "FL": 1_450, "HI": 1_500, "OK": 1_450,
    "KS": 1_420, "GA": 1_380, "SC": 1_380, "NC": 1_350, "AL": 1_350,
    "MS": 1_350, "AR": 1_330, "TN": 1_300, "LA": 1_330, "MO": 1_320,
    "VA": 1_300, "MD": 1_280, "DE": 1_280, "DC": 1_280, "NJ": 1_250,
    "ID": 1_320, "MT": 1_300, "WY": 1_450, "SD": 1_350, "ND": 1_300,
    "NE": 1_380, "IA": 1_300, "MN": 1_270, "WI": 1_230, "IL": 1_270,
    "IN": 1_240, "OH": 1_200, "KY": 1_250, "WV": 1_180, "PA": 1_200,
    "NY": 1_180, "CT": 1_220, "RI": 1_230, "MA": 1_220, "VT": 1_150,
    "NH": 1_180, "ME": 1_180, "MI": 1_180, "OR": 1_150, "WA": 1_080,
    "AK": 900,
}

# States where water is scarce enough that districts pay people to remove
# lawns. Turf only pays where those rebates and water prices exist.
WATER_STRESSED_STATES: frozenset[str] = frozenset(
    {"CA", "AZ", "NV", "NM", "UT", "CO", "TX", "OR", "ID", "KS", "OK"}
)

# Average annual homeowners insurance premium for a typical single-family home.
# Wide spread: hurricane and hail exposure dominates.
STATE_INSURANCE_ANNUAL: dict[str, float] = {
    "AL": 2_400, "AK": 1_100, "AZ": 1_700, "AR": 2_700, "CA": 1_500,
    "CO": 3_400, "CT": 1_800, "DE": 1_000, "DC": 1_400, "FL": 4_100,
    "GA": 2_300, "HI": 700, "ID": 1_200, "IL": 2_000, "IN": 1_700,
    "IA": 1_900, "KS": 3_300, "KY": 2_300, "LA": 3_400, "ME": 1_200,
    "MD": 1_400, "MA": 1_700, "MI": 1_600, "MN": 2_400, "MS": 2_900,
    "MO": 2_400, "MT": 2_100, "NE": 4_000, "NV": 1_100, "NH": 1_100,
    "NJ": 1_300, "NM": 1_900, "NY": 1_600, "NC": 1_900, "ND": 2_300,
    "OH": 1_400, "OK": 4_400, "OR": 1_000, "PA": 1_300, "RI": 1_700,
    "SC": 2_100, "SD": 2_600, "TN": 2_100, "TX": 3_900, "UT": 1_100,
    "VT": 1_000, "VA": 1_500, "WA": 1_200, "WV": 1_300, "WI": 1_300,
    "WY": 1_700,
}

# Price-to-rent ratio: home price divided by one year of rent for a comparable
# home. This is the single most predictive number in the buy-vs-rent decision,
# and it is what lets us infer a market rent from a home price alone.
STATE_PRICE_TO_RENT: dict[str, float] = {
    "AL": 13.0, "AK": 14.0, "AZ": 18.0, "AR": 12.5, "CA": 25.0,
    "CO": 22.0, "CT": 15.0, "DE": 15.0, "DC": 21.0, "FL": 17.0,
    "GA": 15.0, "HI": 30.0, "ID": 20.0, "IL": 13.5, "IN": 12.0,
    "IA": 12.0, "KS": 12.0, "KY": 12.5, "LA": 12.0, "ME": 17.0,
    "MD": 15.0, "MA": 20.0, "MI": 13.0, "MN": 15.0, "MS": 11.5,
    "MO": 13.0, "MT": 21.0, "NE": 13.5, "NV": 19.0, "NH": 17.0,
    "NJ": 16.0, "NM": 16.0, "NY": 18.0, "NC": 15.5, "ND": 12.5,
    "OH": 12.0, "OK": 12.0, "OR": 22.0, "PA": 13.0, "RI": 17.0,
    "SC": 15.5, "SD": 14.0, "TN": 15.5, "TX": 14.5, "UT": 22.0,
    "VT": 17.0, "VA": 16.0, "WA": 23.0, "WV": 11.0, "WI": 14.0,
    "WY": 16.0,
}

STATE_MEDIAN_HOME_PRICE: dict[str, float] = {
    "AL": 230_000, "AK": 375_000, "AZ": 440_000, "AR": 210_000, "CA": 790_000,
    "CO": 570_000, "CT": 400_000, "DE": 375_000, "DC": 610_000, "FL": 400_000,
    "GA": 340_000, "HI": 840_000, "ID": 460_000, "IL": 270_000, "IN": 240_000,
    "IA": 220_000, "KS": 230_000, "KY": 220_000, "LA": 210_000, "ME": 400_000,
    "MD": 400_000, "MA": 620_000, "MI": 250_000, "MN": 340_000, "MS": 180_000,
    "MO": 250_000, "MT": 470_000, "NE": 270_000, "NV": 450_000, "NH": 480_000,
    "NJ": 520_000, "NM": 320_000, "NY": 460_000, "NC": 340_000, "ND": 260_000,
    "OH": 230_000, "OK": 210_000, "OR": 500_000, "PA": 270_000, "RI": 460_000,
    "SC": 300_000, "SD": 300_000, "TN": 330_000, "TX": 310_000, "UT": 540_000,
    "VT": 400_000, "VA": 390_000, "WA": 610_000, "WV": 170_000, "WI": 300_000,
    "WY": 350_000,
}

# Rough cost-of-living index vs the national average (100), used only to seed
# a starting spending estimate the user can immediately override.
STATE_COST_INDEX: dict[str, float] = {
    "HI": 180, "DC": 145, "CA": 138, "MA": 135, "NY": 125, "WA": 115, "AK": 125,
    "OR": 113, "MD": 116, "CT": 112, "NJ": 112, "CO": 105, "NH": 110, "RI": 110,
    "VT": 115, "AZ": 102, "NV": 102, "UT": 103, "VA": 102, "DE": 102, "ME": 111,
    "FL": 102, "MT": 103, "ID": 99, "PA": 98, "IL": 95, "MN": 95, "WI": 95,
    "NC": 96, "SC": 96, "TX": 93, "GA": 92, "TN": 92, "NM": 94,
    "OH": 92, "MI": 92, "IN": 91, "MO": 90, "IA": 91, "NE": 92, "ND": 95,
    "SD": 93, "KS": 88, "KY": 93, "LA": 92, "AL": 89, "AR": 90, "MS": 87,
    "OK": 87, "WV": 88, "WY": 95,
}

# Municipal / local income tax on top of the state rate, for the handful of
# places where it is material enough to change a decision.
LOCAL_INCOME_TAX: dict[str, float] = {
    "NEW YORK, NY": 0.0388,
    "YONKERS, NY": 0.0165,
    "PHILADELPHIA, PA": 0.0375,
    "WILMINGTON, DE": 0.0125,
    "BALTIMORE, MD": 0.0320,
    "COLUMBUS, OH": 0.0250,
    "CLEVELAND, OH": 0.0250,
    "CINCINNATI, OH": 0.0180,
    "DETROIT, MI": 0.0240,
    "KANSAS CITY, MO": 0.0100,
    "ST. LOUIS, MO": 0.0100,
    "SAN FRANCISCO, CA": 0.0,
    "PORTLAND, OR": 0.0,
}


@dataclass(frozen=True)
class Metro:
    """Metro-level overrides. Metro variation dwarfs state variation."""

    name: str
    state: str
    property_tax_rate: float
    price_to_rent: float
    median_home_price: float
    insurance_annual: float
    cost_index: float
    aliases: tuple[str, ...] = ()


METROS: tuple[Metro, ...] = (
    Metro("San Francisco", "CA", 0.0074, 28.0, 1_300_000, 1_600, 175, ("sf", "san francisco bay area")),
    Metro("San Jose", "CA", 0.0073, 30.0, 1_600_000, 1_700, 180, ("silicon valley", "santa clara")),
    Metro("Los Angeles", "CA", 0.0073, 26.0, 950_000, 1_700, 150, ("la", "socal")),
    Metro("San Diego", "CA", 0.0068, 26.0, 950_000, 1_600, 145, ()),
    Metro("Sacramento", "CA", 0.0081, 21.0, 580_000, 1_500, 118, ()),
    Metro("Seattle", "WA", 0.0088, 25.0, 850_000, 1_400, 130, ()),
    Metro("Portland", "OR", 0.0099, 22.0, 550_000, 1_100, 120, ()),
    Metro("Denver", "CO", 0.0051, 22.0, 590_000, 3_300, 112, ()),
    Metro("Austin", "TX", 0.0180, 18.0, 450_000, 3_400, 103, ()),
    Metro("Dallas", "TX", 0.0180, 15.5, 400_000, 4_100, 101, ("dfw", "fort worth")),
    Metro("Houston", "TX", 0.0190, 14.5, 340_000, 4_400, 97, ()),
    Metro("San Antonio", "TX", 0.0195, 14.5, 300_000, 3_800, 94, ()),
    Metro("New York", "NY", 0.0141, 22.0, 780_000, 1_900, 155, ("nyc", "manhattan", "brooklyn")),
    Metro("Boston", "MA", 0.0104, 21.0, 720_000, 1_800, 145, ()),
    Metro("Washington", "DC", 0.0062, 21.0, 620_000, 1_500, 145, ("dc", "washington dc")),
    Metro("Chicago", "IL", 0.0215, 13.5, 340_000, 2_200, 105, ()),
    Metro("Miami", "FL", 0.0091, 19.0, 580_000, 6_500, 115, ()),
    Metro("Tampa", "FL", 0.0089, 17.5, 400_000, 5_200, 103, ()),
    Metro("Orlando", "FL", 0.0090, 17.0, 400_000, 4_600, 102, ()),
    Metro("Atlanta", "GA", 0.0092, 16.0, 400_000, 2_500, 100, ()),
    Metro("Charlotte", "NC", 0.0075, 17.0, 400_000, 2_100, 99, ()),
    Metro("Raleigh", "NC", 0.0080, 17.5, 440_000, 2_000, 101, ("durham", "research triangle")),
    Metro("Nashville", "TN", 0.0064, 17.5, 450_000, 2_300, 101, ()),
    Metro("Phoenix", "AZ", 0.0056, 18.5, 460_000, 1_800, 104, ("scottsdale", "tempe")),
    Metro("Las Vegas", "NV", 0.0050, 19.0, 460_000, 1_200, 103, ()),
    Metro("Salt Lake City", "UT", 0.0055, 22.0, 550_000, 1_200, 105, ()),
    Metro("Minneapolis", "MN", 0.0110, 15.5, 350_000, 2_500, 100, ("st paul", "twin cities")),
    Metro("Detroit", "MI", 0.0150, 12.0, 240_000, 1_900, 93, ()),
    Metro("Philadelphia", "PA", 0.0100, 14.0, 280_000, 1_500, 102, ()),
    Metro("Pittsburgh", "PA", 0.0180, 12.0, 230_000, 1_300, 92, ()),
    Metro("Columbus", "OH", 0.0145, 13.0, 300_000, 1_500, 94, ()),
    Metro("Indianapolis", "IN", 0.0085, 12.5, 270_000, 1_700, 92, ()),
    Metro("Kansas City", "MO", 0.0125, 13.5, 290_000, 2_800, 93, ()),
    Metro("St. Louis", "MO", 0.0120, 12.5, 250_000, 2_300, 91, ()),
    Metro("Honolulu", "HI", 0.0030, 32.0, 1_050_000, 800, 185, ()),
    Metro("New Orleans", "LA", 0.0070, 13.5, 280_000, 3_800, 95, ()),
    Metro("Baltimore", "MD", 0.0110, 14.0, 350_000, 1_500, 105, ()),
    Metro("Richmond", "VA", 0.0085, 16.0, 370_000, 1_500, 99, ()),
    Metro("Boise", "ID", 0.0060, 21.0, 520_000, 1_300, 102, ()),
    Metro("Providence", "RI", 0.0130, 17.0, 460_000, 1_800, 108, ()),
)

# First-three ZIP digits -> metro name. Enough to recognise the metros above
# without shipping a 40k-row ZIP database.
ZIP3_TO_METRO: dict[str, str] = {
    "940": "San Francisco", "941": "San Francisco", "944": "San Francisco",
    "945": "San Francisco", "946": "San Francisco", "947": "San Francisco",
    "950": "San Jose", "951": "San Jose",
    "900": "Los Angeles", "901": "Los Angeles", "902": "Los Angeles",
    "903": "Los Angeles", "904": "Los Angeles", "905": "Los Angeles",
    "906": "Los Angeles", "907": "Los Angeles", "908": "Los Angeles",
    "910": "Los Angeles", "911": "Los Angeles", "912": "Los Angeles",
    "913": "Los Angeles", "914": "Los Angeles", "915": "Los Angeles",
    "916": "Sacramento", "917": "Los Angeles", "918": "Los Angeles",
    "919": "San Diego", "920": "San Diego", "921": "San Diego",
    "956": "Sacramento", "957": "Sacramento", "958": "Sacramento",
    "980": "Seattle", "981": "Seattle", "982": "Seattle", "984": "Seattle",
    "970": "Portland", "971": "Portland", "972": "Portland",
    "800": "Denver", "801": "Denver", "802": "Denver", "803": "Denver",
    "787": "Austin", "786": "Austin",
    "750": "Dallas", "751": "Dallas", "752": "Dallas", "753": "Dallas",
    "761": "Dallas", "762": "Dallas",
    "770": "Houston", "772": "Houston", "773": "Houston", "774": "Houston",
    "782": "San Antonio", "781": "San Antonio",
    "100": "New York", "101": "New York", "102": "New York", "103": "New York",
    "104": "New York", "110": "New York", "111": "New York", "112": "New York",
    "113": "New York", "114": "New York", "116": "New York",
    "021": "Boston", "022": "Boston", "023": "Boston", "024": "Boston",
    "200": "Washington", "202": "Washington", "203": "Washington",
    "204": "Washington", "205": "Washington", "220": "Washington",
    "606": "Chicago", "600": "Chicago", "601": "Chicago", "602": "Chicago",
    "603": "Chicago", "604": "Chicago", "605": "Chicago",
    "331": "Miami", "330": "Miami", "333": "Miami",
    "336": "Tampa", "335": "Tampa",
    "328": "Orlando", "327": "Orlando",
    "303": "Atlanta", "300": "Atlanta", "301": "Atlanta", "302": "Atlanta",
    "282": "Charlotte", "280": "Charlotte", "281": "Charlotte",
    "276": "Raleigh", "277": "Raleigh",
    "372": "Nashville", "370": "Nashville", "371": "Nashville",
    "850": "Phoenix", "852": "Phoenix", "853": "Phoenix",
    "891": "Las Vegas", "889": "Las Vegas",
    "841": "Salt Lake City", "840": "Salt Lake City",
    "554": "Minneapolis", "553": "Minneapolis", "551": "Minneapolis",
    "482": "Detroit", "481": "Detroit", "480": "Detroit",
    "191": "Philadelphia", "190": "Philadelphia",
    "152": "Pittsburgh", "150": "Pittsburgh",
    "432": "Columbus", "430": "Columbus",
    "462": "Indianapolis", "460": "Indianapolis",
    "641": "Kansas City", "640": "Kansas City",
    "631": "St. Louis", "630": "St. Louis",
    "968": "Honolulu", "967": "Honolulu",
    "701": "New Orleans", "700": "New Orleans",
    "212": "Baltimore", "210": "Baltimore", "211": "Baltimore",
    "232": "Richmond", "230": "Richmond",
    "837": "Boise",
    "029": "Providence",
}

_METRO_BY_NAME = {m.name.lower(): m for m in METROS}
for _m in METROS:
    for _alias in _m.aliases:
        _METRO_BY_NAME[_alias.lower()] = _m

_STATE_BY_NAME = {name.lower(): code for code, name in STATE_NAMES.items()}


@dataclass
class LocationData:
    """Everything we can infer about money from a place name."""

    query: str
    state: str
    state_name: str = ""
    metro: str | None = None
    zip_code: str | None = None

    property_tax_rate: float = 0.011
    state_income_tax_rate: float = 0.0
    local_income_tax_rate: float = 0.0
    price_to_rent: float = 16.0
    median_home_price: float = 400_000
    insurance_annual: float = 1_800
    cost_index: float = 100.0
    home_appreciation: float = 0.035

    confidence: str = "state"          # "metro" | "state" | "default"
    sources: list[str] = field(default_factory=list)
    as_of: int = REFERENCE_YEAR

    @property
    def label(self) -> str:
        if self.metro:
            return f"{self.metro}, {self.state}"
        return self.state_name or self.state

    def to_dict(self) -> dict:
        return {
            "query": self.query, "state": self.state, "state_name": self.state_name,
            "metro": self.metro, "zip_code": self.zip_code, "label": self.label,
            "property_tax_rate": self.property_tax_rate,
            "state_income_tax_rate": self.state_income_tax_rate,
            "local_income_tax_rate": self.local_income_tax_rate,
            "price_to_rent": self.price_to_rent,
            "median_home_price": self.median_home_price,
            "insurance_annual": self.insurance_annual,
            "cost_index": self.cost_index,
            "home_appreciation": self.home_appreciation,
            "confidence": self.confidence, "sources": self.sources, "as_of": self.as_of,
        }


_ZIP_RE = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def resolve_location(query: str) -> tuple[str, str | None, str | None]:
    """Parse free text into ``(state_code, metro_name, zip_code)``.

    Accepts ``"CA"``, ``"California"``, ``"Austin, TX"``, ``"nyc"``, ``"94110"``.
    Unrecognised input falls back to the national default rather than raising —
    a typo in a city name should never block the rest of the plan.
    """
    text = (query or "").strip()
    if not text:
        return "", None, None

    zip_match = _ZIP_RE.search(text)
    zip_code = zip_match.group(1) if zip_match else None
    if zip_code:
        metro_name = ZIP3_TO_METRO.get(zip_code[:3])
        if metro_name:
            return _METRO_BY_NAME[metro_name.lower()].state, metro_name, zip_code

    cleaned = text.replace(zip_code, "").strip(" ,") if zip_code else text
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]

    state = ""
    city = ""
    if len(parts) >= 2:
        city, maybe_state = parts[0], parts[-1]
        state = _normalize_state(maybe_state)
    elif parts:
        state = _normalize_state(parts[0])
        if not state:
            city = parts[0]

    metro = None
    if city:
        found = _METRO_BY_NAME.get(city.lower())
        if found and (not state or found.state == state):
            metro = found.name
            state = found.state
    if not metro and not state and cleaned:
        found = _METRO_BY_NAME.get(cleaned.lower())
        if found:
            metro, state = found.name, found.state

    return state, metro, zip_code


def _normalize_state(token: str) -> str:
    token = token.strip()
    if len(token) == 2 and token.upper() in STATE_NAMES:
        return token.upper()
    return _STATE_BY_NAME.get(token.lower(), "")


def lookup_location(query: str, *, live: bool = False,
                    taxable_income: float = 0.0, year: int | None = None) -> LocationData:
    """Resolve a place to its financial characteristics.

    ``live=True`` additionally consults :mod:`finrec.providers` for values that
    change frequently. It never raises on network failure.
    """
    from .taxes import state_rate_for_income

    state, metro_name, zip_code = resolve_location(query)
    sources: list[str] = []

    if not state:
        data = LocationData(
            query=query, state="", state_name="United States (average)",
            confidence="default",
            sources=["National averages — enter a state or city for local figures."],
        )
        if live:
            _apply_live(data)
        return data

    data = LocationData(
        query=query,
        state=state,
        state_name=STATE_NAMES.get(state, state),
        zip_code=zip_code,
        property_tax_rate=STATE_PROPERTY_TAX.get(state, 0.011),
        state_income_tax_rate=state_rate_for_income(state, taxable_income, year),
        price_to_rent=STATE_PRICE_TO_RENT.get(state, 16.0),
        median_home_price=STATE_MEDIAN_HOME_PRICE.get(state, 400_000),
        insurance_annual=STATE_INSURANCE_ANNUAL.get(state, 1_800),
        cost_index=STATE_COST_INDEX.get(state, 100.0),
        confidence="state",
    )
    sources.append(f"{data.state_name} statewide averages ({REFERENCE_YEAR})")
    sources.append("State/local income tax is a coarse planning estimate, not a complete tax schedule.")

    if metro_name:
        metro = _METRO_BY_NAME[metro_name.lower()]
        data.metro = metro.name
        data.property_tax_rate = metro.property_tax_rate
        data.price_to_rent = metro.price_to_rent
        data.median_home_price = metro.median_home_price
        data.insurance_annual = metro.insurance_annual
        data.cost_index = metro.cost_index
        data.confidence = "metro"
        sources.insert(0, f"{metro.name} metro averages ({REFERENCE_YEAR})")

    key = f"{(metro_name or '').upper()}, {state}"
    data.local_income_tax_rate = LOCAL_INCOME_TAX.get(key, 0.0)
    if data.local_income_tax_rate:
        sources.append(f"{metro_name} local income tax {data.local_income_tax_rate:.2%}")

    # High price-to-rent markets have historically appreciated faster, but the
    # long-run national anchor is inflation + ~1%. Blend rather than extrapolate.
    data.home_appreciation = round(0.030 + min(0.015, max(0.0, (data.price_to_rent - 15) * 0.0009)), 4)

    data.sources = sources
    if live:
        _apply_live(data)
    return data


def _apply_live(data: LocationData) -> None:
    """Best-effort live enrichment. Silently keeps bundled values on failure."""
    try:
        from . import providers

        rate = providers.current_mortgage_rate()
        if rate:
            data.sources.append(f"Live 30-yr mortgage rate {rate:.2%} (FRED MORTGAGE30US)")
    except Exception:  # pragma: no cover - network paths are not unit tested
        pass


def estimate_rent(home_price: float, location: LocationData | str) -> float:
    """Monthly rent for a home of this price in this market.

    Uses the local price-to-rent ratio, which is why the user only has to give
    us a price. A $1M home in San Jose (P/R 30) rents for far less than a $1M
    home in Houston (P/R 14.5), and that difference *is* the buy-vs-rent answer.
    """
    loc = location if isinstance(location, LocationData) else lookup_location(location)
    ratio = max(5.0, loc.price_to_rent)
    return home_price / ratio / 12.0


def estimate_home_price(monthly_rent: float, location: LocationData | str) -> float:
    """Inverse of :func:`estimate_rent` — what your rent would buy."""
    loc = location if isinstance(location, LocationData) else lookup_location(location)
    return monthly_rent * 12.0 * max(5.0, loc.price_to_rent)


def autofill_fields(query: str, *, live: bool = False) -> dict:
    """Profile fields we can fill in from a location alone.

    Keys map directly onto :class:`finrec.profile.Profile` attributes so the UI
    can apply them without a translation table.
    """
    loc = lookup_location(query, live=live)
    return {
        "state": loc.state or "",
        "property_tax_rate": loc.property_tax_rate,
        "home_insurance_annual": loc.insurance_annual,
        "local_income_tax_rate": loc.local_income_tax_rate,
        "home_appreciation": loc.home_appreciation,
        "_meta": loc.to_dict(),
    }
