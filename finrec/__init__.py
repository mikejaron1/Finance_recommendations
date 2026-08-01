"""finrec — a personal financial planning engine.

Modules
-------
core         Time-value-of-money primitives (pmt/fv/npv/irr/xirr).
taxes        Federal + state tax engine with bracket data by year and status.
mortgage     Amortization, PMI, extra payments, refinance break-even.
montecarlo   Stochastic return simulation and sequence-of-returns risk.
housing      Buy vs rent, investment property, sell/keep/rent-out, affordability.
retirement   Roth vs Traditional, drawdown planning, contribution waterfall.
budget       Spend analysis, savings opportunities, emergency fund sizing.
advisors     Robo-advisor / brokerage fee comparison and cash vehicles.
projects     Solar, turf and renovation ROI.
portfolio    Read-only holdings tracking with env-only credentials.
profile      The shared Profile input object.
recommend    Ranked recommendations and a financial health score.

All figures are estimates for planning purposes, not tax or investment advice.
"""

from .profile import Profile, DEFAULT_PROFILE

__version__ = "2.0.0"
__all__ = ["Profile", "DEFAULT_PROFILE", "__version__"]
