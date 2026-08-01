# Finance Recommendations

**An interactive personal-finance dashboard that does what a fee-only financial planner would do — and shows its work.**

Every question from the original idea list is now a working, tested model behind a Streamlit UI. No black boxes: each module documents its assumptions and, where it differs from the naive approach, explains why.

```bash
pip install -r requirements.txt
streamlit run app/Home.py
```

Then open http://localhost:8501.

---

## What it answers

| Question | Where |
|---|---|
| Wealthfront vs Schwab vs DIY? | 📈 Investing → Provider fees |
| 401k vs Roth? How much in each? | 🎯 Retirement → Roth vs Traditional, Contribution waterfall |
| Solar? Turf? Renovations? | 🔨 Home Projects |
| Investment property? | 🏘️ Investment Property |
| How much cash should I hold? | 🧾 Spending & Cash → Emergency fund |
| How much do I spend, and what's easiest to cut? | 🧾 Spending & Cash |
| What impact would cutting it have? | 🧾 Spending & Cash → What to cut |
| Best place to park savings? | 🧾 Spending & Cash → Cash vehicles |
| Buy a home vs invest the money? | 🏡 Buy vs Rent |
| Sell the home, keep it, or rent it out? | 🏘️ Investment Property → Sell/keep/rent |
| Should I refinance or pay extra? | 🏦 Mortgage |
| What's my tax bill really? | 🧮 Taxes |
| What should I do *first*? | 🏠 Home → ranked recommendations |

---

## How it works

You fill in **one profile** (👤 Your Profile). Every page reads from it, so changing your income or savings rate updates the whole dashboard. The Home page ranks concrete actions by priority and estimated lifetime impact, and scores your finances 0–100 across six weighted dimensions.

### Layout

```
finrec/                 Pure-Python financial engine (no Streamlit imports)
  core.py               TVM primitives: pmt, fv, pv, nper, npv, irr, xirr, rate conversions
  taxes.py              2024/2025 brackets × 4 filing statuses, FICA, LTCG, NIIT, SALT cap
  mortgage.py           Amortization with PMI drop-off, extra payments, refinance break-even
  montecarlo.py         Return simulation, wealth paths, drawdown, safe withdrawal rate
  housing.py            Buy vs rent, rental underwriting, sell/keep/rent, affordability
  retirement.py         Roth vs Traditional, contribution waterfall, tax-aware drawdown
  budget.py             Transaction categorisation, savings opportunities, emergency fund
  advisors.py           Provider fee comparison, savings-vehicle after-tax yields
  projects.py           Solar, turf, renovation ROI
  portfolio.py          Read-only holdings tracker, XIRR, concentration, rebalancing
  profile.py            The single shared Profile object
  recommend.py          Ranked recommendations, health score, net-worth projection

app/                    Streamlit dashboard
  Home.py               Entry point
  _shared.py            Profile state, formatting, chart helpers
  pages/                One page per topic

tests/                  Unit tests plus a suite that renders every dashboard page
scripts/                Sample-data generator
data/                   sample_transactions.csv (synthetic; real data is gitignored)
```

`finrec/` has no Streamlit dependency, so you can import it from a notebook, a script, or another app.

---

## What was corrected

This replaces a Jupyter notebook that no longer ran. Beyond fixing it, every model was reviewed. The substantive corrections:

**Broken outright**
- `np.pmt` / `np.fv` / `np.irr` were removed from NumPy 1.20. Reimplemented in `core.py` against Excel's conventions.
- The `mortgage` and `py_mortgage` packages are unmaintained. Replaced with a first-party amortization engine.

**Wrong maths**
- **Refinance balance** was computed as `loan − principal_paid − interest_paid`, subtracting interest from principal. Now pulled from the actual amortization schedule.
- **Deduction savings** used hand-rolled bracket-straddling logic that mishandled the boundary. Now computes tax twice and differences — simple and exactly correct across any number of brackets.
- **Rate conversion** used `rate / 12` everywhere. A 7% annual rate compounded monthly that way is really 7.23%. Investment growth now uses a true geometric conversion; mortgages deliberately keep `rate/12`, because that *is* the lender convention.
- **Real returns** used `nominal − inflation`. Now the exact Fisher relation.
- **Capital gains** were a flat 15%. Now the real LTCG brackets, stacked on top of ordinary income.

**Wrong model**
- **Fixed 7% returns.** The single biggest flaw: it produced one confident number. Replaced with Monte Carlo across three return models (lognormal, Student-t for fat tails, historical bootstrap), reported as percentile bands.
- **Retirement drawdown** subtracted spending pre-tax, understating how much you must actually withdraw. Now grosses up through the bracket schedule, withdraws at the *start* of each year to capture sequence-of-returns risk, applies RMDs, and offers Guyton-Klinger guardrails.
- **Roth vs Traditional** contributed equal *nominal* dollars to each, which isn't a fair comparison — $23,500 into a Roth costs more take-home than $23,500 into a Traditional. Two explicit comparison bases are now offered, and the result is reported as a break-even future tax rate.
- **Mortgage interest and property tax** were treated as fully deductible. Post-2018 the $10k SALT cap and $750k mortgage-debt limit mean most high earners get no benefit at all — the main reason naive buy-vs-rent models overstate buying.
- **Emergency fund** was a flat 6 months. Now sized from actual risk factors: single vs dual income, job stability, self-employment, dependents, disability coverage.
- **Projections** were quoted in nominal dollars. Net-worth projections are now in today's purchasing power.

**Bugs found while testing this rebuild**
- In buy-vs-rent, only the *renter* invested their monthly surplus. Once rent inflation pushed renting above the cost of owning, the owner's surplus silently vanished from the ledger — a permanent thumb on the scale against buying. Both sides now invest symmetrically.
- Rental underwriting charged year-1 property tax and maintenance against an already-appreciated value, which made the reported break-even rent inconsistent with the cashflow table. Expenses now accrue against the start-of-year value.
- Roth vs Traditional seeded each scenario with only its own existing balance, so the "winner" depended on which pot you already happened to have rather than on the contribution decision. Existing balances now appear in both scenarios.
- The reported marginal rate was federal-only while the effective rate was all-in, so a high earner in California could see a marginal rate *below* their effective rate. Marginal is now federal + state + payroll on the next dollar.

Each of these is locked in by a regression test.

---

## Security

The old notebook had live Coinbase API keys and a passphrase pasted into a cell. Those were never committed to git (verified with `git log --all -S`), and the notebook is gone — **but the keys sat in plaintext on disk, so rotate them.**

The rebuild's posture:

- **Credentials come from environment variables only.** Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.
- **Nothing is hardcoded and nothing is logged.** `portfolio.credential_status()` shows only a masked fingerprint so you can confirm a key is loaded without ever displaying it.
- **API access is read-only.** The tracker reads balances and public prices. There is no trading code path.
- **Your financial data stays local.** `data/*.csv` is gitignored except `sample_*.csv`. Nothing is uploaded anywhere.
- **The dashboard binds to localhost** and has no authentication — don't expose the port publicly.

---

## Testing

```bash
python3 -m pytest                            # everything
python3 -m pytest tests/test_dashboard.py    # renders every page
```

The dashboard suite executes every page through Streamlit's `AppTest` harness, so a broken page fails the test run rather than surfacing when someone clicks a tab.

Regenerate the sample spending data with `python3 scripts/make_sample_data.py`.

---

## Assumptions worth checking

The models are only as good as their inputs. These are reasonable defaults, not gospel:

- Tax brackets are 2024/2025 statutory values; state rates are **flat top-rate approximations**, not full state bracket schedules.
- Renovation cost-recouped rates are national averages and vary enormously by region.
- Savings-vehicle APYs and solar production figures are illustrative, not live data.
- Expected returns and volatility are long-run assumptions you can change on any page.

## Disclaimer

This is a personal modelling tool, not financial, tax, or legal advice. Verify anything material with a professional before acting on it.
