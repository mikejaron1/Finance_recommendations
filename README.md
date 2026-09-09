# Finance Recommendations

**Personal-finance planning models with editable assumptions and explainable trade-offs.**

The Streamlit interface combines a shared household profile, dated tax rules, scenario planning and explicit cash-flow comparisons. Results are estimates, not a substitute for a financial planner or a tax return.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Then open http://localhost:8501.

**Your plan is saved automatically, and saves are versioned.** Returning sessions resume the last plan. Storage defaults to an owner-readable SQLite database at `~/.finrec/finrec.db` (override with `FINREC_HOME` or `FINREC_DB`). Data stays on the server running the application unless you export it or explicitly approve hosted AI processing. In the default localhost setup, that server is your computer.

Saving **appends a version rather than overwriting**, so the sidebar has a real undo: pick any earlier save and restore it. Restoring is itself a save, so you can undo the undo. That is also the recovery path — if the newest snapshot is ever unreadable, loading walks back through the history instead of showing you an empty profile.

**You start with two answers: where you live and what you earn.** Everything else — property tax rate, home insurance, state and local income tax, typical rents, current mortgage rates — is looked up for you and shown as an editable assumption. Every page leads with a plain-English answer; the inputs and the detail sit behind *Advanced*.

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
| I'm moving — keep the house and rent it, or sell and invest? | 🏘️ Investment Property → Keep or sell |
| Should I refinance or pay extra? | 🏦 Mortgage |
| What's my tax bill really? | 🧮 Taxes |
| What should I do *first*? | 🏠 Home → ranked recommendations |
| What rent makes buying worth it? | 🏡 Buy vs Rent → break-even rent |
| We're moving, the kids cost more every year, and I want a bigger house — does the plan still work? | 🔮 Scenarios |
| Suppose the business turns a profit in three years, or I get the promotion — what then? | 🔮 Scenarios → Is your income changing? |
| Which child's 529 is behind? | 👤 My Profile → College savings |
| Should we buy the next house or rent it, given everything else we're doing? | 🔮 Scenarios → break-even rent |
| We own a rental — are we better off keeping it or selling and investing the proceeds? | 🔮 Scenarios → other property |
| What if the market returns less than I've been assuming? | 🔮 Scenarios → change the assumptions |
| My wife's business lost money this year — should I contribute to Roth instead? | 🎯 Retirement → Roth vs Traditional |
| One of us is W-2 and one is self-employed — what do we actually owe? | 👤 Profile → Deductions & taxable income |

---

## How it works

You fill in **one profile** (👤 Your Profile). Every page reads from it, so changing your income or savings rate updates the whole dashboard. The Home page ranks concrete actions by priority and estimated lifetime impact, and scores your finances 0–100 across six weighted dimensions.

You can **tick actions off** as you do them; they move to a Done list so you can still see what you've handled. Completed items are matched on a slug that ignores the dollar figures in the title, so a raise changes the amount without resurrecting an action you already finished.

The profile also records **what you save each year** — 401k, Roth, HSA, 529 and taxable — plus your employer match terms and HDHP coverage tier. That turns vague advice into arithmetic: the app can tell you the exact dollars of employer match you're leaving on the table, how much HSA room is left, and whether your total savings rate gets you to retirement on time. (HSA limits key off *coverage tier*, not filing status — a married couple on a self-only plan gets the lower limit — and the catch-up starts at 55, not 50.)

### The minimum you have to type

Onboarding asks for a location and your compensation, split into **salary, bonus and stock** to distinguish reliable and variable pay. RSU vesting is compensation income; selling shares creates a separate gain or loss relative to their cost basis. The provider comparison asks for actual realized gains rather than treating vesting proceeds as capital gains.

From the location alone, `finrec/lookup.py` fills in the effective property tax rate, average home insurance premium, state and local income tax rates, the local price-to-rent ratio, a median home price and a cost-of-living index for all 50 states, DC and ~40 metros. `finrec/providers.py` optionally fetches the current 30- and 15-year mortgage rate from FRED (no API key, cached on disk, falls back silently if offline).

Nothing is hidden: every derived number appears in an **Assumptions** panel with its source, and every one is overridable. An auto-filled model you cannot inspect is worse than no model.

You can also skip the typing entirely. The profile page reads your own documents — a **transaction CSV** from your bank or card, or a **screenshot** of an account summary — and proposes values for the fields it recognises (`finrec/ingest.py`). Every proposal is shown with the line it came from and a confidence level, and **nothing is applied until you approve it**. Where a proposal disagrees with a number you typed, the default is to keep yours; you choose per field. Screenshots are read locally with `tesseract` (`brew install tesseract`) — financial screenshots are never sent to a third party, and the temp file is deleted immediately. CSV import works without it.

### Telling it about your life

Rules only see the numbers. A profile can't tell that you're expecting a baby, that half your pay
is RSUs in your employer's stock, or that you need a down payment in eighteen months — and those
facts change the advice more than another decimal place of return assumption would.

So the profile has a free-text box. `finrec/advisor_llm.py` reads it with a set of local keyword
rules and produces specific, actionable insights **with no API key, no network call and no data
leaving your machine**. That is the default and it is fully functional on its own.

If you want a language model to add to it, put a key in a git-ignored `.env` at the project root:

```
GEMINI_API_KEY=...        # Google AI Studio; preferred when both provider keys are present
OPENAI_API_KEY=...        # also supported — used only if no Gemini key is present
```

Gemini's Flash-Lite tier is the default, reached through Google's OpenAI-compatible endpoint so
both providers share one request path. The model name is the unversioned `gemini-flash-lite-latest`
alias on purpose: the pinned names this was first written against were both retired mid-development
and started returning 404, which would have silently disabled the feature.

Even with a key, opening the preview, checking consent or saving the profile sends nothing.
Review both request payloads, give consent, then click **Send reviewed requests and remember results**.
Structured balances use bands, and numeric amounts in notes and recommendation text are redacted.
Narrative can still identify people, so review it carefully. Failures are surfaced while local advice
remains available. Saved hosted results are scoped to the plan and fingerprinted against the profile
and requests; stale results are not silently reused.

### Your inputs stay put

Streamlit discards the state of any widget that wasn't on the most recent run, so by default every
number you type on a page is gone the moment you open another one. Each input is therefore mirrored
into a plain session key and into the `user_state` table, which means a home price, a slider or a
dropdown is still where you left it after navigating away, and after closing the browser entirely.

A remembered value is only reused while the caller's own default is unchanged — edit your home value
on the profile and the buy-vs-rent page picks the new number up rather than sitting on a stale one
you'd have no obvious way to clear.

### Simple first, then advanced

Each calculator is inverted to ask for the fewest numbers that still make the answer meaningful, and to return a number you can act on:

| Page | You enter | You get back |
|---|---|---|
| Buy vs Rent | A home price | The **rent below which renting wins** — go check listings against it |
| Investment property | A purchase price | The rent the deal **needs**, vs what the market actually pays |
| Taxes | Nothing | Effective and marginal rate, from your profile |
| Mortgage | A loan amount | Payment and lifetime interest, rate pre-filled from today's market |
| Retirement | Nothing | Roth vs traditional, and the future tax rate that flips it |

Asking for a home price *and* a comparable rent invites people to enter the number they hope for. Asking only for the price and solving for the break-even removes that degree of freedom.

### Finding your way around

Navigation sits in the **left sidebar**, open on desktop and collapsed on narrow screens. **Profile**, **Dashboard**
and **Scenarios** sit at the very top under no heading at all, so they are always
visible in the navigation; *Home & property* and *Money & tax* are grouped below them.
Twelve pages across a top bar wrap and truncate, and the first casualty was the
profile — the one page people need the moment a number looks wrong. It is now the
first link in the sidebar, without duplicating it beneath the menu.

The menu is built with `expanded=True`. Streamlit's default (`expanded=False`) hides
the overflow behind a "view more" button once there are enough pages, which collapses
exactly the long menu that needs to stay visible.

### Look and feel

The interface is styled as a product, not a notebook: one content column on a
tinted canvas, a page header with the icon as a badge rather than an emoji
wedged into the `<h1>`, metrics reduced to a label, a number and a hairline,
and charts stripped back to a faint horizontal grid.

The typeface is **Inter, self-hosted** from `app/static/fonts/` (latin subset,
four weights, ~24 KB each) and served through Streamlit's `enableStaticServing`.
It is deliberately *not* loaded from Google Fonts: that would send every
visitor's IP address to a third party on each page load. Self-hosting avoids that
unnecessary third-party dependency in the local-first interface.

Charts are themed from the same palette and typeface as the page, so a category
keeps its colour everywhere and no chart falls back to Plotly's defaults.

Underneath the links the sidebar carries a running snapshot (net worth, household
income, savings rate, progress to financial independence), the simple/advanced toggle,
and the saved-plans panel.

The look is one design system rather than per-page choices: colours come from
`PALETTE`, `CHART_SEQUENCE` and `EVENT_COLORS` in `app/_shared.py`, and `.streamlit/config.toml`
is pinned to the same brand colour. A test fails the build if a view hand-types a hex,
because that is how the same category ended up green on one page and red on another.

### Layout

```
finrec/                 Pure-Python financial engine (no Streamlit imports)
  core.py               TVM primitives: pmt, fv, pv, nper, npv, irr, xirr, rate conversions
  taxes.py              Effective-dated brackets, per-earner payroll, LTCG, NIIT and SALT
  mortgage.py           Amortization with PMI drop-off, extra payments, refinance break-even
  montecarlo.py         Return simulation, wealth paths, drawdown, safe withdrawal rate
  housing.py            Buy vs rent, rental underwriting, keep-vs-sell, capital gains, affordability
  retirement.py         Roth vs Traditional, contribution waterfall, tax-aware drawdown
  budget.py             Transaction categorisation, savings opportunities, emergency fund
  advisors.py           Provider fee comparison, savings-vehicle after-tax yields
  projects.py           Solar, turf, renovation ROI
  portfolio.py          Read-only holdings tracker, XIRR, concentration, rebalancing
  profile.py            The single shared Profile object
  recommend.py          Ranked recommendations, health score, net-worth projection
  scenario.py           Life events on a timeline: moving house, rising costs, big purchases —
                        one ledger in today's money, with a financial-independence target that moves
  advisor_llm.py        Turns your free-text notes into advice — local rules first, LLM optional

  lookup.py             Location -> tax rates, insurance, price-to-rent, cost index (50 states + metros)
  providers.py          Optional live data (FRED mortgage rates, market quotes), cached, fail-soft
  db.py                 SQLite: schema migrations, users, versioned snapshots, backup, health
  storage.py            Saved plans and their history, scoped per user — the only module that
                        knows where data lives
  ingest.py             Reads transaction CSVs and screenshots into *proposed* profile updates
  service.py            JSON-in/JSON-out layer: every analysis returns {simple, advanced, assumptions, meta}
  api.py                FastAPI wrapper over service.py — no business logic of its own
  actions.py            Planned/completed/snoozed/not-applicable action decisions
  analysis.py           Shared scenario comparisons and housing decision orchestration

app/                    Streamlit front end (presentation only)
  main.py               Entry point: navigation, onboarding gate, site chrome
  _shared.py            Profile state, design system, layout primitives, formatting, charts
  ui/                   Shared UI responsibilities, re-exported through _shared.py
  assets/               Brand mark (SVG)
  views/                One view per topic

tests/                  Unit tests plus a suite that renders every dashboard page
scripts/                Sample-data generator
data/                   sample_transactions.csv (synthetic; real data is gitignored)
```

`finrec/` has no Streamlit dependency, so you can import it from a notebook, a script, or another app.

### Using it as an API

The UI holds no calculations. Everything goes through `finrec/service.py`, which takes and returns plain JSON, so the Streamlit layer can be replaced with a real web front end without touching the engine.

```bash
python -m pip install -r requirements-api.txt
uvicorn finrec.api:app --host 127.0.0.1
```

```bash
curl "http://localhost:8000/api/location?q=Austin,%20TX"
curl -X POST http://localhost:8000/api/buy-vs-rent \
  -H "Content-Type: application/json" \
  -d '{"home_price": 650000, "location": "Austin, TX"}'
```

Every response has the same four keys:

| Key | Purpose |
|---|---|
| `simple` | The headline answer and the two or three numbers behind it |
| `advanced` | Full detail: schedules, year-by-year series, component breakdowns |
| `assumptions` | Every value we filled in for you, each with its source |
| `meta` | Inputs used, resolution confidence, whether live data was reached |

`GET /api/defaults` lists the available services and an empty profile.

### Saving through the API

The same store the Streamlit app uses is exposed over HTTP, so a browser
frontend can persist and read back exactly what the desktop app does:

| Route | Does |
|---|---|
| `GET /api/plans` | List a user's plans |
| `POST /api/plans` | Save a profile — appends a version, returns the new number |
| `GET /api/plans/{slug}` | Current state (`?version=` for a historical one) |
| `GET /api/plans/{slug}/history` | Every save, newest first |
| `POST /api/plans/{slug}/restore/{version}` | Roll back, non-destructively |
| `DELETE /api/plans/{slug}` | Soft delete (`?purge=true` to really remove) |
| `GET /api/export` | Everything held for a user, as JSON |
| `GET /health` | Liveness plus store integrity and schema version |

The API defaults to **trusted-local operation**: both the client socket and
request host must be loopback. Caller-supplied user IDs and email addresses do
not establish identity, including through generic service dispatch.

For an explicitly authenticated, single-account deployment, configure both
`FINREC_API_TOKEN` and `FINREC_API_USER`; requests must use bearer authentication
and the server chooses the account. `FINREC_CORS_ORIGINS` is an explicit
allowlist. This is **not multi-user sign-in**, and it does not authenticate the
separate Streamlit UI.

Invalid inputs are rejected before persistence. Stale version writes return a
conflict rather than replacing a newer snapshot. Full JSON backups contain all
retained historical payloads and include an explicit deleted-plan policy;
restore supports a preview before changes are applied.

### What still has to happen before this is a public website

The local application is not a turnkey public financial-data service:

1. **Authentication.** Public multi-user use still needs real sign-in, sessions
   and authorization for both UI and API. The static API token is one account,
   not a substitute for user identity.
2. **Transport and secrets.** TLS, and the database file moved somewhere with
   backups. `finrec.db.backup()` uses SQLite's online backup API (copying the
   file while it is being written produces a file that looks fine and isn't).
3. **Postgres, when concurrency demands it.** SQLite in WAL mode handles a
   single-server deployment comfortably. `finrec/db.py` is the only module that
   would need reimplementing.
4. **Operational limits.** Analysis inputs are bounded, but public hosting
   still needs rate limiting, resource quotas and monitoring.
5. **Keep Streamlit unless deployment needs change.** Pure engines and service
   envelopes support another frontend without requiring a rewrite now.

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
- **Home sale tax** was also a flat 15% of the gain, ignoring nearly everything that determines it. `home_sale_tax()` now computes the adjusted basis from capital improvements and depreciation, treats depreciation recapture as unrecaptured §1250 gain taxed at ordinary rates capped at 25% (and **never** covered by the §121 exclusion), stacks the remaining LTCG on your ordinary income, adds the 3.8% NIIT including the way the gain itself lifts your MAGI, and applies state tax. On a rental-converted home the difference was $172k versus the old ~$93k — large enough to reverse the decision.

**Wrong question**
- **Keep, sell, or rent** offered three options, but if you're moving out you are not going to keep
  living there — so equivalent housing costs cancel and only two paths are real: keep it and rent it
  out, or sell and invest the proceeds. The page now compares those two on **after-tax wealth over
  time**, not cash flow: a rental can be cash-flow negative and still win, and cash flow was the
  chart most likely to mislead. Rent growth, expense inflation, and rental income sheltered by
  depreciation are all modelled, and the §121 two-of-five-year exclusion produces a visible cliff in
  the wealth chart (in one example the tax on sale jumps from $28,922 to $178,938 in year 4). That
  deadline is usually the single most decisive fact, so the page warns about it explicitly.

**Wrong model**
- **Fixed 7% returns.** The single biggest flaw: it produced one confident number. Replaced with Monte Carlo across three return models (lognormal, Student-t for fat tails, historical bootstrap), reported as percentile bands.
- **Retirement drawdown** subtracted spending pre-tax, understating how much you must actually withdraw. Now grosses up through the bracket schedule, withdraws at the *start* of each year to capture sequence-of-returns risk, applies RMDs, and offers Guyton-Klinger guardrails.
- **Roth vs Traditional** contributed equal *nominal* dollars to each, which isn't a fair comparison — $23,500 into a Roth costs more take-home than $23,500 into a Traditional. Two explicit comparison bases are now offered, and the result is reported as a break-even future tax rate.
- **Mortgage interest and property tax** are subject to deduction limits, including effective-dated SALT rules and the applicable mortgage-debt limit. Only deductions above the alternative standard deduction create an incremental tax benefit.
- **Emergency fund** was a flat 6 months. Now sized from actual risk factors: single vs dual income, job stability, self-employment, dependents, disability coverage.
- **Projections** were quoted in nominal dollars. Net-worth projections are now in today's purchasing power.

**Bugs found while testing this rebuild**
- In buy-vs-rent, only the *renter* invested their monthly surplus. Once rent inflation pushed renting above the cost of owning, the owner's surplus silently vanished from the ledger — a permanent thumb on the scale against buying. Both sides now invest symmetrically.
- Rental underwriting charged year-1 property tax and maintenance against an already-appreciated value, which made the reported break-even rent inconsistent with the cashflow table. Expenses now accrue against the start-of-year value.
- Roth vs Traditional seeded each scenario with only its own existing balance, so the "winner" depended on which pot you already happened to have rather than on the contribution decision. Existing balances now appear in both scenarios.
- The reported marginal rate was federal-only while the effective rate was all-in, so a high earner in California could see a marginal rate *below* their effective rate. Marginal is now federal + state + payroll on the next dollar.

**Bugs found while building the scenario modeller**
- **A plan that bankrupted you scored as a win.** The wealth simulator clamped the portfolio at zero, so spending it couldn't fund was silently discarded while home equity kept compounding. A scenario that emptied the portfolio therefore *outranked* one that didn't. Unfunded spending is now carried as compounding debt, subtracted from net worth, and any scenario that can't be funded is ranked last with the year it breaks named.
- **Home maintenance was counted twice.** The engine strips today's housing cost out of your spending and adds the scenario's back, but the two used different definitions of "housing" — one included maintenance, the other didn't. On a $1.4M home that overstated the deficit by exactly $14,000 a year. Both sides now use one definition, and a test pins the identity that catches it: baseline savings must equal after-tax income minus stated spending, to the cent.
- **A renter's rent disappeared entirely.** Today's housing cost was removed but nothing replaced it in years the scenario didn't say where you live. That also silently zeroed the gap between selling one house and buying the next. Any uncovered year is now charged rent.
- **The chart contradicted the answer.** It plotted net worth against the financial-independence target, but the target is a test on the *portfolio*. With most wealth in property the net-worth line sailed over the finish line years before the portfolio did — or when the portfolio never did at all — while the metric beside it said independence never arrives. The portfolio is now on the chart, the target is labelled "what your investments must reach", and when the two disagree the page says why in as many words.
- **A partner's business loss was erased on sight.** The profile page only showed the partner's section when the partner had a *salary*, so a partner who runs a business and takes no pay had their profit-or-loss zeroed on every visit — and the household's tax rate was computed as if the business didn't exist.
- **State income tax was counted twice in the Roth decision.** The combined marginal rate already includes state tax; the retirement page added the state's rate on top of it, and used the state's *top* rate rather than the rate for that income. A California household on $115,000 was told its marginal rate was 38.6% when the honest figure is 21.3% — a gap wide enough to reverse the recommendation.
- **A 401k balance you already held was labelled "employer contributions".** The Roth page added your existing pre-tax balance to the employer match and captioned the total as the employer's, so a $250,000 balance made it look as though an employer capped at $11,000 a year had supplied two thirds of the pot. The two are now separate lines with separate percentages. The match itself was also a flat percentage of *household* pay: it now respects that a partner's employer matches into a partner's plan, the IRS $350,000 compensation cap, the §415(c) total, and your plan's own dollar cap — and the page names whichever one bit.
- **Adding a row to a table crashed the page.** An editable table keeps a blank row at the bottom; touch one cell and pandas fills the rest of that row with NaN. NaN is truthy, so `float(x or 0)` returned NaN instead of 0, sailed past the guard, and the next `int(...)` raised — typing a name for a new spending change took the whole scenario page down. Table cells are now read through helpers that treat blank, missing and NaN alike, and a half-typed holding no longer turns every price below it into NaN.
- **Deleting a row worked exactly once, then bricked the table.** Removing the last row of a scenario table handed back an empty list, and `pd.DataFrame([])` is 0x0 — no columns at all. The next render drew a table with no headers and no blank row to type into, so there was no way to add anything back short of wiping the saved inputs. Tables are now built from a template row that fixes the columns whether or not there are any rows in them, and the page says how to delete one.
- **The financial-independence number was the answer to a question nobody asked.** The dashboard drew one flat line at `desired_retirement_spending / 4%` and declared independence the moment your portfolio crossed it. But that figure is the cost of stopping *at retirement age*, once the mortgage has gone and spending has fallen to the retirement budget — it says nothing about stopping at 38. On a household spending $240,000 a year against a $91,000 retirement budget it came to $2,275,000 against $2,139,000 already invested, so the page reported **98% of the way to financial independence at 36** for a portfolio that in fact funded a permanent 62% cut in living standards. The target is now a curve that falls with age — the cost of stopping *that* year, funding the gap between today's spending and the retirement budget over the years in between, with the mortgage discounted over its own shorter term — and it converges exactly on the old number at retirement age. The Scenarios page had the same flaw plus one of its own: it re-derived retirement spending as a ratio of today's and clipped that ratio at 0.5, which quietly overrode the $91,000 you had typed and made the two pages quote $2.99M and $2.275M for the same year.
- **A household spending more than it earned was recorded as saving nothing.** `annual_savings` was floored at zero, so the same plan's $119,706 annual *deficit* reached the projection as `0.0` and the chart showed the portfolio compounding gently upward on market returns alone — right past a finish line it was never going to reach. Savings are now signed, the projection withdraws a real deficit, and the dashboard says plainly how big the gap is and that a deliberate, temporary one belongs in a scenario rather than in your long-run path.
- **Every heading on the site rendered in the wrong font.** The stylesheet declared Inter and every paragraph used it, but Streamlit ships its own heading rule at a higher specificity than a bare `h1 {}`, so every title on every page quietly fell back to Source Sans while the prose beneath it was Inter. The font-size rule sitting next to it already carried `!important`, which is exactly why the mismatch looked deliberate. Caught by driving a real browser and reading the *computed* style rather than trusting the stylesheet — the CSS was correct; the cascade was not.
- **Every saved page input was silently thrown away.** `remember()` records the default a value belongs to so a later profile edit can flush a stale entry, and `recall()` refuses a value whose default has moved. Every hand-written pair in the views passed a default to `recall` and none to `remember`, so the recorded default was always `None`, never matched, and the value was discarded on the very next render. The inputs *were* being written to the database — they just never came back. `remember` now inherits the default from the matching `recall`.

Each of these is locked in by a regression test.

---

## Security

The old notebook had live Coinbase API keys and a passphrase pasted into a cell. Those were never committed to git (verified with `git log --all -S`), and the notebook is gone — **but the keys sat in plaintext on disk, so rotate them.**

The rebuild's posture:

- **Credentials come from environment variables only.** Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.
- **Nothing is hardcoded and nothing is logged.** `portfolio.credential_status()` shows only a masked fingerprint so you can confirm a key is loaded without ever displaying it.
- **API access is read-only.** The tracker reads balances and public prices. There is no trading code path.
- **Financial documents are processed on the application server.** `data/*.csv` is gitignored except `sample_*.csv`. Uploads from another computer travel to that server; do not describe a remote deployment as device-local.
- **Saved plans live in an owner-readable SQLite file.** Use full-disk encryption on shared machines. Prepare a versioned JSON backup from the sidebar and keep backups private; they include historical financial information. SQLite files and local backup directories are gitignored.
- **Screenshots are OCR'd by a local `tesseract` process.** The temporary image is removed after processing. Hosted AI is separate and opt-in.
- **The dashboard binds to localhost** and has no authentication — don't expose the port publicly.

---

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider         # complete regression suite
python3 -m pytest tests/test_views.py          # renders every view, simple and advanced
python3 -m pytest tests/test_service.py        # every service returns valid, JSON-safe output
python3 -m pytest tests/test_lookup.py         # location resolution and reference data
python3 -m pytest tests/test_simple_flows.py   # break-even rent, compensation split, quick start
python3 -m pytest tests/test_storage.py        # saved plans survive restarts and corruption
python3 -m pytest tests/test_durability.py     # nothing typed is ever lost: widget -> database -> back
python3 -m pytest tests/test_api.py            # HTTP surface, user scoping, version history
python3 -m pytest tests/test_formatting.py     # every page shows commas and percent signs
python3 -m pytest tests/test_ingest.py         # document import proposes, never overwrites
python3 -m pytest tests/test_contributions.py  # 401k match, HSA limits, 529, action IDs
python3 -m pytest tests/test_advisor_llm.py    # context advice works with no key and no network
python3 -m pytest tests/test_page_persistence.py  # inputs survive navigating away and restarting
python3 -m pytest tests/test_location_consistency.py  # your rates match the address you gave
python3 -m pytest tests/test_dollar_rendering.py      # dollar amounts never turn into maths
python3 -m pytest tests/test_dashboard_and_tailoring.py  # age slider, home projects, crypto, tailoring
python3 -m pytest tests/test_recommend_robustness.py  # the engine survives every realistic profile
python3 -m pytest tests/test_scenarios.py      # life events, rental property, break-even rents, running out of money
python3 -m pytest tests/test_work_and_deductions.py  # W-2 vs self-employed, business losses, deductions, extra properties
python3 -m pytest tests/test_editable_tables.py  # adding *or deleting* a row never breaks a table
python3 -m pytest tests/test_college_plans.py  # one 529 per child, totalling back to the household figure
python3 -m pytest tests/test_fi_target.py      # the finish line moves with the age you stop at
python3 -m pytest tests/test_shell.py         # navigation, theme, no unreachable pages
```

`tests/test_views.py` executes every view through Streamlit's `AppTest` harness **in both simple and advanced mode**, so a broken view fails the test run rather than surfacing when someone clicks a tab. `tests/test_service.py` asserts the `{simple, advanced, assumptions, meta}` contract, that no `NaN` or `inf` can reach a client, that every assumption carries a source, and that no service raises on an empty payload.

Regenerate the sample spending data with `python3 scripts/make_sample_data.py`.

---

## Assumptions worth checking

The models are only as good as their inputs. These are reasonable defaults, not gospel:

- Federal rules are dated for 2024-2026; unsupported years must not silently reuse another year's rules. State estimates do not constitute a complete state tax return: deductions, credits, exclusions and local rules can differ materially. Review the model's state assumptions.
- Location reference data — property tax rates, insurance premiums, price-to-rent ratios, median home prices — are **2025 aggregate estimates**, accurate enough to make a first pass meaningful and wrong enough that you should replace them with real local numbers before acting. Every one is shown in the Assumptions panel and is editable.
- Auto-filled rents come from a metro-level price-to-rent ratio applied to the home price. That is a market average, not a quote for a specific property.
- The starting spending estimate in onboarding is a heuristic (a share of after-tax income, scaled by local cost of living). Replace it with your real number as soon as you can — most outputs are sensitive to it.
- Renovation cost-recouped rates are national averages and vary enormously by region.
- Savings-vehicle APYs and solar production figures are illustrative, not live data.
- Expected returns and volatility are long-run assumptions you can change on any page.
- The scenario FI target is a gross-financial-asset screen, not a tax-aware withdrawal plan.
  Restricted accounts cannot silently fund home purchases or spending deficits, but this
  scenario engine does not model their later withdrawals. Use the Retirement analysis
  for account-aware after-tax spending and sequence-risk modeling.
- Scenario starting debts use disclosed assumed repayment terms where the profile lacks
  an actual schedule. Replace estimates with actual debt terms before relying on a result.
- State tax remains approximate; Washington capital-gains excise is not modeled. The
  selected federal tax year does not make every state reference assumption current.
- Retirement projections apply the selected tax-year rules as a planning convention;
  they do not forecast future legislation. Qualified Roth treatment and stated withdrawal
  ordering are assumptions, not individualized tax optimization.

## Disclaimer

This is a personal modelling tool, not financial, tax, or legal advice. Verify anything material with a professional before acting on it.

## Decision workflows

- The dashboard distinguishes **planned, completed, snoozed and not-applicable** actions. Snoozes have a revisit date; dismissals require a reason. Completed actions remain in history even when updated numbers stop generating the recommendation.
- Profile provenance distinguishes entered, imported and estimated inputs. Missing provenance on older plans is shown as unknown, not retrospectively claimed as verified.
- Save named alternatives on **Scenarios**, then compare them with the same market assumptions and random paths. The summary separates terminal net worth, financial-independence age, planned cash commitments and funding gaps. A cash-commitment total is not the same as the cash needed today.
- Scenario spending starts empty. Example childcare spending is inserted only when requested.
- Retirement success means meeting after-tax spending. Mortgage comparisons include remaining debt and equal cash commitments; rental profit deducts initial capital once.

## Reproducible development

Python **3.11 and 3.12** are supported. The core, API, development and optional
integration requirements are separate. Direct dependencies are pinned and
`constraints.txt` records the resolved development environment. Platform-specific
packages can differ; CI runs the complete suite on both supported Python versions.

Upgrade dependencies deliberately in a fresh virtual environment, run
`python -m pip check` and the regression suite, then regenerate the constraints
with `python -m pip freeze > constraints.txt`. Do not freeze an unrelated global
Python installation. Browser-level walkthroughs should use a temporary
`FINREC_HOME`, never an existing user's saved plans.
