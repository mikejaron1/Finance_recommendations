"""Shared green theme and self-hosted typography."""

PALETTE = {
    "primary": "#0E7C5A",
    "secondary": "#C1810B",
    "accent": "#2F6FB5",
    "danger": "#BE3A31",
    "neutral": "#64748B",
    "band": "rgba(14,124,90,0.14)",
}

CHART_SEQUENCE = [
    "#0E7C5A", "#2F6FB5", "#C1810B", "#7C5BA6",
    "#2A8C8A", "#BE3A31", "#64748B", "#A15C2E",
]

PRIORITY_COLORS = {
    "🔴 Critical": "#BE3A31",
    "🟠 High": "#C1810B",
    "🟡 Medium": "#C9A227",
    "🟢 Low": "#0E7C5A",
    "ℹ️ Info": "#64748B",
}

EVENT_COLORS = {
    "buy_home": "#7C5BA6",
    "sell_home": "#A15C2E",
    "let_home": "#2A8C8A",
    "rent": "#2A8C8A",
    "sell_property": "#C1810B",
    "spending_start": "#BE3A31",
    "spending_end": "#0E7C5A",
    "income_change": "#2F6FB5",
    "purchase": "#7C5BA6",
}

EVENT_FALLBACK = "#64748B"

PLOTLY_FONT = ('InterFP, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, '
               '"Helvetica Neue", Arial, sans-serif')

DISCLAIMER = (
    "These are modelled estimates for planning purposes, based on the assumptions you enter. "
    "They are not tax, legal or investment advice. Verify tax figures against current IRS "
    "guidance and consult a professional before acting on large decisions."
)

CSS = """
<style>
/* Inter, self-hosted from app/static/fonts. Linking Google Fonts would send
   every visitor's IP to a third party on each page load, which contradicts
   the local-only promise this app makes about financial data. */
@font-face { font-family: "InterFP"; font-style: normal; font-weight: 400;
  font-display: swap; src: url("app/static/fonts/inter-400.woff2") format("woff2"); }
@font-face { font-family: "InterFP"; font-style: normal; font-weight: 500;
  font-display: swap; src: url("app/static/fonts/inter-500.woff2") format("woff2"); }
@font-face { font-family: "InterFP"; font-style: normal; font-weight: 600;
  font-display: swap; src: url("app/static/fonts/inter-600.woff2") format("woff2"); }
@font-face { font-family: "InterFP"; font-style: normal; font-weight: 700;
  font-display: swap; src: url("app/static/fonts/inter-700.woff2") format("woff2"); }

:root {
  --fp-primary: #0E7C5A;
  --fp-primary-dark: #0A5C43;
  --fp-primary-tint: #ECF6F1;
  --fp-accent: #2F6FB5;
  --fp-ink: #0B1512;
  --fp-body: #3D4A45;
  --fp-muted: #6B7A75;
  --fp-line: #E8ECEA;
  --fp-line-soft: #F1F4F3;
  --fp-surface: #ffffff;
  --fp-canvas: #F7F9F8;
  --fp-tint: #F7F9F8;
  --fp-radius: 14px;
  --fp-shadow-sm: 0 1px 2px rgba(11,21,18,.04);
  --fp-shadow: 0 1px 3px rgba(11,21,18,.05), 0 8px 24px -8px rgba(11,21,18,.08);
  --fp-font: "InterFP", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
             "Helvetica Neue", Arial, sans-serif;
}

html, body, [class*="css"], .stMarkdown, .stApp, button, input, select, textarea {
  font-family: var(--fp-font);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.stApp { background: var(--fp-canvas); }

/* A single wide content column on a tinted canvas. Generous horizontal
   padding and a capped measure are what separate a product page from a
   wall-to-wall dashboard grid. */
.block-container {
  padding: 1.8rem 3.2rem 3rem 3.2rem; max-width: 1180px;
  background: var(--fp-surface);
  border: 1px solid var(--fp-line);
  border-radius: 16px;
  margin-top: 1.25rem; margin-bottom: 2.5rem;
  box-shadow: var(--fp-shadow);
}
@media (max-width: 900px) {
  .block-container { padding: 1rem .85rem 1.5rem; border-radius: 12px; margin-top: .5rem; }
}

/* ---------- Typography ----------
   font-family needs !important: Streamlit ships its own heading rule at a
   higher specificity, so a plain `h1 {}` here loses and every heading renders
   in Source Sans while the body text around it is Inter. The size rules below
   already carried !important, which is why the mismatch was easy to miss. */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2, [data-testid="stHeading"] h3 {
  color: var(--fp-ink); font-family: var(--fp-font) !important;
}
h1 { font-size: 2.15rem !important; font-weight: 700; letter-spacing: -0.032em;
     line-height: 1.15; margin-bottom: .3rem; }
h2 { font-size: 1.32rem !important; font-weight: 640; letter-spacing: -0.022em;
     margin-top: 2.4rem !important; padding-top: 0; }
h3 { font-size: 1.06rem !important; font-weight: 640; letter-spacing: -0.014em;
     margin-top: 1.5rem !important; }
h4 { font-size: .95rem !important; font-weight: 640; letter-spacing: -0.01em; }
p, li { color: var(--fp-body); font-size: .945rem; line-height: 1.62; }
.stMarkdown p { margin-bottom: .7rem; }
small, .stCaption, [data-testid="stCaptionContainer"] p {
  color: var(--fp-muted) !important; font-size: .83rem !important; line-height: 1.55;
}
a { color: var(--fp-primary-dark); text-decoration: none; font-weight: 520; }
a:hover { text-decoration: underline; }
strong, b { font-weight: 640; color: var(--fp-ink); }
code, kbd { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: .86em; background: var(--fp-line-soft); padding: .1rem .34rem;
  border-radius: 5px; color: var(--fp-ink); }

header[data-testid="stHeader"] { background: transparent; border-bottom: none; height: 0; }
[data-testid="stToolbar"] { right: .6rem; }
[data-testid="stDecoration"] { display: none; }

/* ---------- Page header ---------- */
.fp-pagehead {
  display: flex; align-items: flex-start; gap: 1rem;
  margin: 0 0 1.1rem 0; padding-bottom: .75rem;
  border-bottom: 1px solid var(--fp-line);
}
.fp-pagehead-badge {
  flex: 0 0 auto; display: inline-flex; align-items: center; justify-content: center;
  width: 2.9rem; height: 2.9rem; border-radius: 12px; font-size: 1.35rem;
  background: var(--fp-primary-tint); border: 1px solid #DCEBE4;
  margin-top: .15rem; line-height: 1;
}
.fp-pagehead-text { min-width: 0; }
.fp-eyebrow {
  display: inline-flex; align-items: center; gap: .4rem;
  font-size: .69rem; font-weight: 660; letter-spacing: .11em; text-transform: uppercase;
  color: var(--fp-muted); margin-bottom: .3rem;
}
.fp-pagehead h1 { margin: 0; }
.fp-pagehead .fp-lede {
  color: var(--fp-muted) !important; font-size: 1.02rem; line-height: 1.55;
  margin: .5rem 0 0 0; max-width: 68ch; font-weight: 400;
}
@media (max-width: 640px) {
  .fp-pagehead-badge { display: none; }
  .fp-pagehead .fp-lede { font-size: .9rem; margin-top: .3rem; }
  h1 { font-size: 1.7rem !important; }
}
.fp-card-link { display: block; color: inherit; }
.fp-card-link:hover { text-decoration: none; }
.fp-card-link:focus-visible { outline: 3px solid var(--fp-primary); outline-offset: 3px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
  background: var(--fp-surface);
  border-right: 1px solid var(--fp-line);
}
section[data-testid="stSidebar"] > div { padding-top: .3rem; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: .6rem; }

[data-testid="stSidebarNav"] { padding-top: .1rem; }
[data-testid="stSidebarNav"] ul { padding-left: .4rem; padding-right: .4rem; }
[data-testid="stSidebarNav"] a {
  border-radius: 8px; padding: .38rem .62rem; margin: .06rem 0;
  transition: background .12s ease, color .12s ease;
}
[data-testid="stSidebarNav"] a:hover { background: var(--fp-line-soft); text-decoration: none; }
[data-testid="stSidebarNav"] a span { color: var(--fp-body); font-size: .905rem; font-weight: 500; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: var(--fp-primary-tint); }
[data-testid="stSidebarNav"] a[aria-current="page"] span {
  color: var(--fp-primary-dark) !important; font-weight: 640;
}
[data-testid="stSidebarNav"] [data-testid="stNavSectionHeader"] {
  font-size: .67rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--fp-muted); font-weight: 660; padding: 1rem .62rem .25rem .62rem;
}

/* The brand mark is an SVG rendered by st.logo above the navigation. */
[data-testid="stLogo"] { margin: .4rem 0 .7rem .2rem; height: 2.3rem; }

.fp-sidecard {
  background: var(--fp-canvas); border: 1px solid var(--fp-line);
  border-radius: 11px; padding: .8rem .9rem; margin: .1rem 0 .7rem 0;
}
.fp-sidecard .fp-side-label {
  font-size: .655rem; letter-spacing: .11em; text-transform: uppercase;
  color: var(--fp-muted); font-weight: 660; margin-bottom: .35rem;
}
.fp-side-row {
  display: flex; justify-content: space-between; align-items: baseline;
  padding: .22rem 0; font-size: .86rem; color: var(--fp-muted);
}
.fp-side-row b { color: var(--fp-ink); font-weight: 640;
                 font-variant-numeric: tabular-nums; }

/* ---------- Custom surfaces ----------
   Every one is a light surface, so its text colour is stated explicitly
   rather than inherited. Inheriting leaves white-on-white if the theme ever
   resolves to dark. */
.fp-hero, .fp-answer, .fp-card, .fp-chip, .fp-foot { color: var(--fp-ink); }
.fp-hero h1, .fp-hero h2, .fp-hero p,
.fp-card h4, .fp-card p,
.fp-answer div, .fp-answer p, .fp-answer li,
.fp-foot { color: inherit; }

.fp-hero {
  background: linear-gradient(150deg, #F3F9F6 0%, #F1F6FA 100%);
  border: 1px solid var(--fp-line); border-radius: var(--fp-radius);
  padding: 2.4rem 2.5rem; margin-bottom: 1.6rem;
}
.fp-hero h1 { margin: 0 0 .55rem 0; color: var(--fp-ink); }
.fp-hero p { color: var(--fp-muted) !important; font-size: 1.05rem; margin: 0; max-width: 62ch; }

/* The single sentence the user came for. Deliberately the loudest thing on
   the page — bigger than the H1 that sits above it. */
.fp-answer {
  background: linear-gradient(180deg, #FCFDFD 0%, var(--fp-surface) 100%);
  border: 1px solid var(--fp-line);
  border-left: 3px solid var(--fp-primary); border-radius: 12px;
  padding: 1.35rem 1.6rem; margin: .6rem 0 1.5rem 0;
  box-shadow: var(--fp-shadow-sm);
}
.fp-answer .fp-label {
  font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
  color: var(--fp-muted) !important; font-weight: 660;
}
.fp-answer .fp-headline {
  font-size: 1.46rem; font-weight: 660; color: var(--fp-ink) !important;
  margin-top: .38rem; line-height: 1.32; letter-spacing: -.024em;
}
.fp-answer .fp-sub { color: var(--fp-muted) !important; margin-top: .55rem;
                     font-size: .93rem; line-height: 1.6; }
.fp-answer.warn { border-left-color: #C1810B; }
.fp-answer.bad { border-left-color: #BE3A31; }

.fp-card {
  background: var(--fp-surface); border: 1px solid var(--fp-line); border-radius: 12px;
  padding: 1.15rem 1.3rem; margin-bottom: .45rem;
  transition: box-shadow .16s ease, border-color .16s ease, transform .16s ease;
}
.fp-card:hover { box-shadow: var(--fp-shadow); border-color: #D6E0DB; transform: translateY(-1px); }
.fp-card h4 { margin: 0 0 .35rem 0; font-size: 1rem; color: var(--fp-ink) !important; }
.fp-card p { color: var(--fp-muted) !important; font-size: .88rem; margin: 0; line-height: 1.55; }

.fp-chip {
  display: inline-block; background: var(--fp-primary-tint); border: 1px solid #DCEBE4;
  border-radius: 999px; padding: .2rem .72rem; font-size: .78rem; font-weight: 560;
  color: var(--fp-primary-dark) !important; margin-right: .35rem;
}
.fp-foot { color: var(--fp-muted) !important; font-size: .78rem; line-height: 1.6;
           border-top: 1px solid var(--fp-line); margin-top: 3rem; padding-top: 1.2rem; }
.fp-source { color: var(--fp-muted) !important; font-size: .84rem; }
.fp-subtitle { color: var(--fp-muted) !important; font-size: 1.02rem;
               margin-top: -.35rem; margin-bottom: 1.3rem; line-height: 1.55; }

/* ---------- Metrics ----------
   Understated: a label, a number, a hairline. Boxed metrics in a grid are
   the single most "admin dashboard" thing a page can do. */
[data-testid="stMetric"] {
  background: transparent; border: none; border-top: 2px solid var(--fp-line);
  border-radius: 0; padding: .7rem .1rem .2rem .1rem;
}
[data-testid="stMetricLabel"] p {
  color: var(--fp-muted) !important; font-size: .78rem !important; font-weight: 560;
  letter-spacing: .005em;
}
[data-testid="stMetricValue"] {
  color: var(--fp-ink); font-size: 1.62rem; font-weight: 660; letter-spacing: -.03em;
  font-variant-numeric: tabular-nums; line-height: 1.25;
}
[data-testid="stMetricDelta"] { font-size: .8rem; font-weight: 560; }
section[data-testid="stSidebar"] [data-testid="stMetric"] {
  border-top: 1px solid var(--fp-line); padding: .5rem .1rem .1rem .1rem;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.15rem; }

/* ---------- Controls ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 9px; font-weight: 560; font-size: .9rem;
  border: 1px solid var(--fp-line); padding: .42rem 1rem;
  transition: background .14s ease, border-color .14s ease, box-shadow .14s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: #CBD8D2; background: var(--fp-canvas); box-shadow: var(--fp-shadow-sm);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background: var(--fp-primary); border-color: var(--fp-primary); color: #fff;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
  background: var(--fp-primary-dark); border-color: var(--fp-primary-dark);
}

label[data-testid="stWidgetLabel"] p {
  font-size: .855rem !important; font-weight: 550; color: var(--fp-body) !important;
}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
  border-radius: 9px !important;
}
.stTextInput input, .stNumberInput input, .stTextArea textarea { font-size: .91rem; }
[data-testid="stWidgetLabel"] { margin-bottom: .25rem; }

[data-testid="stExpander"] {
  border: 1px solid var(--fp-line); border-radius: 11px; background: var(--fp-surface);
  box-shadow: none; margin-bottom: .6rem;
}
[data-testid="stExpander"] summary { font-size: .9rem; font-weight: 560; padding: .7rem .95rem; }
[data-testid="stExpander"] summary:hover { color: var(--fp-primary-dark); }

.stTabs [data-baseweb="tab-list"] { gap: .15rem; border-bottom: 1px solid var(--fp-line); }
.stTabs [data-baseweb="tab"] {
  border-radius: 8px 8px 0 0; padding: .55rem .95rem; font-weight: 550; font-size: .9rem;
  color: var(--fp-muted);
}
.stTabs [data-baseweb="tab"]:hover { color: var(--fp-ink); background: var(--fp-line-soft); }
.stTabs [aria-selected="true"] { color: var(--fp-primary-dark) !important; background: transparent; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--fp-primary); }

[data-testid="stProgress"] > div > div > div { background: var(--fp-line-soft); }
[data-testid="stProgress"] > div > div > div > div { background: var(--fp-primary); }

/* Alerts: a tinted panel with a coloured spine, not a saturated block. */
[data-testid="stAlert"] {
  border-radius: 10px; border: 1px solid var(--fp-line); border-left-width: 3px;
  box-shadow: none; padding: .85rem 1.05rem;
}
[data-testid="stAlert"] p { font-size: .9rem; line-height: 1.6; }

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { border-color: var(--fp-primary); }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border-radius: 10px; border: 1px solid var(--fp-line); overflow: hidden;
}
hr { border-color: var(--fp-line); margin: 2rem 0; }
[data-testid="stHorizontalBlock"] { gap: 1.15rem; }
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] { gap: .85rem; }
.stPlotlyChart { border-radius: 10px; }
</style>
"""
