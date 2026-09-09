"""Finance Planner — site entrypoint.

Run with::

    streamlit run app/main.py

This file owns the *shell*: navigation, global styling, and the onboarding
gate. It contains no analysis. Each view under ``app/views/`` renders one
question; each of those calls :mod:`finrec.service`, which is the same code
path a future browser frontend would hit over HTTP.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
for _p in (str(_APP_DIR), str(_APP_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _shared import st  # noqa: E402

from _shared import CSS, get_profile, is_onboarded  # noqa: E402

BRAND = "Finance Planner"
LOGO = str(_APP_DIR / "assets" / "logo.svg")
MARK = str(_APP_DIR / "assets" / "mark.svg")

st.set_page_config(
    page_title=BRAND,
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"about": f"{BRAND} — modelled estimates for planning, not advice."},
)
st.markdown(CSS, unsafe_allow_html=True)

# Sits above the navigation in the sidebar, and collapses to the bare mark
# when the sidebar is closed — the slot a real product puts its name in.
st.logo(LOGO, icon_image=MARK, size="large")

get_profile()

# Until there is a profile there is nothing to analyse, so the site is a single
# page. Showing a logged-out visitor nine empty calculators is how planning
# tools lose people in the first thirty seconds.
if not is_onboarded():
    st.navigation([st.Page("views/welcome.py", title="Get started", icon="👋", default=True)],
                  position="hidden").run()
else:
    # Navigation lives in the left sidebar rather than across the top. Twelve
    # pages in a top bar wrap and truncate; a sidebar gives every page a
    # permanent, readable home — including the profile, which people need to
    # reach from anywhere the moment a number looks wrong.
    navigation = {
        # An empty section header pins these three to the top of the sidebar
        # with no heading to collapse them under. They are the pages people
        # move between constantly, so they stay put on every page.
        "": [
            st.Page("views/profile.py", title="Profile", icon="👤", url_path="profile"),
            st.Page("views/home.py", title="Dashboard", icon="📊", default=True),
            st.Page("views/scenarios.py", title="Scenarios", icon="🔮", url_path="scenarios"),
        ],
        "Home & property": [
            st.Page("views/buy_vs_rent.py", title="Buy vs rent", icon="🏡", url_path="buy_vs_rent"),
            st.Page("views/mortgage.py", title="Mortgage", icon="🏦", url_path="mortgage"),
            st.Page("views/investment_property.py", title="Rental property", icon="🏘️", url_path="investment_property"),
            st.Page("views/projects.py", title="Home projects", icon="🔨", url_path="projects"),
        ],
        "Money & tax": [
            st.Page("views/retirement.py", title="Retirement", icon="🎯", url_path="retirement"),
            st.Page("views/investing.py", title="Investing", icon="📈", url_path="investing"),
            st.Page("views/spending.py", title="Spending & cash", icon="🧾", url_path="spending"),
            st.Page("views/taxes.py", title="Taxes", icon="🧮", url_path="taxes"),
        ],
    }
    st.navigation(navigation, position="sidebar", expanded=True).run()
