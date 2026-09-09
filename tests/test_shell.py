"""The site shell: navigation, theme, and the chrome every page shares.

The shell is checked two ways: statically, by parsing ``main.py``, and for
real, by booting it under ``AppTest`` (see ``TestTheShellActuallyRuns`` — it
does execute, provided a plan is seeded first, otherwise the app stops at the
onboarding gate whose navigation is ``position="hidden"``). The contracts:

* every page the navigation offers resolves to a real file;
* every view file is reachable from the navigation (an unreachable page is
  dead code that still ships);
* the profile page has a permanent home, because it is the one page people
  need from anywhere the moment a number looks wrong;
* colours come from the shared palette rather than being hand-typed per view,
  which is how the same category ended up green on one page and red on another.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
VIEWS = APP_DIR / "views"
for _p in (str(ROOT), str(APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAIN = APP_DIR / "main.py"
MAIN_SRC = MAIN.read_text()


def nav_pages() -> list[dict]:
    """Every ``st.Page(...)`` in main.py, as {path, title, icon, default}."""
    tree = ast.parse(MAIN_SRC)
    pages = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "Page":
            continue
        entry = {"path": node.args[0].value if node.args else None, "default": False}
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant):
                entry[kw.arg] = kw.value.value
        pages.append(entry)
    return pages


def nav_groups() -> list[str]:
    """The section headings the sidebar shows, in order.

    An empty string is a real, meaningful key: it is how Streamlit renders a
    group with no heading, which is what keeps the top three pages unlabelled
    and always visible.
    """
    return re.findall(r'^\s{8}"([^"]*)": \[', MAIN_SRC, flags=re.M)


class TestEveryPageResolves:
    def test_the_navigation_is_not_empty(self):
        assert len(nav_pages()) >= 5

    @pytest.mark.parametrize("page", nav_pages(), ids=lambda p: p["path"])
    def test_the_file_behind_each_link_exists(self, page):
        assert (APP_DIR / page["path"]).is_file(), (
            f"navigation points at {page['path']}, which is not a file — "
            "the link would 404 for every visitor"
        )

    def test_every_page_has_a_title_and_an_icon(self):
        for page in nav_pages():
            assert page.get("title"), f"{page['path']} has no title"
            assert page.get("icon"), f"{page['path']} has no icon"

    def test_exactly_one_page_is_the_default_landing(self):
        """Scoped to the main navigation. The onboarding gate is a separate
        single-page navigation with its own default, which is correct."""
        defaults = [p for p in nav_pages()
                    if p.get("default") and p["path"] != "views/welcome.py"]
        assert len(defaults) == 1, "a site needs exactly one front door"
        assert defaults[0]["path"] == "views/home.py"

    def test_no_two_pages_share_a_title(self):
        titles = [p["title"] for p in nav_pages()]
        assert len(titles) == len(set(titles))


class TestNothingIsStranded:
    def test_every_view_is_reachable_from_the_navigation(self):
        """A view nobody can click is dead code that still ships. ``welcome``
        is the exception: it is the onboarding gate, shown by a separate
        hidden navigation before a profile exists."""
        linked = {p["path"].split("/")[-1] for p in nav_pages()}
        on_disk = {p.name for p in VIEWS.glob("*.py") if not p.name.startswith("_")}
        stranded = on_disk - linked - {"welcome.py"}
        assert not stranded, f"unreachable views: {sorted(stranded)}"

    def test_the_onboarding_gate_is_still_wired_up(self):
        assert "views/welcome.py" in MAIN_SRC
        assert "is_onboarded" in MAIN_SRC


class TestTheProfileIsAlwaysOneClickAway:
    def test_the_profile_page_is_in_the_navigation(self):
        paths = [p["path"] for p in nav_pages()]
        assert "views/profile.py" in paths

    def test_it_is_the_very_first_link(self):
        """Above the dashboard. It is where people go to correct an input, so
        it should not be something they have to look for."""
        pages = [p for p in nav_pages() if p["path"] != "views/welcome.py"]
        assert pages[0]["path"] == "views/profile.py"

    def test_it_is_labelled_simply(self):
        page = next(p for p in nav_pages() if p["path"] == "views/profile.py")
        assert page["title"] == "Profile"

    def test_profile_is_in_the_permanent_navigation(self):
        assert any(p["path"] == "views/profile.py" for p in nav_pages())

    def test_that_link_survives_being_rendered_outside_the_navigation(self):
        """Views run standalone (as every test here does) have no navigation
        context, and ``st.page_link`` raises there. It must not take the
        whole sidebar down with it."""
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(VIEWS / "home.py"), default_timeout=120).run()
        assert not app.exception


class TestTheBrandIsPresent:
    """``st.logo`` renders above the navigation. A missing or malformed asset
    leaves the sidebar looking unfinished rather than raising, so the files
    are checked rather than trusted."""

    def test_the_logo_files_exist(self):
        for name in ("logo.svg", "mark.svg"):
            assert (APP_DIR / "assets" / name).is_file(), f"missing brand asset {name}"

    def test_they_are_valid_svg(self):
        import xml.dom.minidom

        for name in ("logo.svg", "mark.svg"):
            doc = xml.dom.minidom.parse(str(APP_DIR / "assets" / name))
            assert doc.documentElement.tagName == "svg"

    def test_the_mark_uses_the_brand_colour(self):
        from _shared import PALETTE

        mark = (APP_DIR / "assets" / "mark.svg").read_text()
        assert PALETTE["primary"] in mark

    def test_the_shell_registers_the_logo(self):
        assert "st.logo(" in MAIN_SRC


class TestTheShellActuallyRuns:
    """End-to-end through the real ``st.navigation``.

    An earlier version of this file asserted the shell could not be rendered
    under ``AppTest``. That was wrong: running ``main.py`` with no saved plan
    hits the onboarding gate, whose navigation is ``position="hidden"`` and
    renders nothing. Seed a plan and the whole shell executes, which is worth
    far more than the static checks above.
    """

    @staticmethod
    def _seeded_app(tmp_path, monkeypatch):
        from finrec import db, storage
        from finrec.profile import Profile

        monkeypatch.setenv("FINREC_HOME", str(tmp_path / "home"))
        db.reset_connection()
        storage.save_plan(
            Profile(age=36, gross_income=200_000, monthly_spending=8_000,
                    location="Austin, TX"),
            "My plan",
        )
        from streamlit.testing.v1 import AppTest

        return AppTest.from_file(str(MAIN), default_timeout=180).run()

    @staticmethod
    def _headings(app) -> str:
        """The page title is rendered as HTML inside a markdown block, not via
        ``st.title``, so ``AppTest`` reports no Title element. The real DOM
        does get an ``<h1>`` — this reads it back out of the markup."""
        return " ".join(m.value for m in app.markdown if m.value)

    def test_the_site_renders_a_page_through_the_navigation(self, tmp_path, monkeypatch):
        app = self._seeded_app(tmp_path, monkeypatch)
        assert not app.exception
        assert app.markdown, "navigation ran but no page rendered"

    def test_it_lands_on_the_dashboard_even_though_profile_is_listed_first(
            self, tmp_path, monkeypatch):
        """Order in the menu and the default landing page are independent.
        Moving the profile to the top must not change the front door."""
        app = self._seeded_app(tmp_path, monkeypatch)
        assert "Dashboard" in self._headings(app)

    def test_the_page_title_is_a_real_h1_not_just_styled_text(
            self, tmp_path, monkeypatch):
        """The header is hand-written HTML, so nothing forces it to stay a
        heading. Downgrading it to a styled <div> would look identical and
        silently break screen readers and the document outline."""
        app = self._seeded_app(tmp_path, monkeypatch)
        # The stylesheet is itself a markdown element and mentions the class,
        # so match on the rendered element rather than the rule that styles it.
        head = next((m.value for m in app.markdown
                     if "fp-pagehead-text" in m.value and "<style>" not in m.value), None)
        assert head, "no page header block rendered"
        assert "<h1>" in head, f"page header is not an h1: {head[:200]}"

    def test_an_empty_section_header_is_accepted_alongside_labelled_ones(
            self, tmp_path, monkeypatch):
        """The docs only describe the empty-string section for top navigation.
        This pins that it works for the sidebar too."""
        app = self._seeded_app(tmp_path, monkeypatch)
        assert not app.exception

    def test_a_visitor_with_no_plan_gets_the_onboarding_gate(self, tmp_path, monkeypatch):
        from finrec import db

        monkeypatch.setenv("FINREC_HOME", str(tmp_path / "empty"))
        db.reset_connection()
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(MAIN), default_timeout=180).run()
        assert not app.exception


class TestNavigationIsGroupedForNewcomers:
    def test_pages_are_grouped(self):
        groups = nav_groups()
        assert len(groups) >= 3, "twelve ungrouped links is a wall, not a menu"

    def test_the_core_pages_sit_in_an_unlabelled_group_at_the_top(self):
        """Profile, Dashboard and Scenarios are the pages people move between
        constantly. An empty section header keeps them at the top with no
        heading to collapse them under."""
        assert nav_groups()[0] == ""
        first_three = [p["path"] for p in nav_pages()
                       if p["path"] != "views/welcome.py"][:3]
        assert first_three == ["views/profile.py", "views/home.py", "views/scenarios.py"]

    def test_the_old_plan_heading_is_gone(self):
        assert "Your plan" not in nav_groups()

    def test_the_remaining_groups_are_labelled(self):
        """Only the top group is unlabelled; the rest need headings or the
        sidebar is a wall of twelve links again."""
        assert [g for g in nav_groups() if g] == ["Home & property", "Money & tax"]

    def test_navigation_lives_in_the_sidebar(self):
        """A top bar truncates at twelve pages, and the profile link was the
        first casualty."""
        assert 'position="sidebar"' in MAIN_SRC
        assert 'position="top"' not in MAIN_SRC

    def test_sidebar_adapts_to_screen_width(self):
        assert 'initial_sidebar_state="auto"' in MAIN_SRC

    def test_the_menu_never_hides_pages_behind_a_view_more_button(self):
        """``st.navigation`` defaults to ``expanded=False``, which collapses
        the overflow once there are enough pages — so the lower groups vanish
        behind a button on exactly the long menu that needs them visible."""
        assert "expanded=True" in MAIN_SRC


class TestTheLookIsCentralised:
    def test_views_do_not_hand_type_brand_colours(self):
        """Chart colours belong in ``PALETTE`` / ``CHART_SEQUENCE``. Hand-typed
        hexes drift: the same category was green on one page, red on another."""
        offenders = {}
        for path in VIEWS.glob("*.py"):
            hexes = re.findall(r'"#[0-9A-Fa-f]{6}"', path.read_text())
            if hexes:
                offenders[path.name] = sorted(set(hexes))
        assert not offenders, f"hardcoded colours outside the palette: {offenders}"

    def test_the_palette_and_the_stylesheet_agree_on_the_brand_colour(self):
        from _shared import CSS, PALETTE

        assert f"--fp-primary: {PALETTE['primary']};" in CSS

    def test_the_configured_theme_matches_the_palette(self):
        from _shared import PALETTE

        config = (ROOT / ".streamlit" / "config.toml").read_text()
        assert f'primaryColor = "{PALETTE["primary"]}"' in config

    def test_the_theme_stays_light(self):
        """Custom cards are light surfaces with explicit dark text. Following
        the visitor's OS dark mode puts white body text on white cards."""
        config = (ROOT / ".streamlit" / "config.toml").read_text()
        assert 'base = "light"' in config

    def test_the_chart_sequence_has_no_duplicates(self):
        from _shared import CHART_SEQUENCE

        assert len(CHART_SEQUENCE) == len(set(CHART_SEQUENCE))
        assert len(CHART_SEQUENCE) >= 6, "donuts need enough distinct slices"

    def test_the_stylesheet_is_applied_by_the_shared_page_setup(self):
        """Styling injected only by main.py would vanish on every page that a
        test — or a deep link — renders on its own."""
        shared = "\n".join(path.read_text() for path in
                           [APP_DIR / "_shared.py", *sorted((APP_DIR / "ui").glob("*.py"))])
        assert "st.markdown(CSS, unsafe_allow_html=True)" in shared


class TestTheDesignSystem:
    """The look is a contract too. These pin the pieces that are easy to
    half-apply: a font that is named but never loaded, charts that keep
    Plotly's defaults while the prose around them is restyled, and page
    headers that drift back to bare ``st.title`` calls."""

    def test_the_font_files_are_actually_present(self):
        fonts = sorted((APP_DIR / "static" / "fonts").glob("inter-*.woff2"))
        assert len(fonts) >= 4, f"expected four weights, found {[f.name for f in fonts]}"
        for f in fonts:
            assert f.stat().st_size > 5_000, f"{f.name} looks truncated"
            assert f.read_bytes()[:4] == b"wOF2", f"{f.name} is not a woff2 file"

    def test_every_font_the_css_asks_for_exists_on_disk(self):
        """A typo in a @font-face URL fails silently — the browser just falls
        back to the system font and the page looks almost right."""
        from _shared import CSS

        urls = re.findall(r'url\("([^"]+\.woff2)"\)', CSS)
        assert urls, "no self-hosted font declared"
        for u in urls:
            path = ROOT / u.lstrip("/")
            assert path.exists(), f"CSS references {u}, which is not on disk"

    def test_serving_the_font_directory_is_switched_on(self):
        """Without this the @font-face URLs 404 and Inter never loads."""
        config = (ROOT / ".streamlit" / "config.toml").read_text()
        assert "enableStaticServing = true" in config

    def test_the_font_is_self_hosted_rather_than_pulled_from_a_cdn(self):
        """A Google Fonts link would send every visitor's IP to a third party
        on page load, which contradicts what the README promises about this
        app keeping financial data local."""
        from _shared import CSS

        for bad in ("fonts.googleapis.com", "fonts.gstatic.com", "@import url(http"):
            assert bad not in CSS, f"stylesheet reaches out to {bad}"

    def test_headings_override_streamlits_own_font_rule(self):
        """Streamlit ships a heading rule at a higher specificity, so a plain
        ``h1 {}`` loses and every title renders in Source Sans while the body
        text is Inter. Only ``!important`` wins."""
        from _shared import CSS

        # Comments mention `h1 {}` in prose, so strip them before matching.
        css = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)
        rules = [r for r in re.findall(r"([^{}]*)\{([^}]*)\}", css)
                 if re.search(r"\bh1\b", r[0])]
        assert rules, "no rule targets h1 at all"
        assert any("font-family: var(--fp-font) !important" in body for _, body in rules), \
            "no h1 rule forces the site font over Streamlit's own heading rule"

    def test_charts_use_the_same_typeface_as_the_page(self):
        import plotly.graph_objects as go

        from _shared import PLOTLY_FONT, base_layout

        fig = base_layout(go.Figure([go.Scatter(x=[1, 2], y=[1, 2])]), title="T")
        assert fig.layout.font.family == PLOTLY_FONT
        assert "InterFP" in PLOTLY_FONT

    def test_charts_are_transparent_so_they_sit_on_the_page_card(self):
        import plotly.graph_objects as go

        from _shared import base_layout

        fig = base_layout(go.Figure([go.Scatter(x=[1, 2], y=[1, 2])]))
        assert "rgba(0,0,0,0)" in str(fig.layout.paper_bgcolor)
        assert "rgba(0,0,0,0)" in str(fig.layout.plot_bgcolor)

    def test_charts_draw_from_the_shared_palette(self):
        import plotly.graph_objects as go

        from _shared import CHART_SEQUENCE, base_layout

        fig = base_layout(go.Figure([go.Scatter(x=[1, 2], y=[1, 2])]))
        assert list(fig.layout.colorway) == list(CHART_SEQUENCE)

    def test_chart_gridlines_stay_out_of_the_way(self):
        """Plotly's default grid is dark enough to compete with the data."""
        import plotly.graph_objects as go

        from _shared import base_layout

        fig = base_layout(go.Figure([go.Scatter(x=[1, 2], y=[1, 2])]))
        assert fig.layout.xaxis.showgrid is False
        assert fig.layout.yaxis.gridcolor == "#F1F4F3"

    def test_dollar_formatting_survived_the_restyle(self):
        """The theming block runs after the axis formatting and could quietly
        overwrite it, which would bring back raw floats in the hover."""
        import plotly.graph_objects as go

        from _shared import base_layout

        fig = base_layout(go.Figure([go.Scatter(x=[1, 2], y=[1, 2])]), fmt="dollar")
        assert fig.layout.yaxis.tickprefix == "$"
        assert fig.layout.yaxis.tickformat == ",.0f"
        assert "$" not in str(fig.layout.yaxis.hoverformat or ""), "hover would print $$"

    def test_page_titles_go_through_the_shared_header(self):
        """A stray ``st.title`` renders an unstyled heading with no badge and
        no rule under it, which is exactly the notebook look being removed."""
        offenders = [p.name for p in sorted(VIEWS.glob("*.py"))
                     if "st.title(" in p.read_text()]
        assert not offenders, f"views calling st.title directly: {offenders}"

    def test_the_header_escapes_what_it_interpolates(self):
        """It is raw HTML, and the pages it serves are one edit away from
        passing profile text through it."""
        shared = "\n".join(path.read_text() for path in
                           [APP_DIR / "_shared.py", *sorted((APP_DIR / "ui").glob("*.py"))])
        head = shared[shared.index("def page_setup"):shared.index("def sidebar_profile_summary")]
        assert "html.escape(title)" in head
        assert "html.escape(subtitle)" in head
