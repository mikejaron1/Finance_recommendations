"""Page chrome, disclosure and plan-management controls."""
from __future__ import annotations
import html
import json
import sqlite3
import time
from pathlib import Path
from finrec import storage
from finrec.profile import Profile
from .rendering import st
from .theme import CSS, DISCLAIMER
from .formatting import inline_md, money, money_exact, pct, assumption_value
from .state import *
from . import page_store

def mode_toggle() -> bool:
    """Global simple/advanced switch. Sticky across pages."""
    st.sidebar.toggle(
        "Advanced mode",
        key="advanced_mode",
        help="Simple mode asks for the fewest inputs and infers the rest from your location. "
             "Advanced mode exposes every assumption.",
    )
    return advanced_mode()

def page_setup(title: str, icon: str = "", subtitle: str = "",
               eyebrow: str = "", *, namespace: str | None = None) -> Profile:
    """Standard view header + sidebar. Returns the active profile.

    The header is rendered as one HTML block rather than ``st.title`` plus a
    caption. An emoji sitting inside the H1 ("📊 Dashboard") is the single
    most "internal tool" thing a page can do, and two separate Streamlit
    elements can't share a rule underneath them. Here the icon becomes a
    badge, the title and lede sit beside it, and a hairline closes the group.

    Everything interpolated is escaped: titles and subtitles are static
    today, but this is raw HTML and the profile-driven pages are one small
    edit away from passing user text through it.
    """
    set_page_namespace(namespace or storage.slugify(title))
    st.markdown(CSS, unsafe_allow_html=True)
    profile = get_profile()
    autosave(profile)
    sidebar_profile_summary(profile)
    if title:
        badge = (f"<span class='fp-pagehead-badge' aria-hidden='true'>{html.escape(icon)}</span>"
                 if icon else "")
        kicker = (f"<div class='fp-eyebrow'>{html.escape(eyebrow)}</div>"
                  if eyebrow else "")
        lede = (f"<p class='fp-lede'>{html.escape(subtitle)}</p>"
                if subtitle else "")
        st.markdown(
            "<div class='fp-pagehead'>"
            f"{badge}"
            f"<div class='fp-pagehead-text'>{kicker}<h1>{html.escape(title)}</h1>{lede}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif subtitle:
        st.markdown(f"<p class='fp-subtitle'>{html.escape(subtitle)}</p>",
                    unsafe_allow_html=True)
    return profile

def sidebar_profile_summary(profile: Profile) -> None:
    with st.sidebar:
        st.markdown(
            "<div class='fp-sidecard'><div class='fp-side-label'>Your snapshot</div>"
            f"<div class='fp-side-row'><span>Net worth</span><b>{money(profile.net_worth)}</b></div>"
            f"<div class='fp-side-row'><span>Household income</span><b>{money(profile.household_income)}</b></div>"
            f"<div class='fp-side-row'><span>Savings rate</span><b>{profile.savings_rate:.0%}</b></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, profile.fi_progress_by_retirement)),
                    text=f"{profile.fi_progress_by_retirement:.0%} of what you need by {profile.retirement_age}")
        st.caption(f"Stopping today would take {money(profile.fi_target_now)} — "
                   f"you're {profile.fi_progress:.0%} of the way there.")
        if profile.location:
            st.caption(f"📍 {profile.market().label}")
        st.divider()
        mode_toggle()
        saved_plans_panel()

def saved_plans_panel() -> None:
    """Plan management: what's saved, switch between plans, start over."""
    try:
        plans = storage.list_plans()
    except (OSError, sqlite3.Error, ValueError) as exc:
        plans = []
        st.warning(f"Saved-plan list unavailable: {exc}. Your current session is unchanged.")
    try:
        migration = page_store.page_inputs_migration_status()
    except (OSError, sqlite3.Error, ValueError) as exc:
        migration = {}
        st.warning(f"Page-input recovery status unavailable: {exc}.")
    if migration.get("status") == "corrupt":
        st.warning(
            "An older page-input cache could not be restored. Its original data is "
            "preserved in backups. Financial profiles can still be saved, and new "
            "page inputs are saved separately."
        )
    current = st.session_state.get("plan_name", "My plan")

    with st.expander(f"💾 {current}", expanded=False):
        if migration.get("error"):
            st.caption(f"Page-input recovery details: {migration['error']}")
        failure = st.session_state.get("save_failed")
        if failure:
            st.warning(f"**Not saving.** {failure}\n\nYour numbers still work for this "
                       "session, but they won't be here next time. Export below to keep them.")
            st.button("Retry saving", key="plan_retry", on_click=lambda: save_profile(get_profile()))
            if st.session_state.get("plan_slug"):
                st.button("Reload saved plan (discard this tab's edits)",
                          key="plan_reload", on_click=_reload_current_plan)
        elif st.session_state.get("plan_id") is not None:
            saved_at = st.session_state.get("last_saved_at")
            when = (time.strftime("%H:%M:%S", time.localtime(saved_at)) if saved_at else None)
            st.caption(f"Saved automatically{f' · last save {when}' if when else ''}. "
                       "It'll be here when you come back.")
        else:
            st.caption("New draft — complete setup to create a saved plan.")

        if st.session_state.get("plan_notice"):
            st.warning(st.session_state["plan_notice"])
        if st.session_state.get("plan_recovery_notice"):
            st.warning(st.session_state["plan_recovery_notice"])
        st.text_input("Plan name", value=current, key="plan_name_input",
                      on_change=_rename_current_plan,
                      help="Renames this plan. Use New plan to keep a separate plan.")

        others = [p for p in plans if p.plan_id != st.session_state.get("plan_id")]
        if others:
            labels = {p.slug: p.label for p in others}
            st.selectbox(
                "Switch to another plan", ["—"] + list(labels),
                key="plan_switch",
                format_func=lambda slug: labels.get(slug, "—"),
                on_change=_switch_selected_plan,
                help="Loads that plan's numbers into every page.",
            )

        _history_controls(current)
        _export_control()

        st.button("New plan", key="plan_new", on_click=start_over,
                  help="Begin a separate plan. Existing plans and history are kept.")
        st.button("Start over", key="plan_reset", on_click=start_over,
                     help="Clears this session and returns to the questions. "
                          "Your saved plans are kept.")


def _switch_selected_plan() -> None:
    slug = st.session_state.get("plan_switch", "—")
    if slug != "—":
        switch_plan(slug)
    st.session_state["plan_switch"] = "—"


def _rename_current_plan() -> None:
    name = st.session_state.get("plan_name_input", "").strip()
    if name:
        old = st.session_state.get("plan_name", "My plan")
        st.session_state["plan_name"] = name
        save_profile(get_profile())
        if st.session_state.get("save_failed"):
            st.session_state["plan_name"] = old


def _reload_current_plan() -> None:
    slug = st.session_state.get("plan_slug")
    failure = st.session_state.pop("save_failed", None)
    if not slug or not switch_plan(slug):
        if failure:
            st.session_state["save_failed"] = failure

def _history_controls(current_name: str) -> None:
    """Let the user walk back to an earlier save.

    Every save appends a version rather than overwriting, so this is a real
    undo — including undo of a document import that proposed something wrong,
    or of a number typed into the wrong field an hour ago.
    """
    slug = st.session_state.get("plan_slug")
    if not slug:
        return
    try:
        versions = storage.list_versions(slug, limit=25)
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.warning(f"History unavailable: {exc}. Saved versions have not been changed.")
        return
    if len(versions) < 2:
        return

    with st.popover(f"🕘 History ({len(versions)})", width="stretch"):
        st.caption("Every save is kept. Restoring doesn't delete anything — "
                   "you can always come back.")
        labels = {}
        for entry in versions:
            when = time.strftime("%-d %b, %H:%M", time.localtime(entry["saved_at"]))
            note = f" · {entry['note']}" if entry["note"] else ""
            labels[f"v{entry['version']} — {when} · {entry['summary']}{note}"] = entry["version"]

        picked = st.radio("Restore a previous version", list(labels),
                          key=f"history_{slug}", index=0)
        st.button("↩️ Restore this version", key=f"restore_{slug}",
                  on_click=lambda: restore_plan_version(
                      labels[st.session_state[f"history_{slug}"]]))

def _export_control() -> None:
    """Hand the whole store back as plain JSON.

    Data you can't take with you isn't really yours, and this is also the
    escape hatch if saving ever breaks — the warning above points here.
    """
    if st.button("Prepare backup", key="prepare_backup"):
        export_failed = False
        try:
            payload = storage.export_all()
        except (OSError, sqlite3.Error, ValueError) as exc:
            export_failed = True
            payload = {"format_version": storage.FORMAT_VERSION, "plans": []}
            st.warning(f"Saved history could not be read: {exc}. This backup includes the current session.")
        payload["session_draft"] = {
            "name": st.session_state.get("plan_name", "My plan"),
            "profile": get_profile().to_dict(),
            "page_inputs": dict(st.session_state.get("_sticky", {})),
            "pending_page_inputs": st.session_state.get("_pending_plan_inputs", {}),
            "unsaved": bool(st.session_state.get("save_failed")),
        }
        # Make a failed disk save restorable, not just inspectable JSON.
        draft = payload["session_draft"]
        if export_failed or draft["unsaved"] or not st.session_state.get("plan_id"):
            now = time.time()
            payload["plans"].append({
                "slug": "session-draft", "name": f"{draft['name']} (session draft)",
                "profile": draft["profile"], "saved_at": now, "version": 1,
                "history": [{"version": 1, "payload": json.dumps(draft["profile"]),
                             "saved_at": now, "summary": "Unsaved session backup",
                             "note": "Recovered from the current browser session"}],
            })
        st.session_state["_backup_payload"] = json.dumps(payload, indent=2, default=str)
    if st.session_state.get("_backup_payload"):
        st.download_button(
            "⬇️ Download prepared backup (JSON)",
            data=st.session_state["_backup_payload"],
            file_name=f"finrec-export-{time.strftime('%Y-%m-%d')}.json",
            mime="application/json", width="stretch", key="export_all",
            help="A snapshot taken when you clicked Prepare backup. Prepare again after further edits.",
        )
    _import_control()


def _import_control() -> None:
    uploaded = st.file_uploader(
        "Restore a backup", type=["json"], key="backup_upload",
        on_change=lambda: st.session_state.pop("backup_confirm", None),
    )
    if uploaded is None:
        return
    try:
        payload = json.loads(uploaded.getvalue())
        if not isinstance(payload, dict) or not isinstance(payload.get("plans"), list):
            raise ValueError("Choose a Finance Planner backup with a plans list.")
        preview = storage.import_all(payload, dry_run=True, collision="copy")
        names = [str(p.get("name", "Unnamed plan")) for p in payload["plans"]]
        st.caption(f"Preview: {preview['plans']} saved plans, {preview['versions']} versions. "
                   "Imported plans are kept separately; "
                   "this tab's current plan will not be replaced.")
        if preview.get("corrupt_versions"):
            st.warning(f"{preview['corrupt_versions']} historical snapshots could not be decoded.")
        for name in names:
            st.text(name)
        confirmed = st.checkbox("I reviewed this backup and want to import it", key="backup_confirm")
        if st.button("Import confirmed backup", key="backup_import", disabled=not confirmed):
            storage.import_all(payload, collision="copy")
            st.session_state["plan_notice"] = "Backup imported. Choose an imported plan from the plan switcher."
            st.rerun()
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        st.error(f"Backup was not imported: {exc}")

def page_footer() -> None:
    flush_sticky()
    if st.session_state.get("sticky_save_failed"):
        st.warning(f"Page inputs are only saved in this session: "
                   f"{st.session_state['sticky_save_failed']}. They will be retried.")
    if st.session_state.get("save_failed"):
        st.warning(f"Latest profile changes are not saved: {st.session_state['save_failed']}. "
                   "Use Prepare backup to keep this session, or retry saving.")
    st.markdown(f"<div class='fp-foot'>{DISCLAIMER}</div>", unsafe_allow_html=True)

def answer(headline: str, sub: str = "", tone: str = "good", label: str = "The answer") -> None:
    """The single sentence the user came for. Always render this first."""
    css_class = {"good": "", "warn": " warn", "bad": " bad"}.get(tone, "")
    sub_html = f"<div class='fp-sub'>{inline_md(sub)}</div>" if sub else ""
    st.markdown(
        f"<div class='fp-answer{css_class}'><div class='fp-label'>{inline_md(label)}</div>"
        f"<div class='fp-headline'>{inline_md(headline)}</div>{sub_html}</div>",
        unsafe_allow_html=True,
    )

def assumptions_panel(assumptions: list[dict], title: str = "What we filled in for you",
                      caption: str | None = None) -> None:
    """Show every auto-filled value and where it came from.

    Auto-filling inputs is only acceptable if the user can see what was
    assumed. This panel is the price of admission for that convenience.

    ``caption`` overrides the default blurb for panels whose values are not
    location lookups — claiming they were would be worse than saying nothing.
    """
    if not assumptions:
        return
    with st.expander(f"🔎 {title} ({len(assumptions)} values)"):
        st.caption(
            caption
            or "Looked up from your location. Every one is an estimate you can override in Advanced mode."
        )
        for item in assumptions:
            shown = assumption_value(item.get("value"), item.get("unit"))
            st.markdown(
                f"<strong>{html.escape(str(item.get('name', 'Assumption')))}</strong> "
                f"— {html.escape(shown)}<br>"
                f"<span class='fp-source'>{html.escape(str(item.get('source', 'Unknown source')))}</span>",
                unsafe_allow_html=True,
            )


def confidence_panel(profile: Profile, *, expanded: bool = False) -> None:
    """Explain provenance without inventing a numeric confidence score."""
    provenance = getattr(profile, "input_provenance", {}) or {}
    if not isinstance(provenance, dict):
        provenance = {}
    important = {
        "monthly_spending": "Current monthly spending",
        "desired_retirement_spending": "Retirement spending target",
        "annual_401k_contribution": "Annual retirement contributions",
        "cash": "Cash available",
    }
    labels = {
        "entered": "Entered by you", "imported": "Imported — verify against the source",
        "estimated": "Estimated — review before relying on it",
        "outdated": "Outdated — update needed", "unknown": "Source not confirmed",
    }
    rows, needs_review = [], []
    for field in dict.fromkeys([*important, *provenance]):
        record = provenance.get(field, {})
        # Legacy contribution metadata used the unseparated spelling.
        if field == "annual_401k_contribution" and not record:
            record = provenance.get("annual401k", {})
        if not isinstance(record, (str, dict)):
            record = {}
        status = record if isinstance(record, str) else record.get("kind", record.get("status", "unknown"))
        status = status if isinstance(status, str) and status in labels else "unknown"
        if field in important and getattr(profile, field, None) is None:
            status = "unknown"
        name = important.get(field, str(field).replace("_", " ").capitalize())
        if field in important and status in ("unknown", "estimated", "outdated"):
            needs_review.append(name)
        source = record.get("source", "") if isinstance(record, dict) else ""
        if isinstance(record, dict) and record.get("as_of"):
            source = f"{source} · as of {record['as_of']}".strip(" ·")
        rows.append((name, labels[status], source))
    if needs_review:
        st.caption(f"{len(needs_review)} high-impact inputs need review — "
                   "open Input confidence & sources for details.")
    with st.expander("Input confidence & sources", expanded=expanded):
        st.caption("Projection precision is not certainty. Entered and imported figures "
                   "are not independently verified; future outcomes remain estimates.")
        if needs_review:
            st.warning("High-impact inputs to confirm: " + ", ".join(needs_review) + ".")
        for name, status, source in rows:
            suffix = f" · {source}" if source else ""
            st.markdown(f"<strong>{html.escape(name)}</strong> — "
                        f"{html.escape(status + suffix)}", unsafe_allow_html=True)

def advanced_section(title: str = "Advanced settings"):
    """Container for detail. Expanded only when the user is in advanced mode."""
    return st.expander(f"⚙️ {title}", expanded=advanced_mode())


provenance_panel = confidence_panel

def nav_card(title: str, description: str, page: str) -> None:
    stem = Path(page).stem
    href = "./" if stem == "home" else "./" + stem
    st.markdown(f"<a class='fp-card fp-card-link' href='{html.escape(href, quote=True)}' "
                f"target='_self'><h4>{inline_md(title)}</h4>"
                f"<p>{inline_md(description)}</p></a>",
                unsafe_allow_html=True)

def verdict(text: str, kind: str = "info") -> None:
    """Render the headline answer for a page."""
    {"success": st.success, "warning": st.warning, "error": st.error, "info": st.info}[kind](f"**{text}**")

def assumption_note(text: str) -> None:
    st.caption(f"ℹ️ {text}")
