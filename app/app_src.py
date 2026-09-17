"""
app_src.py
----------
advQMcalc_test — QM Crystal Workflow.
Thin UI over the existing backend. Handles:
  1. Run new QE crystal calculations from uploaded CIF files (via SLURM)
  2. Monitor / resume existing workflow runs
  3. Aggregate per-CIF results into a summary table

Built on ipywidgets (ipywidgets 8 / py3.12 stack).

Layout: a dark left sidebar (Dashboard / New Run / Runs / Results) next to a
single main content area whose contents are swapped on navigation — the
closest approximation of a multi-page dashboard achievable inside Voila.
"""

import json
import logging
import sys
import os
import threading
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, display

# ── Locate helpers ────────────────────────────────────────────────────────────
_APP_DIR = None
for _root in [Path.cwd(), Path.cwd().parent, *Path.cwd().parents]:
    _candidate = _root / "app" / "helpers.py"
    if _candidate.is_file():
        _APP_DIR = _root / "app"
        break
if _APP_DIR is not None:
    sys.path.insert(0, str(_APP_DIR))
    sys.path.insert(0, str(_APP_DIR.parent))

from helpers import (
    UPLOAD_DIR,
    build_namespace,
    attach_log_handler,
    save_uploads,
    enumerate_runs,
    load_run_state,
    resume_target,
    status_badge_html,
    status_category,
    _process_single_cif,
    _aggregate_crystal_results,
    _query_slurm_job,
    _runs_dir_for_cif,
)

DEFAULT_RUNS_DIR = "/mnt/own6a/zoya/QM_Workflow/runs"


def _run_log(msg: str) -> None:
    logging.getLogger("advQMcalc_test").info(msg)


class _Nav:
    """Small indirection so page builders can trigger navigation before the
    router itself exists yet (pages are built before the sidebar wiring)."""

    def __init__(self):
        self.show = lambda key: None


def _nav_button(label: str) -> widgets.Button:
    b = widgets.Button(description=label, layout=widgets.Layout(width="auto"))
    b.add_class("advqm-navbtn")
    return b


def _fmt_energy(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "–"


def _fmt(v) -> str:
    return str(v) if v not in (None, "") else "–"


# ── Dashboard page ───────────────────────────────────────────────────────────
def _build_dashboard_page(nav: _Nav):
    header = widgets.HTML(
        '<div class="advqm-page-header">'
        '<div><div class="advqm-page-title">Dashboard</div>'
        '<div class="advqm-page-sub">Overview of your Quantum ESPRESSO calculation workflow</div></div>'
        "</div>"
    )
    new_run_btn = widgets.Button(description="New Run", button_style="success", icon="plus")
    new_run_btn.on_click(lambda _: nav.show("new_run"))

    stats_out = widgets.Output()
    recent_out = widgets.Output()
    refresh_btn = widgets.Button(description="Refresh", icon="refresh")

    def render(_=None):
        base = Path(DEFAULT_RUNS_DIR)
        stats_out.clear_output()
        recent_out.clear_output()
        rows = []
        if base.is_dir():
            try:
                rows = _aggregate_crystal_results(base, only_successful=False)
            except Exception:
                rows = []

        n_success = sum(1 for r in rows if status_category(r.get("crystal_status")) == "success")
        n_progress = sum(
            1 for r in rows if status_category(r.get("crystal_status")) in ("info", "warning")
        )
        n_failed = sum(1 for r in rows if status_category(r.get("crystal_status")) == "danger")

        with stats_out:
            display(
                HTML(
                    f"""
                    <div class="advqm-stats-row">
                      <div class="advqm-stat-card">
                        <div class="advqm-stat-num success">{n_success}</div>
                        <div class="advqm-stat-label">Completed</div>
                      </div>
                      <div class="advqm-stat-card">
                        <div class="advqm-stat-num info">{n_progress}</div>
                        <div class="advqm-stat-label">In Progress</div>
                      </div>
                      <div class="advqm-stat-card">
                        <div class="advqm-stat-num danger">{n_failed}</div>
                        <div class="advqm-stat-label">Failed</div>
                      </div>
                    </div>
                    """
                )
            )

        with recent_out:
            if not rows:
                display(
                    HTML(
                        '<div class="advqm-card"><div class="advqm-card-title">Recent Runs</div>'
                        '<div class="advqm-muted">No runs found yet in '
                        f"{base}.</div></div>"
                    )
                )
                return
            rows_sorted = sorted(rows, key=lambda r: r.get("run_id") or "", reverse=True)[:5]
            items = "".join(
                f'<div class="advqm-list-row"><div><b>{_fmt(r.get("cif"))}</b>'
                f'<div class="advqm-muted">{_fmt(r.get("run_id"))} &middot; job {_fmt(r.get("job_id"))}</div></div>'
                f"{status_badge_html(r.get('crystal_status'))}</div>"
                for r in rows_sorted
            )
            display(
                HTML(
                    f'<div class="advqm-card"><div class="advqm-card-title">Recent Runs</div>{items}</div>'
                )
            )

    refresh_btn.on_click(render)
    render()

    return widgets.VBox([header, widgets.HBox([new_run_btn, refresh_btn]), stats_out, recent_out])


# ── New Run page ─────────────────────────────────────────────────────────────
def _build_new_run_page(nav: _Nav):
    upload = widgets.FileUpload(accept=".cif", multiple=True)
    calc_type_w = widgets.Dropdown(
        options=["scf", "relax", "vc-relax", "nscf", "bands"], value="scf",
        layout=widgets.Layout(width="100%"),
    )
    runs_dir_w = widgets.Text(value=DEFAULT_RUNS_DIR, layout=widgets.Layout(width="100%"))
    log_name_w = widgets.Text(value="advQMcalc.log", layout=widgets.Layout(width="100%"))
    force_pp_w = widgets.Checkbox(value=False, description="Force pseudopotential cleanup")
    kpoints_w = widgets.Text(
        value="3 3 2 0 0 0", placeholder="e.g. 3 3 2 0 0 0 (auto if empty)",
        layout=widgets.Layout(width="100%"),
    )
    ksep_w = widgets.FloatText(value=0.03, layout=widgets.Layout(width="100%"))
    pseudo_dir_w = widgets.Text(
        value="/mnt/own6a/zoya/QM_Workflow/pseudos", placeholder="/path/to/pseudo",
        layout=widgets.Layout(width="100%"),
    )
    pp_map_w = widgets.Text(value="", placeholder="/path/to/pp_map.json", layout=widgets.Layout(width="100%"))
    walltime_w = widgets.Text(value="12:00:00", layout=widgets.Layout(width="100%"))
    ntasks_w = widgets.IntText(value=8, layout=widgets.Layout(width="100%"))
    qe_cmd_w = widgets.Text(value="pw.x", layout=widgets.Layout(width="100%"))

    run_btn = widgets.Button(description="Submit Run", button_style="success", icon="play")
    reset_btn = widgets.Button(description="Cancel", icon="undo")
    sample_btn = widgets.Button(description="Try Sample", icon="refresh")
    run_log = widgets.Output()
    attach_log_handler(run_log)

    def on_sample_clicked(b: widgets.Button) -> None:
        calc_type_w.value = "scf"
        runs_dir_w.value = DEFAULT_RUNS_DIR
        log_name_w.value = "advQMcalc.log"
        kpoints_w.value = "3 3 2 0 0 0"
        ksep_w.value = 0.03
        pseudo_dir_w.value = "/mnt/own6a/zoya/QM_Workflow/pseudos"
        pp_map_w.value = ""
        walltime_w.value = "12:00:00"
        ntasks_w.value = 8
        qe_cmd_w.value = "pw.x"
        force_pp_w.value = False
        run_log.clear_output()
        run_log.append_stdout("Sample values loaded. Upload a .cif file and click Submit Run.\n")

    sample_btn.on_click(on_sample_clicked)

    def on_run_clicked(b: widgets.Button) -> None:
        run_btn.disabled = True
        try:
            staged = save_uploads(upload, UPLOAD_DIR)
        except Exception as e:
            _run_log(f"Upload error: {e}")
            run_btn.disabled = False
            return
        if not staged:
            _run_log("No CIF files selected for upload.")
            run_btn.disabled = False
            return
        base_runs = Path(runs_dir_w.value)
        multi = len(staged) > 1
        common = dict(
            calc_type=calc_type_w.value or "scf",
            runs_dir=str(base_runs),
            log_name=log_name_w.value,
            force_pp_cleanup=force_pp_w.value,
            pseudo_dir=pseudo_dir_w.value.strip() or None,
            pp_map=pp_map_w.value.strip() or None,
            slurm_walltime=walltime_w.value,
            slurm_ntasks=int(ntasks_w.value),
            qe_command=qe_cmd_w.value,
            kpoints=kpoints_w.value.strip() or None,
            kpoint_separation=float(ksep_w.value),
        )

        def work() -> None:
            _run_log(f"Starting run for {len(staged)} uploaded CIF file(s)")
            for cif in staged:
                ns = build_namespace(cif=[str(cif)], **common)
                runs_dir = _runs_dir_for_cif(base_runs, cif, multi)
                try:
                    _process_single_cif(cif, runs_dir, ns)
                except Exception as e:
                    _run_log(f"Run failed for {cif.name}: {e}")
            _run_log("Workflow batch finished.")

        def on_done() -> None:
            run_btn.disabled = False

        t = threading.Thread(target=work, daemon=True)
        t.start()

        def _check() -> None:
            while t.is_alive():
                pass
            on_done()

        threading.Thread(target=_check, daemon=True).start()

    def on_reset_clicked(b: widgets.Button) -> None:
        upload.value = {}
        run_log.clear_output()
        run_log.append_stdout("Ready.\n")

    run_btn.on_click(on_run_clicked)
    reset_btn.on_click(on_reset_clicked)

    header = widgets.HTML(
        '<div class="advqm-page-header">'
        '<div><div class="advqm-page-title">New Calculation Run</div>'
        '<div class="advqm-page-sub">Configure and submit a Quantum ESPRESSO workflow</div></div>'
        "</div>"
    )

    upload_card = widgets.VBox(
        [widgets.HTML('<div class="advqm-card-title">\U0001F4C4 Crystal Structure Files</div>'), upload],
    )
    upload_card.add_class("advqm-card")

    params_card = widgets.VBox(
        [
            widgets.HTML('<div class="advqm-card-title">⚙️ Calculation Parameters</div>'),
            widgets.HBox(
                [
                    widgets.VBox([widgets.HTML("<b>Calculation Type</b>"), calc_type_w],
                                 layout=widgets.Layout(width="calc(50% - 8px)")),
                    widgets.VBox([widgets.HTML("<b>Pseudopotential Directory</b>"), pseudo_dir_w],
                                 layout=widgets.Layout(width="calc(50% - 8px)")),
                ],
                layout=widgets.Layout(width="100%", justify_content="space-between"),
            ),
            widgets.HBox(
                [
                    widgets.VBox([widgets.HTML("<b>K-Points Grid (optional)</b>"), kpoints_w],
                                 layout=widgets.Layout(width="calc(50% - 8px)")),
                    widgets.VBox([widgets.HTML("<b>K-Point Separation</b>"), ksep_w],
                                 layout=widgets.Layout(width="calc(50% - 8px)")),
                ],
                layout=widgets.Layout(width="100%", justify_content="space-between"),
            ),
        ]
    )
    params_card.add_class("advqm-card")

    slurm_card = widgets.VBox(
        [
            widgets.HTML('<div class="advqm-card-title">\U0001F5A5️ SLURM Configuration</div>'),
            widgets.HBox(
                [
                    widgets.VBox([widgets.HTML("<b>Walltime</b>"), walltime_w],
                                 layout=widgets.Layout(width="calc(33.33% - 10px)")),
                    widgets.VBox([widgets.HTML("<b>Tasks</b>"), ntasks_w],
                                 layout=widgets.Layout(width="calc(33.33% - 10px)")),
                    widgets.VBox([widgets.HTML("<b>QE Command</b>"), qe_cmd_w],
                                 layout=widgets.Layout(width="calc(33.33% - 10px)")),
                ],
                layout=widgets.Layout(width="100%", justify_content="space-between"),
            ),
        ]
    )
    slurm_card.add_class("advqm-card")

    advanced = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.HTML("<b>Runs Directory</b>"), runs_dir_w,
                    widgets.HTML("<b>Log File Name</b>"), log_name_w,
                    widgets.HTML("<b>PP Map (optional)</b>"), pp_map_w,
                    force_pp_w,
                ]
            )
        ]
    )
    advanced.set_title(0, "Advanced settings")
    advanced.selected_index = None

    actions = widgets.HBox([sample_btn, run_btn, reset_btn])

    return widgets.VBox(
        [header, upload_card, params_card, slurm_card, advanced, actions, run_log]
    )


# ── Runs page (table + monitor/resume) ──────────────────────────────────────
_ROW_COL_WIDTHS = ["27%", "15%", "11%", "15%", "21%", "10%"]


def _run_row_widget(r: dict, on_view) -> widgets.HBox:
    cif_path = r.get("cif")
    cif_name = Path(cif_path).name if cif_path else _fmt(cif_path)
    cif_html = widgets.HTML(
        f'<span title="{_fmt(cif_path)}">{cif_name}</span>',
        layout=widgets.Layout(width=_ROW_COL_WIDTHS[0], overflow="hidden"),
    )
    cif_html.add_class("advqm-rowcell-cif")
    cells = [
        cif_html,
        widgets.HTML(_fmt(r.get("run_id")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[1])),
        widgets.HTML(_fmt(r.get("job_id")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[2])),
        widgets.HTML(_fmt_energy(r.get("energy_ry")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[3])),
        widgets.HTML(status_badge_html(r.get("crystal_status")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[4])),
    ]
    view_btn = widgets.Button(description="View", icon="eye", layout=widgets.Layout(width=_ROW_COL_WIDTHS[5]))
    view_btn.on_click(lambda _b, row=r: on_view(row))
    row_box = widgets.HBox(cells + [view_btn], layout=widgets.Layout(width="100%"))
    row_box.add_class("advqm-rowlist-row")
    return row_box


def _build_runs_page(nav: _Nav):
    header = widgets.HTML(
        '<div class="advqm-page-header">'
        '<div><div class="advqm-page-title">Runs</div>'
        '<div class="advqm-page-sub">All workflow runs and their current status — click View for details</div></div>'
        "</div>"
    )
    new_run_btn = widgets.Button(description="New Run", button_style="success", icon="plus")
    new_run_btn.on_click(lambda _: nav.show("new_run"))

    runs_dir_w = widgets.Text(value=DEFAULT_RUNS_DIR, layout=widgets.Layout(width="60%"))
    search_w = widgets.Text(placeholder="Search by CIF name or job ID...", layout=widgets.Layout(width="60%"))
    refresh_btn = widgets.Button(description="Refresh", icon="refresh")

    col_head = widgets.HTML(
        '<div class="advqm-rowlist-head">'
        + "".join(
            f'<span style="width:{w}; display:inline-block;">{label}</span>'
            for w, label in zip(
                _ROW_COL_WIDTHS,
                ["CIF File", "Run ID", "Job ID", "Energy (Ry)", "Status", ""],
            )
        )
        + "</div>"
    )
    rows_box = widgets.VBox([])
    rows_wrap = widgets.VBox([col_head, rows_box])
    rows_wrap.add_class("advqm-card")
    rows_wrap.add_class("advqm-rowlist")

    detail_out = widgets.Output()
    current_rows = []

    def show_details(r: dict):
        detail_out.clear_output()
        run_path = Path(r["run_path"]) if r.get("run_path") else None
        job_status = widgets.Label(value="—")
        query_btn = widgets.Button(description="Query SLURM", icon="search")
        resume_btn = widgets.Button(description="Resume", button_style="success", icon="play")
        raw_out = widgets.Output()

        def on_query(_=None):
            jid = r.get("job_id")
            if not jid:
                job_status.value = "No SLURM job_id recorded for this run"
                return
            try:
                s = _query_slurm_job(str(jid))
                job_status.value = f"Job {jid}: {s}"
            except Exception as e:
                job_status.value = f"Query failed: {e}"

        def on_resume(_=None):
            if run_path is None:
                job_status.value = "No run path available to resume"
                return
            base = Path(runs_dir_w.value)
            label = run_path.parent.name
            try:
                runs_dir, run_id = resume_target(base, label, run_path)
            except Exception as e:
                job_status.value = f"Resume setup failed: {e}"
                return
            st = load_run_state(run_path) or {}
            kpoints = st.get("tasks", {}).get("crystal", {}).get("kpoints")
            ns = build_namespace(
                resume=True,
                run_id=run_id,
                runs_dir=str(runs_dir),
                calc_type=st.get("calc_type") or "scf",
                kpoints=kpoints,
            )
            resume_btn.disabled = True

            def work():
                try:
                    _process_single_cif(run_path, runs_dir, ns)
                except Exception as e:
                    _run_log(f"Resume failed: {e}")
                finally:
                    resume_btn.disabled = False

            t = threading.Thread(target=work, daemon=True)
            t.start()

            def _check():
                while t.is_alive():
                    pass
                resume_btn.disabled = False

            threading.Thread(target=_check, daemon=True).start()

        def on_raw_state(_=None):
            raw_out.clear_output()
            with raw_out:
                if run_path is None:
                    print("No run path available.")
                    return
                st = load_run_state(run_path)
                if st is None:
                    print(f"No state.json found for {run_path}")
                    return
                display(HTML(f"<pre>{json.dumps(st, indent=2)}</pre>"))

        query_btn.on_click(on_query)
        resume_btn.on_click(on_resume)

        raw_btn = widgets.Button(description="Show raw state.json", icon="code")
        raw_btn.on_click(on_raw_state)

        energy = r.get("energy_ry")
        energy_block = (
            f'<div class="advqm-detail-energy">{_fmt_energy(energy)} <span class="advqm-detail-energy-unit">Ry</span></div>'
            if isinstance(energy, (int, float))
            else '<div class="advqm-detail-energy advqm-muted" style="font-size:1.3rem;">Not available yet — the calculation may still be running, or needs Resume to extract.</div>'
        )

        with detail_out:
            display(
                HTML(
                    f"""
                    <div class="advqm-card">
                      <div class="advqm-card-title">📄 {_fmt(r.get('cif'))}</div>
                      <div class="advqm-detail-energy-label">Total Energy</div>
                      {energy_block}
                      <div class="advqm-detail-grid">
                        <div><b>Run ID</b><br>{_fmt(r.get('run_id'))}</div>
                        <div><b>Job ID</b><br>{_fmt(r.get('job_id'))}</div>
                        <div><b>Status</b><br>{status_badge_html(r.get('crystal_status'))}</div>
                        <div><b>K-Points</b><br>{_fmt(r.get('kpoints'))}</div>
                        <div><b>Pseudopotential Dir</b><br>{_fmt(r.get('pseudo_dir'))}</div>
                        <div><b>SLURM State</b><br>{_fmt(r.get('slurm_state'))}</div>
                      </div>
                    </div>
                    """
                )
            )
            display(widgets.HBox([job_status, query_btn, resume_btn, raw_btn]))
            display(raw_out)

    def render_rows(*_):
        filt = (search_w.value or "").strip().lower()
        rows = [
            r
            for r in current_rows
            if not filt
            or filt in str(r.get("cif", "")).lower()
            or filt in str(r.get("job_id", "")).lower()
            or filt in str(r.get("run_id", "")).lower()
        ]
        if not rows:
            empty = widgets.HTML('<div class="advqm-muted" style="padding:16px;">No runs found.</div>')
            rows_box.children = [empty]
            return
        rows_box.children = [_run_row_widget(r, show_details) for r in rows]

    def refresh(_=None):
        nonlocal current_rows
        base = Path(runs_dir_w.value)
        if not base.is_dir():
            current_rows = []
            rows_box.children = [
                widgets.HTML(f'<div class="advqm-muted" style="padding:16px;">Runs dir does not exist: {base}</div>')
            ]
            return
        try:
            current_rows = _aggregate_crystal_results(base, only_successful=False)
        except Exception as e:
            current_rows = []
            rows_box.children = [
                widgets.HTML(f'<div class="advqm-muted" style="padding:16px;">Aggregation failed: {e}</div>')
            ]
            return
        render_rows()

    refresh_btn.on_click(refresh)
    search_w.observe(render_rows, names="value")

    refresh()

    return widgets.VBox(
        [
            header,
            widgets.HBox([new_run_btn]),
            widgets.HBox([runs_dir_w, refresh_btn]),
            search_w,
            rows_wrap,
            detail_out,
        ]
    )


# ── Results / Aggregation page ───────────────────────────────────────────────
def _build_results_page(nav: _Nav):
    header = widgets.HTML(
        '<div class="advqm-page-header">'
        '<div><div class="advqm-page-title">Results</div></div>'
        "</div>"
    )
    return widgets.VBox([header])


# ── App factory ──────────────────────────────────────────────────────────────
def app(app_mode: bool = True, debug: bool = False):
    """
    Build the advQMcalc_test app shell.

    Returns:
        HubApp instance with .form ready to display
    """
    from template_widgets import HubApp

    my_app = HubApp(app_name="advQMcalc_test")

    nav = _Nav()

    pages = {
        "dashboard": _build_dashboard_page(nav),
        "new_run": _build_new_run_page(nav),
        "runs": _build_runs_page(nav),
        "results": _build_results_page(nav),
    }

    nav_specs = [
        ("dashboard", "Dashboard", "th-large"),
        ("new_run", "New Run", "plus"),
        ("runs", "Runs", "list"),
        ("results", "Results", "bar-chart"),
    ]

    nav_buttons = {}
    for key, label, icon in nav_specs:
        b = _nav_button(label)
        b.icon = icon
        nav_buttons[key] = b

    main_area = widgets.VBox([])
    main_area.add_class("advqm-main")

    def show_page(key: str) -> None:
        for k, b in nav_buttons.items():
            if k == key:
                b.add_class("advqm-navbtn-active")
            else:
                b.remove_class("advqm-navbtn-active")
        main_area.children = [pages[key]]
        if key == "dashboard":
            # rebuild so the stats reflect any runs created since page construction
            pages["dashboard"] = _build_dashboard_page(nav)
            main_area.children = [pages["dashboard"]]
        elif key == "runs":
            pages["runs"] = _build_runs_page(nav)
            main_area.children = [pages["runs"]]

    nav.show = show_page

    for key, b in nav_buttons.items():
        b.on_click(lambda _btn, k=key: show_page(k))

    sidebar = widgets.VBox(list(nav_buttons.values()))
    sidebar.add_class("advqm-sidebar")

    shell = widgets.HBox([sidebar, main_area])
    shell.add_class("advqm-shell")

    show_page("dashboard")

    my_app.content.children = [shell]
    return my_app
