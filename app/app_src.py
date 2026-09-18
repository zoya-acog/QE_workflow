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

import base64
import io
import json
import logging
import re
import sys
import os
import threading
from pathlib import Path

import ipywidgets as widgets
from IPython.display import HTML, display

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def _stamp_submitted_by(runs_dir: Path, submitted_by: str) -> None:
    """Record who submitted the most-recently-created run under runs_dir.

    Frontend-only enrichment of state.json (no cli.py changes) — mirrors the
    pattern already used to add calc_type to the Runs table via a post-hoc
    load_run_state read, just as a write this time.
    """
    try:
        pairs = enumerate_runs(runs_dir)
        if not pairs:
            return
        _, latest_run_path = max(pairs, key=lambda p: p[1].name)
        state_file = latest_run_path / "state.json"
        state = json.loads(state_file.read_text())
        state["submitted_by"] = submitted_by or None
        state_file.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


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
        options=["scf", "relax", "vc-relax"], value="scf",
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
    submitted_by_w = widgets.Text(value="", placeholder="Your name/initials", layout=widgets.Layout(width="100%"))

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
        submitted_by_w.value = ""
        run_log.clear_output()
        run_log.append_stdout("Sample values loaded. Upload a .cif file and click Submit Run.\n")

    sample_btn.on_click(on_sample_clicked)

    def on_run_clicked(b: widgets.Button) -> None:
        if not submitted_by_w.value.strip():
            _run_log("Submitted By is required — enter your name/initials before submitting.")
            return
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

        submitted_by = submitted_by_w.value.strip()

        def work() -> None:
            _run_log(f"Starting run for {len(staged)} uploaded CIF file(s)")
            for cif in staged:
                ns = build_namespace(cif=[str(cif)], **common)
                runs_dir = _runs_dir_for_cif(base_runs, cif, multi)
                try:
                    _process_single_cif(cif, runs_dir, ns)
                    _stamp_submitted_by(runs_dir, submitted_by)
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
            widgets.VBox([widgets.HTML("<b>Submitted By *</b>"), submitted_by_w]),
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
_ROW_COL_WIDTHS = ["26%", "15%", "11%", "14%", "16%", "18%"]
_ROW_COL_LABELS = ["Cif file", "Run id", "Job id", "Calculation type", "Energy (ry)", "Status"]


def _run_row_widget(r: dict) -> widgets.HBox:
    cif_path = r.get("cif")
    cif_name = Path(cif_path).name if cif_path else _fmt(cif_path)
    cif_html = widgets.HTML(
        f'<span title="{_fmt(cif_path)}">{cif_name}</span>',
        layout=widgets.Layout(width=_ROW_COL_WIDTHS[0]),
    )
    cif_html.add_class("advqm-rowcell-cif")
    cells = [
        cif_html,
        widgets.HTML(_fmt(r.get("run_id")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[1])),
        widgets.HTML(_fmt(r.get("job_id")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[2])),
        widgets.HTML(_fmt(r.get("calc_type")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[3])),
        widgets.HTML(_fmt_energy(r.get("energy_ry")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[4])),
        widgets.HTML(status_badge_html(r.get("crystal_status")), layout=widgets.Layout(width=_ROW_COL_WIDTHS[5])),
    ]
    row_box = widgets.HBox(cells, layout=widgets.Layout(width="100%"))
    row_box.add_class("advqm-rowlist-row")
    return row_box


def _build_runs_page(nav: _Nav):
    header = widgets.HTML(
        '<div class="advqm-page-header">'
        '<div><div class="advqm-page-title">Runs</div>'
        '<div class="advqm-page-sub">All workflow runs and their current status</div></div>'
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
            f'<span style="width:{w};">{label}</span>'
            for w, label in zip(_ROW_COL_WIDTHS, _ROW_COL_LABELS)
        )
        + "</div>"
    )
    rows_box = widgets.VBox([])
    rows_wrap = widgets.VBox([col_head, rows_box])
    rows_wrap.add_class("advqm-card")
    rows_wrap.add_class("advqm-rowlist")

    current_rows = []

    # ── Monitor / Resume (kept as a utility, independent of the row list) ──
    mon_drop = widgets.Dropdown(layout=widgets.Layout(width="100%"))
    job_status = widgets.Label(value="—")
    query_btn = widgets.Button(description="Query SLURM", icon="search")
    resume_btn = widgets.Button(description="Resume", button_style="success", icon="play")
    state_out = widgets.Output()

    def _refresh_monitor_options():
        options = [
            (f"{_fmt(r.get('cif'))} @ {_fmt(r.get('run_id'))}", Path(r["run_path"]))
            for r in current_rows
            if r.get("run_path")
        ]
        mon_drop.options = options
        if options:
            mon_drop.value = options[0][1]

    def show_state(_=None):
        state_out.clear_output()
        rp = mon_drop.value
        with state_out:
            if not rp:
                return
            st = load_run_state(rp)
            if st is None:
                print(f"No state.json found for {rp}")
                return
            display(HTML(f"<pre>{json.dumps(st, indent=2)}</pre>"))

    def on_query(_=None):
        rp = mon_drop.value
        if not rp:
            return
        st = load_run_state(rp) or {}
        jid = st.get("tasks", {}).get("crystal", {}).get("job_id")
        if not jid:
            job_status.value = "No SLURM job_id recorded for this run"
            return
        try:
            s = _query_slurm_job(str(jid))
            job_status.value = f"Job {jid}: {s}"
        except Exception as e:
            job_status.value = f"Query failed: {e}"

    def on_resume(_=None):
        rp = mon_drop.value
        if not rp:
            return
        label = rp.parent.name
        base = Path(runs_dir_w.value)
        try:
            runs_dir, run_id = resume_target(base, label, rp)
        except Exception as e:
            job_status.value = f"Resume setup failed: {e}"
            return
        st = load_run_state(rp) or {}
        kpoints = st.get("tasks", {}).get("crystal", {}).get("kpoints")
        ns = build_namespace(
            resume=True, run_id=run_id, runs_dir=str(runs_dir),
            calc_type=st.get("calc_type") or "scf", kpoints=kpoints,
        )
        resume_btn.disabled = True

        def work():
            try:
                _process_single_cif(rp, runs_dir, ns)
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

    mon_drop.observe(show_state, names="value")
    query_btn.on_click(on_query)
    resume_btn.on_click(on_resume)

    monitor_body = widgets.VBox([mon_drop, widgets.HBox([job_status, query_btn, resume_btn]), state_out])
    monitor_acc = widgets.Accordion(children=[monitor_body])
    monitor_acc.set_title(0, "Monitor & Resume a run")
    monitor_acc.selected_index = None

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
        rows_box.children = [_run_row_widget(r) for r in rows]

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
            rows = _aggregate_crystal_results(base, only_successful=False)
        except Exception as e:
            current_rows = []
            rows_box.children = [
                widgets.HTML(f'<div class="advqm-muted" style="padding:16px;">Aggregation failed: {e}</div>')
            ]
            return
        for r in rows:
            st = load_run_state(Path(r["run_path"])) if r.get("run_path") else None
            r["calc_type"] = (st or {}).get("calc_type")
        current_rows = rows
        _refresh_monitor_options()
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
            monitor_acc,
        ]
    )


def _parse_scf_history(out_path: Path) -> dict:
    """Parse SCF convergence history out of a QE .out file.

    Real .out files under this project have been observed with every line
    duplicated (interleaved sub-run writes), so this scans in file order and
    plots by occurrence index rather than trusting "iteration #" as a unique
    key — the same "don't assume uniqueness" approach already used by
    _extract_qe_total_energy for the total-energy line.
    """
    text = out_path.read_text(errors="ignore")
    accuracy = [float(m.group(1)) for m in re.finditer(r"estimated scf accuracy\s*<\s*([\d.eE+-]+)\s*Ry", text)]
    energies = [float(m.group(1)) for m in re.finditer(r"total energy\s*=\s*([-\d.]+)\s*Ry", text)]
    converged = "convergence has been achieved" in text
    job_done = "JOB DONE" in text
    return {
        "accuracy": accuracy,
        "final_energy": energies[-1] if energies else None,
        "converged": converged,
        "job_done": job_done,
    }


def _render_scf_chart(history: dict) -> str:
    accuracy = history.get("accuracy") or []
    if not accuracy:
        return '<div class="advqm-muted" style="padding:8px 0;">No SCF accuracy values found in this run\'s output.</div>'

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = list(range(1, len(accuracy) + 1))
    ax.plot(x, accuracy, color="#1565c0", linewidth=1.8, marker="o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("SCF step")
    ax.set_ylabel("Estimated accuracy (Ry)")
    ax.set_title("SCF convergence history")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%; border-radius:8px;">'


# ── Results / Aggregation page ───────────────────────────────────────────────
def _build_results_page(nav: _Nav):
    header = widgets.HTML(
        '<div class="advqm-page-header">'
        '<div><div class="advqm-page-title">Results</div>'
        '<div class="advqm-page-sub">Pick a user, load a run, and view its SCF convergence</div></div>'
        "</div>"
    )

    user_w = widgets.Dropdown(layout=widgets.Layout(width="calc(50% - 8px)"))
    run_w = widgets.Dropdown(layout=widgets.Layout(width="calc(50% - 8px)"))
    load_btn = widgets.Button(description="Load Results", button_style="primary", icon="line-chart")
    result_out = widgets.Output()

    all_rows: list[dict] = []

    def _load_all_rows():
        nonlocal all_rows
        base = Path(DEFAULT_RUNS_DIR)
        rows = []
        if base.is_dir():
            try:
                rows = _aggregate_crystal_results(base, only_successful=False)
            except Exception:
                rows = []
        for r in rows:
            st = load_run_state(Path(r["run_path"])) if r.get("run_path") else None
            r["submitted_by"] = (st or {}).get("submitted_by") or None
            r["calc_type"] = (st or {}).get("calc_type")
        all_rows = rows

    def _refresh_users():
        users = sorted({r["submitted_by"] for r in all_rows if r.get("submitted_by")})
        options = [("All users", None)] + [(u, u) for u in users] + [("Unknown", "__unknown__")]
        user_w.options = options
        user_w.value = None

    def _refresh_runs(*_):
        selected = user_w.value
        if selected is None:
            rows = all_rows
        elif selected == "__unknown__":
            rows = [r for r in all_rows if not r.get("submitted_by")]
        else:
            rows = [r for r in all_rows if r.get("submitted_by") == selected]
        options = [
            (f"{Path(r.get('cif') or '?').name} · {r.get('run_id')} · {r.get('crystal_status') or 'pending'}", r)
            for r in sorted(rows, key=lambda r: r.get("run_id") or "", reverse=True)
        ]
        run_w.options = options
        run_w.value = options[0][1] if options else None

    def on_load(_=None):
        result_out.clear_output()
        row = run_w.value
        with result_out:
            if not row:
                print("No run selected.")
                return
            out_file = row.get("qe_output_file")
            if not out_file or not Path(out_file).is_file():
                display(HTML('<div class="advqm-muted" style="padding:8px 0;">'
                              'This run has no QE output file yet — the calculation may still be running.</div>'))
                return
            history = _parse_scf_history(Path(out_file))
            chart_html = _render_scf_chart(history)
            energy = history.get("final_energy")
            display(
                HTML(
                    f"""
                    <div class="advqm-card">
                      <div class="advqm-card-title">📄 {_fmt(Path(row.get('cif') or '?').name)}</div>
                      <div class="advqm-detail-grid">
                        <div><b>Run ID</b><br>{_fmt(row.get('run_id'))}</div>
                        <div><b>Calculation Type</b><br>{_fmt(row.get('calc_type'))}</div>
                        <div><b>Status</b><br>{status_badge_html(row.get('crystal_status'))}</div>
                        <div><b>Final Energy</b><br>{_fmt_energy(energy)} Ry</div>
                        <div><b>SCF Steps</b><br>{len(history.get('accuracy') or [])}</div>
                        <div><b>Submitted By</b><br>{_fmt(row.get('submitted_by'))}</div>
                      </div>
                      <div style="margin-top:16px;">{chart_html}</div>
                    </div>
                    """
                )
            )

    user_w.observe(_refresh_runs, names="value")
    load_btn.on_click(on_load)

    _load_all_rows()
    _refresh_users()
    _refresh_runs()

    card = widgets.VBox(
        [
            widgets.HTML('<div class="advqm-card-title">Load a run</div>'),
            widgets.HBox([user_w, run_w], layout=widgets.Layout(width="100%", justify_content="space-between")),
            load_btn,
        ]
    )
    card.add_class("advqm-card")

    return widgets.VBox([header, card, result_out])


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
        elif key == "results":
            pages["results"] = _build_results_page(nav)
            main_area.children = [pages["results"]]

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
