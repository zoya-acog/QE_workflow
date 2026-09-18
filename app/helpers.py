import sys
import argparse
import logging
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path("/mnt/own6a/zoya/QM_Workflow/.voila_uploads")

for _p in (PROJECT_ROOT / "advQMcalc_test" / "src",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from advQMcalc_test import __version__
    from advQMcalc_test.cli import (
        _process_single_cif,
        _aggregate_crystal_results,
        _query_slurm_job,
        _iter_per_cif_runs,
        _runs_dir_for_cif,
        _write_summary_csv,
        _write_summary_json,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"advQMcalc_test package is not importable ({exc}). "
        "Install it or run from the project root."
    ) from exc


DEFAULTS = dict(
    cif=None,
    cif_dir=None,
    cif_glob="*.cif",
    calc_type="scf",
    runs_dir="runs",
    log_name="advQMcalc.log",
    resume=False,
    run_id=None,
    run_path=None,
    pseudo_dir=None,
    pp_map=None,
    force_pp_cleanup=False,
    slurm_walltime="12:00:00",
    slurm_ntasks=8,
    qe_command="pw.x",
    kpoints=None,
    kpoint_separation=0.03,
    aggregate=False,
    summary_out=None,
    summary_json_out=None,
    summary_only_successful=False,
)


def build_namespace(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace matching advQMcalc_test.cli expectations."""
    values = dict(DEFAULTS)
    values.update(overrides)
    return argparse.Namespace(**values)


class WidgetLogHandler(logging.Handler):
    """Forward logging records into an ipywidgets.Output widget."""

    def __init__(self, output):
        super().__init__()
        self.output = output

    def emit(self, record):
        try:
            text = self.format(record) + "\n"
            if record.levelno >= logging.ERROR:
                self.output.append_stderr(text)
            else:
                self.output.append_stdout(text)
        except Exception:
            pass


def attach_log_handler(output) -> WidgetLogHandler:
    """Attach (idempotently) a widget handler to the workflow logger."""
    logger = logging.getLogger("advQMcalc_test")
    logger.setLevel(logging.INFO)
    for existing in list(logger.handlers):
        if isinstance(existing, WidgetLogHandler):
            logger.removeHandler(existing)
    handler = WidgetLogHandler(output)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    return handler


def save_uploads(upload_widget, staging_dir: Path) -> list[Path]:
    """Write uploaded CIF files (bytes) into staging_dir and return their paths."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    value = upload_widget.value
    if isinstance(value, tuple):
        # ipywidgets 8 with multiple=True returns a tuple of dicts
        items = list(value)
    else:
        items = value.items()
    for item in items:
        if isinstance(item, dict):
            name = item.get("name") or "uploaded_file"
            content = item.get("content", b"")
        else:
            key, meta = item
            name = meta.get("name") or key
            content = meta.get("content", b"")
        if not str(name).lower().endswith(".cif"):
            raise ValueError(f"Skipping non-CIF upload: {name}")
        dest = staging_dir / Path(name).name
        dest.write_bytes(content)
        paths.append(dest)
    return paths


def run_in_thread(fn, on_done=None):
    """Run fn in a daemon thread so the widget UI stays responsive."""

    def wrapper():
        try:
            fn()
        except Exception:
            logging.getLogger("advQMcalc_test").exception("Workflow raised an unhandled error")
        finally:
            if on_done is not None:
                try:
                    on_done()
                except Exception:
                    pass

    threading.Thread(target=wrapper, daemon=True).start()


def enumerate_runs(base: Path) -> list[tuple[str, Path]]:
    """List (cif_label, run_path) pairs for the latest runs under base."""
    return _iter_per_cif_runs(base)


def load_run_state(run_path: Path) -> dict | None:
    """Read state.json from a run directory; None if missing/malformed."""
    state_file = Path(run_path) / "state.json"
    try:
        import json

        with open(state_file) as f:
            return json.load(f)
    except Exception:
        return None


def resume_target(base: Path, cif_label: str, run_path: Path) -> tuple[Path, str]:
    """Return (runs_dir, run_id) to use when resuming the selected run."""
    per_cif = base / cif_label
    if per_cif.is_dir() and Path(run_path).parent == per_cif:
        return per_cif, run_path.name
    return base, run_path.name


def build_app_style() -> str:
    """Global <style> block that gives the widget UI a clean, app-like look."""
    return """
<style>
/* the "classic" voila template loads labvariables.css, which defines
   --jp-ui-font-size1 (13px) / --jp-content-font-size1 (14px) — ipywidgets'
   own CSS keys its font-size off these, so without this override every
   widget label/button/input renders smaller than the app was designed at. */
:root { --jp-ui-font-size1: 14px !important; --jp-content-font-size1: 15px !important; }
body { background:#f4f6f9; font-family:'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size:14px; overflow-x:hidden; }

/* generic widget polish (still used inside cards / accordions) */
.widget-button { border-radius:6px; font-weight:600; min-height:34px; }
.widget-button.mod-primary { background:#1565c0 !important; border-color:#1565c0 !important; color:#fff !important; }
.widget-button.mod-success { background:#2e7d32 !important; border-color:#2e7d32 !important; color:#fff !important; }
.widget-button:hover { filter:brightness(1.05); }
.widget-upload { border:1.5px dashed #c3cbd6; border-radius:8px; background:#fafbfc; }
.advqm-group-title { font-size:0.72rem; font-weight:700; letter-spacing:0.09em;
    text-transform:uppercase; color:#5b6472; margin:0 0 10px; }
.widget-text .widget-label, .widget-int .widget-label,
.widget-float .widget-label, .widget-dropdown .widget-label,
.widget-checkbox, .widget-label { color:#475069; }
.widget-accordion { border:none !important; background:transparent !important; }
.widget-accordion .p-Collapse-header, .widget-accordion .lm-Widget.p-Accordion-title,
.widget-accordion .widget-title { background:#f7f8fa !important; border-radius:8px !important;
    font-weight:600 !important; color:#475069 !important; }

/* ── field sizing: stop the per-field horizontal scrollbars ───────────────
   ipywidgets inputs default to content-box sizing with a fixed intrinsic
   width; inside a flexed 48%/31% column that overflows by a few px and
   Jupyter's output area then renders a scrollbar. Force border-box + 100%
   width everywhere and stop that output wrapper from clipping/scrolling. */
.widget-text, .widget-int, .widget-float, .widget-dropdown,
.widget-text input, .widget-int input, .widget-float input, .widget-dropdown select {
    box-sizing:border-box !important; width:100% !important; max-width:100% !important;
}
.widget-inline-hbox { width:100% !important; box-sizing:border-box !important; }
.jp-OutputArea-output, .jp-RenderedHTMLCommon, .lm-Widget.jp-OutputArea-child {
    overflow-x:hidden !important; }
/* ipywidgets containers carry a default 2px margin on every side; inside a
   percentage-width flex column that pushes the element past 100% of its
   slot and the rounded card border then clips it off. Zero that out and
   make every box/vbox/hbox honor border-box so percentage widths are exact. */
.widget-box, .widget-vbox, .widget-hbox, .widget-gridbox, .widget-html, .widget-html-content {
    box-sizing:border-box !important; margin:0 !important; }

/* ── app shell: sidebar + main content ─────────────────────────────────── */
.advqm-shell { display:flex; align-items:stretch; gap:22px; margin-top:34px; overflow-x:hidden; }
.advqm-sidebar { flex:0 0 220px; background:#eaf2fd; border:1px solid #dbe8fa; border-radius:12px; padding:14px 10px;
    display:flex; flex-direction:column; gap:3px;
    position:sticky; top:14px; align-self:flex-start;
    min-height:calc(100vh - 120px); }
.advqm-main { flex:1 1 auto; min-width:0; }

.advqm-navbtn.widget-button { justify-content:flex-start !important; text-align:left !important;
    background:transparent !important; border:none !important; color:#3c5a80 !important;
    font-weight:600 !important; font-size:1.37rem !important; padding:11px 14px !important;
    border-radius:8px !important; min-height:48px !important; box-shadow:none !important; }
.advqm-navbtn.widget-button:hover { background:#dbe8fa !important; color:#0f3a70 !important; }
.advqm-navbtn-active.widget-button, .advqm-navbtn-active.widget-button:hover {
    background:#1565c0 !important; color:#fff !important; }

/* ── page chrome ────────────────────────────────────────────────────────── */
.advqm-page-title { font-size:2.4rem; font-weight:800; color:#101828; margin:2px 0 2px; }
.advqm-page-sub { color:#667085; font-size:1.3rem; margin-bottom:16px; }
.advqm-page-header { display:flex; align-items:flex-start; justify-content:space-between;
    gap:12px; flex-wrap:wrap; }

.advqm-card { background:#fff; border:1px solid #e5e9f0; border-radius:12px;
    padding:18px 20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(15,23,42,0.05); }
.advqm-card-title { font-weight:700; font-size:1.5rem; color:#101828;
    display:flex; align-items:center; gap:8px; margin-bottom:14px; }

/* ── stat cards ─────────────────────────────────────────────────────────── */
.advqm-stats-row { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:18px; }
.advqm-stat-card { background:#fff; border:1px solid #e5e9f0; border-radius:12px;
    padding:16px 18px; flex:1 1 160px; min-width:150px; }
.advqm-stat-num { font-size:3rem; font-weight:800; color:#101828; line-height:1.1; }
.advqm-stat-label { font-size:1.1rem; color:#667085; font-weight:700;
    text-transform:uppercase; letter-spacing:0.05em; margin-top:4px; }
.advqm-stat-num.success { color:#1c7a3d; }
.advqm-stat-num.info { color:#1565c0; }
.advqm-stat-num.danger { color:#c62828; }

/* ── table + badges ─────────────────────────────────────────────────────── */
.advqm-table-wrap { overflow-x:auto; }
.advqm-table { width:100%; border-collapse:collapse; font-size:0.84rem; table-layout:fixed; }
.advqm-table th { text-align:left; color:#667085; font-size:0.68rem; text-transform:uppercase;
    letter-spacing:0.06em; padding:10px 14px; border-bottom:1px solid #e5e9f0; white-space:nowrap; }
.advqm-table td { padding:12px 14px; border-bottom:1px solid #f1f3f7; color:#1f2937;
    white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.advqm-table td.advqm-col-cif { max-width:0; width:26%; overflow:hidden; text-overflow:ellipsis; }
.advqm-table th:nth-child(1) { width:26%; }
.advqm-table th:nth-child(2) { width:16%; }
.advqm-table th:nth-child(3) { width:12%; }
.advqm-table th:nth-child(4) { width:16%; }
.advqm-table th:nth-child(5) { width:14%; }
.advqm-table th:nth-child(6) { width:16%; }
.advqm-table tr:hover td { background:#fafbfc; }
.advqm-badge { display:inline-block; padding:5px 15px; border-radius:999px;
    font-size:0.9rem; font-weight:700; white-space:nowrap; }
.advqm-badge-success { background:#e3f6e9; color:#1c7a3d; }
.advqm-badge-info { background:#e3f1fd; color:#1565c0; }
.advqm-badge-warning { background:#fdf3d8; color:#9a6b00; }
.advqm-badge-danger { background:#fde3e3; color:#c62828; }
.advqm-badge-neutral { background:#eef0f3; color:#5b6472; }

.advqm-list-row { display:flex; align-items:center; justify-content:space-between;
    padding:10px 4px; border-bottom:1px solid #f1f3f7; font-size:1.2rem; }
.advqm-list-row:last-child { border-bottom:none; }
.advqm-muted { color:#98a2b3; font-size:1rem; }

/* ── clickable runs row-list (Runs page) ───────────────────────────────── */
.advqm-rowlist { padding:0 !important; overflow:hidden; }
.advqm-rowlist-head { display:flex; align-items:center; box-sizing:border-box;
    padding:12px 16px; font-size:1.4rem; font-weight:700; color:#667085;
    letter-spacing:0.02em; border-bottom:1px solid #e5e9f0;
    background:#fafbfc; }
.advqm-rowlist-head span { box-sizing:border-box; flex:0 1 auto; min-width:0;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding-right:8px; }
.advqm-rowlist-row.widget-hbox { padding:10px 16px !important; border-bottom:1px solid #f1f3f7 !important;
    align-items:center !important; font-size:0.95rem !important; }
.advqm-rowlist-row.widget-hbox:hover { background:#fafbfc !important; }
.advqm-rowlist-row.widget-hbox > * { box-sizing:border-box !important; min-width:0 !important; }
.advqm-rowcell-cif span { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* ── run detail card ────────────────────────────────────────────────────── */
.advqm-detail-energy-label { font-size:0.72rem; font-weight:700; color:#667085;
    text-transform:uppercase; letter-spacing:0.06em; margin-top:10px; }
.advqm-detail-energy { font-size:2.6rem; font-weight:800; color:#101828; line-height:1.15; }
.advqm-detail-energy-unit { font-size:1.1rem; font-weight:600; color:#667085; }
.advqm-detail-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:14px 18px;
    margin-top:18px; font-size:0.85rem; color:#344054; }
.advqm-detail-grid b { display:block; font-size:0.68rem; font-weight:700; color:#98a2b3;
    text-transform:uppercase; letter-spacing:0.05em; margin-bottom:3px; }
</style>
"""


def build_app_header() -> str:
    """HTML/CSS header banner styled after the TrueRhoVoila HubApp toolbar."""
    return f"""
<style>
.advqm-header {{
    display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
    background:#1565c0; color:#fff; padding:15px 22px;
    border-radius:0 0 8px 8px; box-shadow:0 2px 6px rgba(0,0,0,0.15);
    font-family:sans-serif;
}}
.advqm-brand {{ display:flex; align-items:center; gap:12px; }}
.advqm-brand img {{ vertical-align:middle; object-fit:contain; filter:brightness(0) invert(1); }}
.advqm-collab {{ color:rgba(255,255,255,0.75); font-size:1.4rem; font-weight:500; letter-spacing:0.06em; white-space:nowrap; }}
.advqm-titlebar {{ display:flex; flex-direction:column; align-items:flex-end; line-height:1.3; }}
.advqm-title {{ font-weight:700; font-size:1.15rem; letter-spacing:0.3px; }}
.advqm-sub {{ color:rgba(255,255,255,0.78); font-size:0.78rem; }}
</style>
<div class="advqm-header">
  <div class="advqm-brand">
    <img src="https://www.aganitha.ai/wp-content/uploads/2023/05/aganitha-logo.png"
         alt="Aganitha" style="height:32px; max-width:140px;">
    <span class="advqm-collab">× Joint Collaboration ×</span>
    <img src="https://upload.wikimedia.org/wikipedia/commons/5/57/Pfizer_%282021%29.svg"
         alt="Pfizer" style="height:28px; max-width:100px;">
  </div>
  <div class="advqm-titlebar">
    <span class="advqm-title">QM Workflow for Crystal Systems (WIP)</span>
  </div>
</div>
"""


_STATUS_MAP = {
    "qe_energy_extracted": ("Finished", "success"),
    "running": ("Running", "info"),
    "completed": ("Completed", "info"),
    "qe_completed": ("Completed", "info"),
    "pending": ("Pending", "warning"),
    "staged": ("Pending", "warning"),
    "submitted": ("Pending", "warning"),
    "inputs_generated": ("Pending", "warning"),
    "job_script_generated": ("Pending", "warning"),
    "resumed": ("Pending", "warning"),
    "qe_failed_validation": ("Failed", "danger"),
    "failed": ("Failed", "danger"),
    "cancelled": ("Failed", "danger"),
    "timeout": ("Failed", "danger"),
    "node_fail": ("Failed", "danger"),
    "out_of_memory": ("Failed", "danger"),
    "boot_fail": ("Failed", "danger"),
    "preempted": ("Failed", "danger"),
}


def status_badge_html(status: str | None) -> str:
    """Render a run status as a colored pill <span>."""
    key = (status or "").lower()
    label, css = _STATUS_MAP.get(key, (status.replace("_", " ").title() if status else "Unknown", "neutral"))
    return f'<span class="advqm-badge advqm-badge-{css}">{label}</span>'


def status_category(status: str | None) -> str:
    """Bucket a raw status string into success / info / warning / danger / neutral."""
    key = (status or "").lower()
    return _STATUS_MAP.get(key, (None, "neutral"))[1]


__all__ = [
    "PROJECT_ROOT",
    "UPLOAD_DIR",
    "build_namespace",
    "attach_log_handler",
    "save_uploads",
    "run_in_thread",
    "enumerate_runs",
    "load_run_state",
    "resume_target",
    "build_app_header",
    "build_app_style",
    "status_badge_html",
    "status_category",
    "_process_single_cif",
    "_aggregate_crystal_results",
    "_query_slurm_job",
    "_runs_dir_for_cif",
    "_write_summary_csv",
    "_write_summary_json",
]