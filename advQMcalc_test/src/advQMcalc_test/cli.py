from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from shutil import which

from advQMcalc_test import __version__
from advQMcalc_test.config import WorkflowConfig
from advQMcalc_test.logging_utils import setup_logger
from advQMcalc_test.run_context import RunContext


# ============================================================
# RUN / CIF SELECTION
# ============================================================
def _select_run_dir(runs_dir: Path, run_id: str | None, run_path: str | None) -> Path:
    if run_path is not None:
        chosen = Path(run_path)
        if not chosen.exists() or not chosen.is_dir():
            raise ValueError(f"--run-path does not exist or is not a directory: {chosen}")
        return chosen

    if run_id is not None:
        chosen = runs_dir / run_id
        if not chosen.exists() or not chosen.is_dir():
            raise ValueError(f"--run-id not found under runs directory: {chosen}")
        return chosen

    if not runs_dir.exists():
        raise ValueError("Runs directory does not exist; nothing to resume")

    existing_runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if not existing_runs:
        raise ValueError("No existing runs found to resume")

    return existing_runs[-1]


def _collect_cif_paths(args: argparse.Namespace) -> list[Path]:
    cif_paths: list[Path] = []

    if args.cif:
        cif_paths.extend(Path(p) for p in args.cif)

    if args.cif_dir is not None:
        cif_dir = Path(args.cif_dir)
        if not cif_dir.exists() or not cif_dir.is_dir():
            raise ValueError(f"--cif-dir does not exist or is not a directory: {cif_dir}")
        cif_paths.extend(sorted(cif_dir.glob(args.cif_glob)))

    out: list[Path] = []
    seen: set[str] = set()
    for p in cif_paths:
        rp = str(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)

    if not out:
        raise ValueError("Provide at least one CIF via --cif or --cif-dir")

    for p in out:
        if not p.exists():
            raise ValueError(f"CIF file does not exist: {p}")

    return out


def _runs_dir_for_cif(base_runs_dir: Path, cif_path: Path, multi_mode: bool) -> Path:
    return base_runs_dir / cif_path.stem if multi_mode else base_runs_dir


# ============================================================
# AGGREGATION HELPERS (Day 16)
# ============================================================
def _iter_per_cif_runs(runs_dir: Path) -> list[tuple[str, Path]]:
    """
    Walk runs_dir and return a list of (cif_label, run_path) for each latest run.

    Supports both:
      multi-cif:  runs_dir / <cif_stem> / <timestamp>
      single-cif: runs_dir / <timestamp>
    """
    if not runs_dir.exists():
        raise RuntimeError(f"Runs directory does not exist: {runs_dir}")

    pairs: list[tuple[str, Path]] = []

    children = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
    if not children:
        return pairs

    # Heuristic: timestamp folders contain digits and an underscore
    def looks_like_timestamp(d: Path) -> bool:
        name = d.name
        return any(ch.isdigit() for ch in name) and "_" in name

    # If every child looks like a timestamp, this is a single-CIF runs_dir
    if all(looks_like_timestamp(d) for d in children):
        latest = children[-1]
        pairs.append((runs_dir.name, latest))
        return pairs

    # Otherwise treat each child as a per-CIF directory
    for cif_dir in children:
        runs = sorted([d for d in cif_dir.iterdir() if d.is_dir()])
        if not runs:
            continue
        latest = runs[-1]
        pairs.append((cif_dir.name, latest))

    return pairs


def _aggregate_crystal_results(
    runs_dir: Path,
    only_successful: bool,
) -> list[dict]:
    """
    Walk runs_dir and collect state.json (and crystal_result.json when present)
    for each latest run, returning a list of summary rows.
    """
    rows: list[dict] = []

    for cif_label, run_path in _iter_per_cif_runs(runs_dir):
        state_file = run_path / "state.json"
        result_file = run_path / "results" / "crystal_result.json"

        if not state_file.exists():
            continue

        with open(state_file) as f:
            state = json.load(f)

        crystal = state.get("tasks", {}).get("crystal", {})
        crystal_status = crystal.get("status")
        if only_successful and crystal_status != "qe_energy_extracted":
            continue

        qe_output_block = crystal.get("qe_output") or {}
        qe_results_block = crystal.get("qe_results") or {}

        row = {
            "cif": state.get("cif") or cif_label,
            "run_id": run_path.name,
            "crystal_status": crystal_status,
            "slurm_state": crystal.get("slurm_state"),
            "job_id": crystal.get("job_id"),
            "kpoints": crystal.get("kpoints"),
            "pseudo_dir": crystal.get("pseudo_dir"),
            "qe_output_file": qe_output_block.get("qe_output_file"),
            "energy_ry": qe_results_block.get("energy_ry"),
            "result_file": str(result_file) if result_file.exists() else None,
            "run_path": str(run_path),
        }

        rows.append(row)

    return rows


def _write_summary_csv(rows: list[dict], out_path: Path) -> None:
    columns = [
        "cif",
        "run_id",
        "crystal_status",
        "slurm_state",
        "job_id",
        "kpoints",
        "pseudo_dir",
        "qe_output_file",
        "energy_ry",
        "result_file",
        "run_path",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_json(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))


# ============================================================
# QE INPUT GENERATION
# ============================================================
def _generate_qe_input_with_cif2cell(crystal_dir: Path, staged_cif: Path, calc_type: str) -> Path:
    if not calc_type or str(calc_type).strip() == "":
        raise RuntimeError(f"Invalid calc_type='{calc_type}'")

    out_path = crystal_dir / f"{staged_cif.stem}.{calc_type}.in"
    expected = crystal_dir / staged_cif.name
    if not expected.exists():
        raise RuntimeError(f"Expected staged CIF not found in crystal_dir: {expected}")

    cmd = [
        "cif2cell",
        staged_cif.name,
        "--program",
        "quantum-espresso",
        "--outputfile",
        out_path.name,
    ]

    result = subprocess.run(
        cmd,
        cwd=crystal_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "cif2cell failed:\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"cif2cell reported success but did not create a valid output file: {out_path}\n"
            f"STDERR:\n{result.stderr}"
        )

    return out_path


# ============================================================
# PSEUDOPOTENTIAL / QE INPUT HELPERS
# ============================================================
def _canonicalize_element(sym: str) -> str:
    if not sym:
        return sym
    s = sym.strip()
    return s.upper() if len(s) == 1 else s[0].upper() + s[1:].lower()


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    os.replace(tmp_path, path)


def _extract_atomic_species_elements(qe_text: str) -> list[str]:
    lines = qe_text.splitlines()
    elements: list[str] = []
    in_block = False

    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("ATOMIC_SPECIES"):
            in_block = True
            continue

        if not in_block:
            continue

        if upper.startswith((
            "ATOMIC_POSITIONS",
            "K_POINTS",
            "CELL_PARAMETERS",
            "OCCUPATIONS",
            "CONSTRAINTS",
            "HUBBARD",
            "&",
            "/",
        )):
            break

        if not stripped or stripped.startswith(("!", "#")):
            continue

        tokens = stripped.split()
        if len(tokens) < 3:
            continue

        sym = tokens[0]
        if not (1 <= len(sym) <= 2 and sym.isalpha()):
            continue

        try:
            float(tokens[1])
        except ValueError:
            continue

        elements.append(_canonicalize_element(sym))

    return elements


def _resolve_pseudo_filenames(
    elements: list[str],
    pseudo_dir: Path,
    pp_map: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    canon_map: dict[str, str] = {}
    if pp_map:
        for k, v in pp_map.items():
            canon_map[_canonicalize_element(k)] = v

    pp_files = [p.name for p in pseudo_dir.iterdir() if p.is_file()]

    resolved: dict[str, str] = {}
    sources: dict[str, str] = {}
    for el in elements:
        if el in canon_map:
            resolved[el] = canon_map[el]
            sources[el] = "explicit"
            continue

        el_lower = el.lower()
        candidates: list[str] = []
        for name in pp_files:
            n_lower = name.lower()
            if not n_lower.endswith(".upf"):
                continue
            if n_lower.startswith(el_lower + ".") or n_lower.startswith(el_lower + "_"):
                candidates.append(name)

        if len(candidates) == 0:
            raise RuntimeError(f"No pseudopotential found for element '{el}' in {pseudo_dir}")
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple pseudopotentials found for element '{el}' in {pseudo_dir}: {sorted(candidates)}\n"
                "Use --pp-map to specify which one to use."
            )

        resolved[el] = candidates[0]
        sources[el] = "auto"

    for el, pp_file in resolved.items():
        full = pseudo_dir / pp_file
        if not full.is_file():
            raise RuntimeError(
                f"Pseudopotential file does not exist: {full} "
                f"(resolved for element '{el}', source={sources[el]})"
            )

    return resolved, sources


def _extract_cell_lengths_from_cif(cif_path: Path) -> tuple[float, float, float]:
    a = b = c = None
    for line in cif_path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("_cell_length_a"):
            a = float(re.split(r"\s+", stripped)[-1])
        elif stripped.startswith("_cell_length_b"):
            b = float(re.split(r"\s+", stripped)[-1])
        elif stripped.startswith("_cell_length_c"):
            c = float(re.split(r"\s+", stripped)[-1])

    if a is None or b is None or c is None:
        raise RuntimeError(f"Could not extract cell lengths a/b/c from CIF: {cif_path}")
    return a, b, c


def _compute_kpoints_from_cif(cif_path: Path, separation: float) -> str:
    a, b, c = _extract_cell_lengths_from_cif(cif_path)
    kp_x = max(int(1.0 / (a * separation) + 0.5), 1)
    kp_y = max(int(1.0 / (b * separation) + 0.5), 1)
    kp_z = max(int(1.0 / (c * separation) + 0.5), 1)
    return f"{kp_x} {kp_y} {kp_z} 0 0 0"


def _replace_or_append_kpoints_block(text: str, kpoints: str) -> str:
    lines = text.splitlines()
    out_lines: list[str] = []
    i = 0
    replaced = False

    while i < len(lines):
        stripped = lines[i].strip().upper()
        if stripped.startswith("K_POINTS"):
            out_lines.append("K_POINTS automatic")
            out_lines.append(kpoints)
            replaced = True
            i += 1
            if i < len(lines):
                i += 1
            continue
        out_lines.append(lines[i])
        i += 1

    if not replaced:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append("K_POINTS automatic")
        out_lines.append(kpoints)

    return "\n".join(out_lines) + "\n"


def _ensure_qe_required_sections(text: str, pseudo_dir_str: str, calc_type: str, kpoints: str) -> str:
    new_text = text
    calculation_line = f"  calculation = '{calc_type}'"
    pseudo_dir_line = f"  pseudo_dir = '{pseudo_dir_str}'"

    control_pattern = re.compile(r"^\s*&CONTROL\b", flags=re.MULTILINE | re.IGNORECASE)
    if control_pattern.search(new_text):
        pseudo_dir_pattern = re.compile(r"^\s*pseudo_dir\s*=.*$", flags=re.MULTILINE)
        if pseudo_dir_pattern.search(new_text):
            new_text = pseudo_dir_pattern.sub(pseudo_dir_line, new_text, count=1)
        else:
            control_header_pattern = re.compile(r"^(\s*&CONTROL\s*\n)", flags=re.MULTILINE | re.IGNORECASE)
            new_text = control_header_pattern.sub(
                lambda m: m.group(1) + calculation_line + "\n" + pseudo_dir_line + "\n",
                new_text,
                count=1,
            )
    else:
        new_text = "&CONTROL\n" + calculation_line + "\n" + pseudo_dir_line + "\n/\n" + new_text

    electrons_pattern = re.compile(r"^\s*&ELECTRONS\b", flags=re.MULTILINE | re.IGNORECASE)
    if not electrons_pattern.search(new_text):
        electrons_block = (
            "&ELECTRONS\n"
            "  conv_thr = 1.0d-8\n"
            "  mixing_beta = 0.5\n"
            "  electron_maxstep = 1500\n"
            "  diago_thr_init = 1.0d-4\n"
            "  scf_must_converge = .true.\n"
            "/\n"
        )
        system_block_pattern = re.compile(
            r"(^\s*&SYSTEM\b.*?^\s*/\s*$)",
            flags=re.MULTILINE | re.IGNORECASE | re.DOTALL,
        )
        if system_block_pattern.search(new_text):
            new_text = system_block_pattern.sub(lambda m: m.group(1) + "\n" + electrons_block, new_text, count=1)
        else:
            new_text = electrons_block + new_text

    return _replace_or_append_kpoints_block(new_text, kpoints)


def _cleanup_qe_input(
    qe_path: Path,
    pseudo_dir: Path,
    pp_map: dict[str, str] | None,
    calc_type: str,
    kpoints: str,
) -> dict[str, object]:
    text = qe_path.read_text()
    elements = _extract_atomic_species_elements(text)
    if not elements:
        raise RuntimeError(f"No ATOMIC_SPECIES block found in {qe_path}")

    pp_resolved, pp_sources = _resolve_pseudo_filenames(elements, pseudo_dir, pp_map)
    pseudo_dir_str = str(pseudo_dir.resolve())

    new_lines: list[str] = []
    in_species_block = False
    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("ATOMIC_SPECIES"):
            in_species_block = True
            new_lines.append(line)
            continue

        if in_species_block:
            if upper.startswith((
                "ATOMIC_POSITIONS",
                "K_POINTS",
                "CELL_PARAMETERS",
                "OCCUPATIONS",
                "CONSTRAINTS",
                "HUBBARD",
                "&",
                "/",
            )):
                in_species_block = False
                new_lines.append(line)
                continue

            if not stripped or stripped.startswith(("!", "#")):
                new_lines.append(line)
                continue

            tokens = stripped.split()
            if len(tokens) >= 3:
                sym_raw = tokens[0]
                if 1 <= len(sym_raw) <= 2 and sym_raw.isalpha():
                    sym = _canonicalize_element(sym_raw)
                    if sym in pp_resolved:
                        try:
                            float(tokens[1])
                            new_lines.append(f"  {sym}  {tokens[1]}  {pp_resolved[sym]}")
                            continue
                        except ValueError:
                            pass
            new_lines.append(line)
            continue

        new_lines.append(line)

    new_text = "\n".join(new_lines)
    if text.endswith("\n") and not new_text.endswith("\n"):
        new_text += "\n"

    new_text = _ensure_qe_required_sections(new_text, pseudo_dir_str, calc_type, kpoints)
    _atomic_write_text(qe_path, new_text)

    return {
        "pseudo_dir": pseudo_dir_str,
        "pp_map_used": pp_resolved,
        "pp_sources": pp_sources,
    }


# ============================================================
# SLURM SUBMISSION / MONITORING HELPERS
# ============================================================
def _submit_slurm_script(script_path: Path) -> tuple[str, str]:
    result = subprocess.run(
        ["sbatch", script_path.name],
        cwd=script_path.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "sbatch submission failed:\n"
            f"CMD: sbatch {script_path.name}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    stdout = result.stdout.strip()
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if not match:
        raise RuntimeError(f"Could not parse SLURM job ID from sbatch output:\n{stdout}")
    return match.group(1), stdout


def _query_slurm_job(job_id: str) -> str:
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=State", "--noheader"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sacct failed for job {job_id}\nSTDERR:\n{result.stderr}")

    states: list[str] = []
    for line in result.stdout.splitlines():
        s = line.strip()
        if not s:
            continue
        s = s.split()[0]
        if "." in s:
            continue
        states.append(s)

    if not states:
        return "UNKNOWN"

    priority = ["FAILED", "CANCELLED", "TIMEOUT", "RUNNING", "PENDING", "COMPLETED"]
    for p in priority:
        if p in states:
            return p
    return states[0]


# ============================================================
# OUTPUT VALIDATION / ENERGY EXTRACTION
# ============================================================
def _inspect_crystal_outputs(run_ctx: RunContext, crystal_task: dict) -> dict[str, object]:
    qe_input_rel = crystal_task.get("qe_input")
    job_script_rel = crystal_task.get("job_script")
    if qe_input_rel is None:
        raise RuntimeError("qe_input missing from crystal task")
    if job_script_rel is None:
        raise RuntimeError("job_script missing from crystal task")

    qe_input_path = run_ctx.run_dir / qe_input_rel
    job_script_path = run_ctx.run_dir / job_script_rel
    qe_out_path = qe_input_path.with_suffix(".out")
    slurm_out_path = job_script_path.with_suffix(".slurm.out")

    info: dict[str, object] = {
        "qe_output_file": str(qe_out_path),
        "qe_output_exists": qe_out_path.exists(),
        "qe_output_size": 0,
        "slurm_output_file": str(slurm_out_path),
        "slurm_output_exists": slurm_out_path.exists(),
        "slurm_output_size": 0,
        "qe_job_done": False,
        "qe_has_error": False,
        "qe_error_markers": [],
        "validation_source": None,
    }

    error_markers = [
        "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%",
        "Error in routine",
        "stopping ...",
        "convergence NOT achieved",
        "Maximum CPU time exceeded",
        "error while reading",
        "from read_namelists",
        "from card_atomic_species",
        "command not found",
        "No such file or directory",
    ]

    if qe_out_path.exists():
        info["qe_output_size"] = qe_out_path.stat().st_size
        if qe_out_path.stat().st_size > 0:
            text = qe_out_path.read_text(errors="ignore")
            info["validation_source"] = "qe_output"
            if "JOB DONE" in text:
                info["qe_job_done"] = True
            found_errors = [m for m in error_markers if m in text]
            if found_errors:
                info["qe_has_error"] = True
                info["qe_error_markers"] = found_errors
            return info

    if slurm_out_path.exists():
        info["slurm_output_size"] = slurm_out_path.stat().st_size
        if slurm_out_path.stat().st_size > 0:
            text = slurm_out_path.read_text(errors="ignore")
            info["validation_source"] = "slurm_output"
            if "JOB DONE" in text:
                info["qe_job_done"] = True
            found_errors = [m for m in error_markers if m in text]
            if found_errors:
                info["qe_has_error"] = True
                info["qe_error_markers"] = found_errors

    return info


def _extract_qe_total_energy(crystal_task: dict) -> dict[str, object]:
    qe_output = crystal_task.get("qe_output")
    if qe_output is None:
        raise RuntimeError("qe_output block missing from crystal task")
    if qe_output.get("validation_source") not in ("qe_output", "slurm_output"):
        raise RuntimeError("No validated output source recorded for energy extraction")

    output_file = qe_output.get("qe_output_file")
    if qe_output.get("validation_source") == "slurm_output":
        output_file = qe_output.get("slurm_output_file")

    if output_file is None:
        raise RuntimeError("Validated output file path missing")

    out_path = Path(output_file)
    if not out_path.exists():
        raise RuntimeError(f"Validated QE output file does not exist: {out_path}")

    text = out_path.read_text(errors="ignore")
    matches = re.findall(r"^\s*!\s+total energy\s+=\s+([-+]?\d+\.\d+)\s+Ry", text, flags=re.MULTILINE)
    if not matches:
        raise RuntimeError(f"Could not find final total energy in QE output: {out_path}")

    energy_ry = float(matches[-1])
    line_match = re.findall(r"^\s*!\s+total energy\s+=\s+[-+]?\d+\.\d+\s+Ry", text, flags=re.MULTILINE)
    energy_line = line_match[-1].strip() if line_match else None
    return {"energy_ry": energy_ry, "energy_line": energy_line, "energy_source": qe_output.get("validation_source")}


def _write_crystal_result_file(run_ctx: RunContext, state: dict) -> Path:
    crystal = state["tasks"]["crystal"]
    result_payload = {
        "cif": state.get("cif"),
        "calc_type": state.get("calc_type"),
        "kpoints": crystal.get("kpoints"),
        "pseudo_dir": crystal.get("pseudo_dir"),
        "job_id": crystal.get("job_id"),
        "slurm_state": crystal.get("slurm_state"),
        "crystal_status": crystal.get("status"),
        "qe_output_file": crystal.get("qe_output", {}).get("qe_output_file"),
        "energy_ry": crystal.get("qe_results", {}).get("energy_ry"),
    }
    result_path = run_ctx.run_dir / "results" / "crystal_result.json"
    result_path.write_text(json.dumps(result_payload, indent=2))
    return result_path


# ============================================================
# SINGLE-CIF WORKFLOW DRIVER
# ============================================================
def _process_single_cif(cif_path: Path, runs_dir: Path, args: argparse.Namespace) -> None:
    base_logger = setup_logger()
    config = WorkflowConfig(cif_path=cif_path, calc_type=args.calc_type)

    if args.resume:
        try:
            chosen_run_dir = _select_run_dir(runs_dir, args.run_id, args.run_path)
        except ValueError as e:
            base_logger.error(str(e))
            return

        run_ctx = RunContext(runs_dir, run_dir=chosen_run_dir)
        run_ctx.create_layout()
        log_file = run_ctx.run_dir / "logs" / args.log_name
        logger = setup_logger(log_file=log_file)
        logger.info(f"Resuming run in: {run_ctx.run_dir}")

        state = run_ctx.load_state()
        original_calc_type = state.get("calc_type")
        if original_calc_type is not None and original_calc_type != config.calc_type:
            logger.error(
                f"Run was initialized with calc_type='{original_calc_type}', "
                f"but resume was requested with --calc-type='{config.calc_type}'. "
                "Refusing to mix calculation types in one run."
            )
            return
        state["status"] = "resumed"
        run_ctx.save_state(state)
    else:
        run_ctx = RunContext(runs_dir)
        initial_state = {
            "cif": str(config.cif_path),
            "calc_type": config.calc_type,
            "status": "initialized",
            "tasks": {"crystal": {"status": "pending"}, "monomer": {"status": "pending"}},
        }
        run_ctx.initialize(initial_state)
        run_ctx.create_layout()
        log_file = run_ctx.run_dir / "logs" / args.log_name
        logger = setup_logger(log_file=log_file)
        logger.info(f"Run directory created: {run_ctx.run_dir}")

    logger.info("Run directory layout initialized")

    # TASK 1: STAGE CIF
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    if crystal_task.get("staged_cif") is not None:
        logger.info("Crystal CIF already staged; skipping copy")
    else:
        staged_path = run_ctx.stage_cif_for_crystal(config.cif_path)
        rel_path = staged_path.relative_to(run_ctx.run_dir)
        state.setdefault("tasks", {}).setdefault("crystal", {})
        state["tasks"]["crystal"]["status"] = "staged"
        state["tasks"]["crystal"]["staged_cif"] = str(rel_path)
        run_ctx.save_state(state)
        logger.info(f"Staged CIF for crystal: {rel_path}")

    # KPOINTS
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    staged_cif_rel = crystal_task.get("staged_cif")
    if staged_cif_rel is None:
        logger.error("Internal error: staged_cif missing from state.json")
        return
    staged_cif_abs = run_ctx.run_dir / staged_cif_rel

    if args.kpoints is not None:
        kpoints_to_use = args.kpoints
        logger.info(f"Using user-provided K_POINTS grid: {kpoints_to_use}")
    else:
        try:
            kpoints_to_use = _compute_kpoints_from_cif(staged_cif_abs, args.kpoint_separation)
        except Exception as e:
            logger.error(f"Failed to compute automatic K_POINTS from CIF: {e}")
            return
        logger.info(
            f"Computed automatic K_POINTS from CIF with separation={args.kpoint_separation}: {kpoints_to_use}"
        )

    state["tasks"]["crystal"]["kpoints"] = kpoints_to_use
    run_ctx.save_state(state)

    # TASK 2: GENERATE QE INPUT
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    if crystal_task.get("qe_input") is not None:
        logger.info("QE input already generated; skipping cif2cell step")
    else:
        if which("cif2cell") is None:
            logger.error("cif2cell not found in PATH. Did you 'module load' it?")
            return
        staged_cif_abs = run_ctx.run_dir / crystal_task["staged_cif"]
        crystal_dir = run_ctx.run_dir / "inputs" / "crystal"
        try:
            qe_in_path = _generate_qe_input_with_cif2cell(crystal_dir, staged_cif_abs, config.calc_type)
        except Exception as e:
            logger.error(str(e))
            return
        state["tasks"]["crystal"]["qe_input"] = str(qe_in_path.relative_to(run_ctx.run_dir))
        state["tasks"]["crystal"]["status"] = "inputs_generated"
        run_ctx.save_state(state)
        logger.info(f"Generated QE input via cif2cell: {qe_in_path.relative_to(run_ctx.run_dir)}")

    # TASK 3: PSEUDOPOTENTIAL CLEANUP
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    if crystal_task.get("pp_cleanup_done") and not args.force_pp_cleanup:
        logger.info("Pseudopotential cleanup already performed; skipping")
    else:
        if args.force_pp_cleanup and crystal_task.get("pp_cleanup_done"):
            logger.info("Force flag detected: re-running pseudopotential cleanup")

        if args.pseudo_dir is None:
            logger.info("No --pseudo-dir provided; skipping pseudopotential cleanup")
        else:
            pseudo_dir = Path(args.pseudo_dir)
            if not pseudo_dir.is_dir():
                logger.error(f"--pseudo-dir is not a directory: {pseudo_dir}")
                return

            pp_map: dict[str, str] | None = None
            if args.pp_map is not None:
                pp_map_file = Path(args.pp_map)
                if not pp_map_file.exists():
                    logger.error(f"--pp-map file does not exist: {pp_map_file}")
                    return
                try:
                    loaded = json.loads(pp_map_file.read_text())
                except Exception as e:
                    logger.error(f"Failed to parse --pp-map JSON: {e}")
                    return
                if not isinstance(loaded, dict):
                    logger.error(f"--pp-map must be a JSON object (dict), got {type(loaded).__name__}")
                    return
                for k, v in loaded.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        logger.error(f"--pp-map entries must be string -> string; got {k!r}: {v!r}")
                        return
                pp_map = loaded

            qe_in_rel = crystal_task.get("qe_input")
            if qe_in_rel is None:
                logger.error("Internal error: qe_input missing from state.json")
                return
            qe_in_abs = run_ctx.run_dir / qe_in_rel

            try:
                cleanup_info = _cleanup_qe_input(qe_in_abs, pseudo_dir, pp_map, config.calc_type, kpoints_to_use)
            except Exception as e:
                logger.error(f"Pseudopotential cleanup failed: {e}")
                return

            state["tasks"]["crystal"]["pp_cleanup_done"] = True
            state["tasks"]["crystal"]["pseudo_dir"] = cleanup_info["pseudo_dir"]
            state["tasks"]["crystal"]["pp_map_used"] = cleanup_info["pp_map_used"]
            state["tasks"]["crystal"]["pp_sources"] = cleanup_info["pp_sources"]
            run_ctx.save_state(state)
            logger.info(
                f"Pseudopotential cleanup complete: map={cleanup_info['pp_map_used']} sources={cleanup_info['pp_sources']}"
            )

    # TASK 4: GENERATE CRYSTAL SLURM SCRIPT
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    if crystal_task.get("job_script") is not None:
        logger.info("Crystal SLURM script already generated; skipping")
    else:
        qe_in_rel = crystal_task.get("qe_input")
        if qe_in_rel is None:
            logger.error("Internal error: qe_input missing; cannot generate SLURM script")
            return
        job_name = Path(qe_in_rel).stem
        try:
            slurm_path = run_ctx.write_crystal_slurm_script(
                qe_input_rel=qe_in_rel,
                job_name=job_name,
                walltime=args.slurm_walltime,
                ntasks=args.slurm_ntasks,
                qe_command=args.qe_command,
            )
        except Exception as e:
            logger.error(f"Failed to generate crystal SLURM script: {e}")
            return
        state["tasks"]["crystal"]["job_script"] = str(slurm_path.relative_to(run_ctx.run_dir))
        state["tasks"]["crystal"]["job_script_generated"] = True
        state["tasks"]["crystal"]["status"] = "job_script_generated"
        run_ctx.save_state(state)
        logger.info(f"Generated crystal SLURM script: {slurm_path.relative_to(run_ctx.run_dir)}")

    # TASK 5: SUBMIT CRYSTAL JOB
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    if crystal_task.get("job_id") is not None:
        logger.info(f"Crystal job already submitted with job_id={crystal_task['job_id']}; skipping sbatch")
    else:
        job_script_rel = crystal_task.get("job_script")
        if job_script_rel is None:
            logger.error("Internal error: job_script missing; cannot submit crystal job")
            return
        job_script_abs = run_ctx.run_dir / job_script_rel
        if not job_script_abs.exists():
            logger.error(f"Crystal SLURM script missing on disk: {job_script_abs}")
            return
        if which("sbatch") is None:
            logger.error("sbatch not found in PATH. Are you on a SLURM login node?")
            return
        try:
            job_id, raw_stdout = _submit_slurm_script(job_script_abs)
        except Exception as e:
            logger.error(str(e))
            return
        state["tasks"]["crystal"]["job_id"] = job_id
        state["tasks"]["crystal"]["submitted_at"] = datetime.now().isoformat(timespec="seconds")
        state["tasks"]["crystal"]["job_submit_stdout"] = raw_stdout
        state["tasks"]["crystal"]["status"] = "submitted"
        run_ctx.save_state(state)
        logger.info(f"Submitted crystal job with job_id={job_id}")

    # TASK 6: MONITOR CRYSTAL JOB
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    job_id = crystal_task.get("job_id")
    if job_id is None:
        logger.info("No submitted crystal job found; skipping monitoring")
    else:
        try:
            slurm_state = _query_slurm_job(job_id)
        except Exception as e:
            logger.error(f"Failed to query job status: {e}")
            return
        state["tasks"]["crystal"]["slurm_state"] = slurm_state
        if slurm_state == "PENDING":
            state["tasks"]["crystal"]["status"] = "pending"
        elif slurm_state == "RUNNING":
            state["tasks"]["crystal"]["status"] = "running"
        elif slurm_state == "COMPLETED":
            state["tasks"]["crystal"]["status"] = "completed"
        elif slurm_state in ("FAILED", "CANCELLED", "TIMEOUT"):
            state["tasks"]["crystal"]["status"] = slurm_state.lower()
        run_ctx.save_state(state)
        logger.info(f"Crystal job {job_id} status: {slurm_state}")

    # TASK 7: VALIDATE OUTPUT
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    slurm_state = crystal_task.get("slurm_state")
    if slurm_state not in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
        logger.info("Crystal job not in terminal state yet; skipping output validation")
    else:
        try:
            out_info = _inspect_crystal_outputs(run_ctx, crystal_task)
        except Exception as e:
            logger.error(f"Failed to inspect crystal outputs: {e}")
            return

        state["tasks"]["crystal"]["qe_output"] = {
            "qe_output_file": out_info["qe_output_file"],
            "qe_output_exists": out_info["qe_output_exists"],
            "qe_output_size": out_info["qe_output_size"],
            "slurm_output_file": out_info["slurm_output_file"],
            "slurm_output_exists": out_info["slurm_output_exists"],
            "slurm_output_size": out_info["slurm_output_size"],
            "validation_source": out_info["validation_source"],
            "qe_job_done": out_info["qe_job_done"],
            "qe_has_error": out_info["qe_has_error"],
            "qe_error_markers": out_info["qe_error_markers"],
        }

        if slurm_state == "COMPLETED" and out_info["qe_job_done"] and not out_info["qe_has_error"]:
            state["tasks"]["crystal"]["status"] = "qe_completed"
            logger.info(f"Crystal output validation successful using {out_info['validation_source']}: JOB DONE found")
        else:
            state["tasks"]["crystal"]["status"] = "qe_failed_validation"
            logger.info(
                "Crystal output validation failed: "
                f"slurm_state={slurm_state}, validation_source={out_info['validation_source']}, "
                f"qe_output_exists={out_info['qe_output_exists']}, slurm_output_exists={out_info['slurm_output_exists']}, "
                f"qe_job_done={out_info['qe_job_done']}, qe_has_error={out_info['qe_has_error']}"
            )
        run_ctx.save_state(state)

    # TASK 8: EXTRACT ENERGY
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    if crystal_task.get("status") != "qe_completed":
        logger.info("Crystal QE output not validated as completed; skipping energy extraction")
    else:
        if crystal_task.get("qe_results") is not None:
            state["tasks"]["crystal"]["status"] = "qe_energy_extracted"
            run_ctx.save_state(state)
            logger.info("Crystal QE results already extracted; skipping energy extraction")
        else:
            try:
                qe_results = _extract_qe_total_energy(crystal_task)
            except Exception as e:
                logger.error(f"Failed to extract QE total energy: {e}")
                return
            state["tasks"]["crystal"]["qe_results"] = qe_results
            state["tasks"]["crystal"]["status"] = "qe_energy_extracted"
            run_ctx.save_state(state)
            logger.info(f"Extracted crystal total energy: {qe_results['energy_ry']} Ry")

    # TASK 9: WRITE RESULT FILE
    state = run_ctx.load_state()
    crystal_task = state.get("tasks", {}).get("crystal", {})
    result_file = run_ctx.run_dir / "results" / "crystal_result.json"
    if crystal_task.get("qe_results") is None:
        logger.info("Crystal energy results not available yet; skipping result file writing")
    else:
        state["tasks"]["crystal"]["status"] = "qe_energy_extracted"
        run_ctx.save_state(state)
        if result_file.exists():
            logger.info("Crystal result file already exists; skipping result write")
        else:
            try:
                written = _write_crystal_result_file(run_ctx, state)
            except Exception as e:
                logger.error(f"Failed to write crystal result file: {e}")
                return
            logger.info(f"Wrote crystal result file: {written.relative_to(run_ctx.run_dir)}")

    logger.info("Configuration loaded successfully")
    logger.info(f"CIF path: {config.cif_path}")
    logger.info(f"Calc type: {config.calc_type}")
    logger.info(f"Current state: {run_ctx.load_state()['status']}")
    logger.info("Day 16 complete: workflow + multi-CIF aggregation available")


# ============================================================
# MAIN ENTRYPOINT / MULTI-CIF DRIVER + AGGREGATION
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="advQMcalc_test",
        description="Advanced QM calculator (single-CIF, multi-CIF, and aggregation modes)",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--cif", nargs="*", help="One or more CIF files")
    parser.add_argument("--cif-dir", default=None, help="Directory containing multiple CIF files")
    parser.add_argument("--cif-glob", default="*.cif", help="Glob pattern used with --cif-dir (default: *.cif)")
    parser.add_argument("--calc-type", default="scf", help="QE calculation type used for naming the generated input")
    parser.add_argument("--runs-dir", default="runs", help="Base directory for workflow runs")
    parser.add_argument("--log-name", default="advQMcalc.log", help="Run log filename")
    parser.add_argument("--resume", action="store_true", help="Resume a run (single or multi-CIF)")
    parser.add_argument("--run-id", default=None, help="Specific run directory name (single-CIF mode)")
    parser.add_argument("--run-path", default=None, help="Full path to a specific run to resume (single-CIF mode)")
    parser.add_argument("--pseudo-dir", default=None, help="Directory containing QE .UPF files")
    parser.add_argument("--pp-map", default=None, help="Optional JSON file mapping elements to .UPF filenames")
    parser.add_argument("--force-pp-cleanup", action="store_true", help="Force pseudopotential cleanup rerun")
    parser.add_argument("--slurm-walltime", default="12:00:00", help="Walltime for generated crystal SLURM script")
    parser.add_argument("--slurm-ntasks", type=int, default=8, help="Number of tasks for crystal SLURM script")
    parser.add_argument("--qe-command", default="pw.x", help="QE executable used in the generated SLURM script")
    parser.add_argument(
        "--kpoints",
        default=None,
        help='K_POINTS grid to use, e.g. "3 3 2 0 0 0". If omitted, it is computed from the CIF.',
    )
    parser.add_argument(
        "--kpoint-separation",
        type=float,
        default=0.03,
        help="Separation used for automatic k-point generation when --kpoints is omitted",
    )

    # Day 16: aggregation options
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate per-CIF run results into a summary instead of running the workflow",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output CSV file for aggregation summary",
    )
    parser.add_argument(
        "--summary-json-out",
        default=None,
        help="Output JSON file for aggregation summary",
    )
    parser.add_argument(
        "--summary-only-successful",
        action="store_true",
        help="Only include runs that reached qe_energy_extracted",
    )

    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    # ----- Day 16: aggregation branch (does NOT need --cif) -----
    if args.aggregate:
        agg_logger = setup_logger()
        base_runs_dir = Path(args.runs_dir)

        try:
            rows = _aggregate_crystal_results(
                runs_dir=base_runs_dir,
                only_successful=args.summary_only_successful,
            )
        except Exception as e:
            agg_logger.error(f"Aggregation failed: {e}")
            return

        agg_logger.info(f"Aggregated {len(rows)} crystal runs from {base_runs_dir}")

        if args.summary_out is not None:
            try:
                _write_summary_csv(rows, Path(args.summary_out))
                agg_logger.info(f"Wrote crystal summary CSV: {args.summary_out}")
            except Exception as e:
                agg_logger.error(f"Failed to write summary CSV: {e}")

        if args.summary_json_out is not None:
            try:
                _write_summary_json(rows, Path(args.summary_json_out))
                agg_logger.info(f"Wrote crystal summary JSON: {args.summary_json_out}")
            except Exception as e:
                agg_logger.error(f"Failed to write summary JSON: {e}")

        if args.summary_out is None and args.summary_json_out is None:
            agg_logger.info(
                "No --summary-out or --summary-json-out provided; summary not persisted to disk"
            )

        return

    # ----- Normal workflow path requires CIFs -----
    try:
        cif_paths = _collect_cif_paths(args)
    except ValueError as e:
        setup_logger().error(str(e))
        return

    base_runs_dir = Path(args.runs_dir)
    multi_mode = len(cif_paths) > 1 or args.cif_dir is not None

    if args.run_path is not None and multi_mode:
        setup_logger().error("--run-path is only supported in single-CIF mode")
        return
    if args.run_id is not None and multi_mode:
        setup_logger().error("--run-id is only supported in single-CIF mode")
        return

    for cif_path in cif_paths:
        runs_dir = _runs_dir_for_cif(base_runs_dir, cif_path, multi_mode)
        _process_single_cif(cif_path, runs_dir, args)


if __name__ == "__main__":
    main()
