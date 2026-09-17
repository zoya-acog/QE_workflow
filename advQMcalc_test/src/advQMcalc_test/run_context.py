from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from shutil import copy2
from textwrap import dedent


class RunContext:
    """
    Represents a single workflow run.
    Owns the run directory and state.json.
    """

    def __init__(self, base_dir: Path, run_dir: Path | None = None):
        self.base_dir = base_dir

        if run_dir is None:
            self.run_dir = self._generate_unique_run_dir(base_dir)
        else:
            self.run_dir = run_dir

        self.state_file = self.run_dir / "state.json"

    def _generate_unique_run_dir(self, base_dir: Path) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        candidate = base_dir / ts

        if not candidate.exists():
            return candidate

        for k in range(1, 1000):
            candidate_k = base_dir / f"{ts}_{k:02d}"
            if not candidate_k.exists():
                return candidate_k

        raise RuntimeError(f"Could not generate a unique run directory under {base_dir}")

    def initialize(self, initial_state: dict):
        self.run_dir.mkdir(parents=True, exist_ok=False)
        with open(self.state_file, "w") as f:
            json.dump(initial_state, f, indent=2)

    def load_state(self) -> dict:
        with open(self.state_file, "r") as f:
            return json.load(f)

    def save_state(self, state: dict):
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def create_layout(self):
        (self.run_dir / "inputs" / "crystal").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "inputs" / "monomer").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "jobs" / "crystal").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "jobs" / "monomer").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "results").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

    def stage_cif_for_crystal(self, source_cif: Path) -> Path:
        dest = self.run_dir / "inputs" / "crystal" / source_cif.name
        copy2(source_cif, dest)
        return dest

    def write_crystal_slurm_script(
        self,
        qe_input_rel: str,
        job_name: str,
        walltime: str = "2:00:00",
        ntasks: int = 8,
        nodes: int = 1,
        mem_per_cpu: str = "4G",
        qe_command: str = "pw.x",
    ) -> Path:
        """
        Write a SLURM script for the crystal QE calculation.

        Defaults requested:
          - #SBATCH --nodes=1
          - #SBATCH --mem-per-cpu=4G
          - #SBATCH --time=2:00:00
        """
        qe_input_path = self.run_dir / qe_input_rel
        if not qe_input_path.exists():
            raise RuntimeError(f"QE input file not found for SLURM generation: {qe_input_path}")

        script_path = self.run_dir / "jobs" / "crystal" / f"{job_name}.slurm"
        slurm_out_path = (script_path.parent / f"{job_name}.slurm.out").resolve()
        input_dir = qe_input_path.parent.resolve()

        script_text = dedent(
            f"""\
            #!/bin/bash
            #SBATCH --job-name={job_name}
            #SBATCH --output={slurm_out_path}
            #SBATCH --partition=pzero
            #SBATCH --nodes={nodes}
            #SBATCH --nodelist=own9
            #SBATCH --cpus-per-task=1
            #SBATCH --ntasks={ntasks}
            #SBATCH --mem-per-cpu={mem_per_cpu}
            #SBATCH --time={walltime}

            set -e

            export LD_LIBRARY_PATH=/mnt/own6a/zoya/QM_Workflow/qe_deps/lib:$LD_LIBRARY_PATH
            PW_BIN=/mnt/own6a/zoya/QM_Workflow/qe-7.2-install/bin/pw.x

            infile=$(basename "{qe_input_path.name}" .in)
            indir="{input_dir}"
            workdir="$indir/work_$SLURM_JOBID"

            mkdir -p "$workdir" && cd "$workdir" && echo "$workdir"
            cp -r "$indir/$infile.in" "$workdir"

            # QE's scratch/coordination files (*.save/, pwscf.para,
            # pwscf.restart, ...) race across MPI ranks when written to the
            # NFS-mounted workdir — NFS's weaker concurrent-unlink semantics
            # cause "File cannot be deleted" Fortran errors (exit code 2)
            # even when the calculation itself converges correctly and
            # prints JOB DONE. disk_io='none' alone isn't enough: QE still
            # writes/deletes small MPI coordination files at shutdown
            # regardless of that setting. Point outdir at node-local scratch
            # instead — only the small .out file (captured via the stdout
            # redirect below) needs to survive, we don't use
            # restart-from-checkpoint anywhere in this pipeline.
            localscratch="/tmp/qe_scratch_$SLURM_JOBID"
            mkdir -p "$localscratch"
            if grep -q "outdir" "$infile.in"; then
                sed -i "s#^\\s*outdir.*#  outdir = '$localscratch'#" "$infile.in"
            else
                sed -i "/&CONTROL/a\\  outdir = '$localscratch'" "$infile.in"
            fi
            if grep -q "disk_io" "$infile.in"; then
                sed -i "s/^\\s*disk_io.*/  disk_io = 'none'/" "$infile.in"
            else
                sed -i "/&CONTROL/a\\  disk_io = 'none'" "$infile.in"
            fi

            srun "$PW_BIN" -i "$infile.in" > "$infile.out"
            rm -rf "$localscratch"
            cp "$infile.out" "$indir"
            """
        )

        script_path.write_text(script_text)
        return script_path