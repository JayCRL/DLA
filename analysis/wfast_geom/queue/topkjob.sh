#!/bin/bash
# One topk_signed job: fields "frac|seed".  Variant is fixed (topk_signed) and so is the
# arm (HE) because HE is the arm that carries the allocation effect; the EH arm is flat.
#
# The output path follows the project's {tag}/{variant}/ convention, so the frac is visible
# in the directory name exactly like g0.5/ and d0.25/ for the write-strength sweep:
#   ~/llm-lab/dla_audit/topk0.2/topk_signed/probe_seed3_HE.json
IFS="|" read -r frac s <<< "$1"
PY="$HOME/.venv/bin/python"
SCRIPT="$HOME/Desktop/dla-v0.2/analysis/wfast_geom/audit_b3.py"
TAG="$HOME/llm-lab/dla_audit/topk$frac"
JSON="$TAG/topk_signed/probe_seed${s}_HE.json"

mkdir -p "$TAG"
# Resume rather than recompute: a finished seed is left alone.
if [ -f "$JSON" ]; then echo "SKIP topk$frac seed $s"; exit 0; fi

echo "RUN  topk$frac seed $s"
"$PY" "$SCRIPT" --seed "$s" --order HE --variant topk_signed --topk-frac "$frac" --out "$TAG" \
  2>&1 | grep --line-buffered -E "seed |Error|Traceback|out of memory"
# A failed seed must never be mistaken for a finished one.
[ -f "$JSON" ] || echo "** FAILED topk$frac seed $s (no output file)"
