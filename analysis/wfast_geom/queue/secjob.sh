#!/bin/bash
s="$1"
"$HOME/.venv/bin/python" "$HOME/Desktop/dla-v0.2/analysis/wfast_geom/audit_second.py" --seed "$s" --span 26000 --out "$HOME/llm-lab/dla_audit_second"
