#!/bin/bash
IFS="|" read -r s v <<< "$1"
"$HOME/.venv/bin/python" "$HOME/Desktop/dla-v0.2/analysis/wfast_geom/audit_t10.py" --seed "$s" --variant "$v" --out "$HOME/llm-lab/dla_audit_t10"
