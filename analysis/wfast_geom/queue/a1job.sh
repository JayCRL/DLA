#!/bin/bash
IFS="|" read -r gd s o v g <<< "$1"
if [ "$gd" = "g1" ]; then tgt="$HOME/llm-lab/dla_audit"; else tgt="$HOME/llm-lab/dla_audit/$gd"; fi
"$HOME/.venv/bin/python" "$HOME/Desktop/dla-v0.2/analysis/wfast_geom/audit_b3.py" --seed "$s" --order "$o" --variant "$v" --gamma "$g" --out "$tgt"
