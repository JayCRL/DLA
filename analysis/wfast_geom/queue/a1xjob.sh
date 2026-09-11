#!/bin/bash
# fields: tag|seed|variant|gamma|sleepdecay
IFS="|" read -r tag s v g d <<< "$1"
extra=""
[ "$g" != "-" ] && extra="$extra --gamma $g"
[ "$d" != "-" ] && extra="$extra --sleep-decay $d"
mkdir -p "$HOME/llm-lab/dla_audit/$tag"
"$HOME/.venv/bin/python" "$HOME/Desktop/dla-v0.2/analysis/wfast_geom/audit_b3.py" --seed "$s" --order HE --variant "$v" $extra --out "$HOME/llm-lab/dla_audit/$tag"
