#!/bin/bash
p="$1"
enc=$(printf '%s' "$p" | sed 's|/|%2f|')
out=$(curl -sS -m 12 "https://registry.npmjs.org/@deepseek-ai%2f$enc" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin); vs=d.get('versions',{})
    print('YES' if '0.1.5-rc.1' in vs else 'NO:'+','.join(list(vs)[-2:]))
except Exception: print('MISSING')
" 2>/dev/null)
echo "${out:-MISSING}|$p"
