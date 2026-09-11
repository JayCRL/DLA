S=$(sed -n '/<body>/,$p' /tmp/msec.html | sed '1d')
cat > /tmp/mwrap.html <<HTML
<!doctype html><html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,Helvetica,sans-serif;font-size:14px;line-height:1.6;max-width:880px;padding:28px}
code{font-family:Menlo,monospace;font-size:12px;background:#f2f2f2;padding:1px 4px;border-radius:3px}
h1{font-size:20px}h3{font-size:15px;margin-top:22px}
table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccc;padding:4px 8px}
math{font-size:16px}
</style></head><body>
$S
</body></html>
HTML
