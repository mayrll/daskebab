from pathlib import Path
import html
import json

agents = """my frontier secure rush efficient economy counter endgame phased ratio bank tactical reply sequence tempo closing forecast expansion search endsearch defense preserve pressure frugal reserve stable""".split()
evidence = Path("evidence")
rows = []
for number, agent in enumerate(agents, 1):
    stem = f"attempt-{number}"
    journal = evidence / f"{stem}.events.jsonl"
    last = json.loads(journal.read_text(encoding="utf-8").splitlines()[-1])
    summary = last.get("summary", last.get("session"))
    files = [evidence / (stem + ext) for ext in (".json", ".events.jsonl", ".html", ".chart.json", ".chart.html")]
    links = " ".join(f'<a href="{p.as_posix()}">{p.name[len(stem):].lstrip(".")}</a>' for p in files if p.exists())
    source = f"{agent}_agent.py"
    command = f"python evaluate.py --agent {source} --output evidence/{stem}.json"
    rows.append(f'<tr><td>{number}</td><td><a href="{source}">{source}</a><br><code>{html.escape(command)}</code></td><td>{summary["completed"]}</td><td>{summary["wins"]}</td><td>{summary["draws"]}</td><td>{summary["winning_groups"]}</td><td>{links}</td></tr>')
Path("explanation.html").write_text("""<!DOCTYPE html>
<html lang="en"><meta charset="utf-8"><title>Island Duel evaluation evidence</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem} td,th{padding:.55rem;text-align:left;border-bottom:1px solid #ddd} code{font-size:.85em}</style>
<h1>Island Duel evaluation evidence</h1>
<p>Requires Python 3.10+ and the native engine from the participant bundle. No third-party Python packages are required.</p>
<p>The policies rank legal actions by score gain, energy cost, and expansion potential. Later policies estimate the strongest visible countercapture, reward affordable follow-up moves, and adjust play as the end of the match approaches.</p>
<p>Qualification requires 308 wins across 512 fresh games and wins in at least 24 groups.</p>
<p>Two preliminary launches failed at import time, before opening an evaluation session. The corrected launches are attempts 5 and 11.</p>
<p><a href="evidence/practice.html">Practice replay</a> for arena 0, seed 144.</p>
<table><thead><tr><th>Attempt</th><th>Agent and command</th><th>Games</th><th>Wins</th><th>Draws</th><th>Groups won</th><th>Evidence</th></tr></thead><tbody>
""" + "\n".join(rows) + "\n</tbody></table></html>\n", encoding="utf-8")

