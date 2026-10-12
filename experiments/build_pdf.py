"""reports/REPORT.md -> reports/REPORT.pdf (Markdown -> styled HTML -> headless Edge print).

usage: python experiments/build_pdf.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.42; color: #0b0b0b; }
h1 { font-size: 18pt; margin: 0 0 6pt; }
h2 { font-size: 13.5pt; margin: 16pt 0 6pt; border-bottom: 1px solid #e4e3df; padding-bottom: 3pt;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
em { color: #52514e; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; background: #f3f2ef;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f3f2ef; padding: 6pt 8pt; border-radius: 3pt; font-size: 8.3pt; line-height: 1.3;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; font-size: 8.3pt; margin: 6pt 0 8pt; break-inside: avoid; }
th, td { border-bottom: 1px solid #e4e3df; padding: 2.5pt 4pt; text-align: left; vertical-align: top; }
th { background: #f3f2ef; font-weight: 600; }
img { max-width: 100%; display: block; margin: 6pt auto; break-inside: avoid; }
"""


def main() -> None:
    md = (REP / "REPORT.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Автоматическая генерация '
            f'признаков для RecSys</title><style>{CSS}</style></head><body>{body}</body></html>')
    tmp = REP / "_report_print.html"
    tmp.write_text(html, encoding="utf-8")
    edge = next((e for e in EDGE if e.exists()), None)
    if edge is None:
        sys.exit("Edge not found; open reports/_report_print.html in a browser and print to PDF")
    out = REP / "REPORT.pdf"
    import shutil
    import tempfile
    prof = tempfile.mkdtemp(prefix="edge_pdf_")   # own profile: do not hand off to a running Edge
    tmp_pdf = Path(prof) / "report.pdf"           # print into the temp dir, then copy into the repo
    # classic headless + --no-sandbox: Edge's own sandbox fails inside restricted shells; it only renders
    # our local HTML file
    subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                    "--no-first-run", f"--user-data-dir={prof}",
                    f"--print-to-pdf={tmp_pdf}", tmp.resolve().as_uri()], check=True, timeout=180)
    import time
    for _ in range(30):                       # the launcher can return before the file is written
        if tmp_pdf.exists() and tmp_pdf.stat().st_size > 0:
            break
        time.sleep(1)
    time.sleep(1)
    shutil.copyfile(tmp_pdf, out)
    tmp.unlink()
    # headless Edge can linger and block the next print; stop only processes using OUR temp profile
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_Process -Filter \"Name='msedge.exe'\" | Where-Object { $_.CommandLine "
                    f"-like '*{Path(prof).name}*' }} | ForEach-Object {{ Stop-Process -Id $_.ProcessId -Force "
                    "-ErrorAction SilentlyContinue }"], check=False)
    from pypdf import PdfReader
    print(f"{out}  pages={len(PdfReader(out).pages)}  size={out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
