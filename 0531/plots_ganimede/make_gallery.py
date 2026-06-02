from pathlib import Path

imgs = sorted(Path(".").glob("*.png"))

ncol = 3

lines = []
lines.append("# plots_ganimede")
lines.append("")
lines.append(f"{len(imgs)} images")
lines.append("")
lines.append("| " + " | ".join([""] * ncol) + " |")
lines.append("| " + " | ".join(["---"] * ncol) + " |")

cells = []
for img in imgs:
    name = img.name
    cells.append(f'<a href="{name}"><img src="{name}" width="260"><br>{name}</a>')

for i in range(0, len(cells), ncol):
    row = cells[i:i+ncol]
    while len(row) < ncol:
        row.append("")
    lines.append("| " + " | ".join(row) + " |")

Path("README.md").write_text("\n".join(lines), encoding="utf-8")

print("README.md generated")
