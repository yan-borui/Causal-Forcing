"""Render the same workshop and peacock moments with three cache rules, on CPU."""
from pathlib import Path
import sys


RUN = Path(__file__).resolve().parent
ORIGINAL_CAP = RUN.parent / "quota-study-20261124"
PILOT = RUN.parent / "segment-pilot-20261124"
sys.path.insert(0, str(ORIGINAL_CAP))
from visualize import COLORS, NAMES, cases, load


def main():
    sources = {"attention": PILOT / "output", "capped": ORIGINAL_CAP / "output",
               "capped_015": RUN / "output"}
    NAMES.update(attention="Deep Forcing", capped="50% + cap", capped_015="15% + cap")
    COLORS["capped_015"] = "#6c4a99"
    figures = RUN / "output/figures"
    figures.mkdir(exist_ok=True)
    for ident, label in ((1, "Workshop hammer"), (86, "Peacock")):
        rows = []
        for method, source in sources.items():
            score = load(source / "scores/subject" / f"{ident:03d}-{method}.json")
            rows.append(dict(window="2", id=ident, method=method, subject=100*score["segments"][2]["subject"]))
        cases(ident, label, list(sources), sources, rows, figures)


if __name__ == "__main__":
    main()
