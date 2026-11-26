import json
from pathlib import Path
import sys


def save_once(name, data):
    target = Path(name)
    if target.exists() and json.loads(target.read_text()) != data:
        raise ValueError(f"Existing run configuration differs: {name}")
    if not target.exists():
        target.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main():
    stage, *extra = sys.argv[1:]
    prompts = json.loads(Path("../segment-pilot-20261124/prompts.json").read_text())
    if stage == "paired" and not extra:
        methods = ["capped", "two_stage"]
    elif stage == "replicate" and extra in (["capped"], ["two_stage"]):
        methods = ["attention", "mixture", extra[0]]
        for prompt in prompts:
            prompt["seed"] = 34000 + prompt["id"]
    else:
        raise ValueError("Use paired, or replicate with one selected intervention")
    save_once("prompts.json", prompts)
    save_once("methods.json", methods)


if __name__ == "__main__":
    main()
