import sys
from pathlib import Path


prompt_file, method, generated_dir, score_input_dir = sys.argv[1:]
lines = Path(prompt_file).read_text(encoding="utf-8-sig").splitlines()
prompts = [line.strip() for line in lines if line.strip()]
prefixes = [prompt[:100] for prompt in prompts]
if len(prompts) != 100 or len(set(prefixes)) != len(prompts):
    raise ValueError("Expected exactly 100 distinct prompt prefixes")

generated = Path(generated_dir).resolve()
score_input = Path(score_input_dir)
score_input.mkdir(parents=True, exist_ok=True)
suffix = "-0.mp4" if method == "self-forcing" else ".mp4"
expected = {prefix + suffix for prefix in prefixes}
actual = {entry.name for entry in generated.glob("*.mp4")}
trailing_blank = len(lines) == 101 and not lines[-1].strip()
if trailing_blank and suffix in actual:
    print(f"Excluded the terminal blank line output: {suffix}")
    actual.remove(suffix)
if actual != expected:
    raise ValueError(f"{method} output list differs from the 100 requested prompts")
for prefix in prefixes:
    video = generated / (prefix + suffix)
    if video.stat().st_size == 0:
        raise ValueError(f"Empty video: {video}")
    link = score_input / (prefix + ".mp4")
    if not (link.is_symlink() and link.resolve() == video):
        link.symlink_to(video)
