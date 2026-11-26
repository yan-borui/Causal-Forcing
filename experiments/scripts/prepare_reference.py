import shutil
import sys
from pathlib import Path
from zipfile import ZipFile


root = Path(sys.argv[1])
data = root / 'data'
prompts = [line.strip() for line in (data / 'text_align.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
prefixes = {prompt[:100] for prompt in prompts}
if len(prompts) != 100 or len(prefixes) != len(prompts):
    raise ValueError('Expected 100 unique evaluation prompts and filename prefixes')

for model in ('causal-forcing', 'self-forcing'):
    output = data / 'reference' / model
    output.mkdir(parents=True, exist_ok=True)
    with ZipFile(data / (model + '.zip')) as archive:
        files = [entry for entry in archive.infolist() if entry.filename.endswith('.mp4')]
        if len(files) != len(prompts) or {Path(entry.filename).stem for entry in files} != prefixes:
            raise ValueError(f'{model} videos do not match the published prompts')
        for entry in files:
            dest = output / Path(entry.filename).name
            with archive.open(entry) as source, dest.open('wb') as target:
                shutil.copyfileobj(source, target)

raft = root / 'vbench-cache' / 'raft_model' / 'models' / 'raft-things.pth'
raft.parent.mkdir(parents=True, exist_ok=True)
with ZipFile(data / 'raft-models.zip') as archive:
    with archive.open('models/raft-things.pth') as source, raft.open('wb') as target:
        shutil.copyfileobj(source, target)
