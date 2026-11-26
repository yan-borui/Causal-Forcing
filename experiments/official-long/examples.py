"""Show the same two diagnostic prompts across the four reported systems."""
from pathlib import Path

import av
from PIL import Image, ImageDraw


def main():
    run=Path.cwd()
    methods=('Causal Forcing (official long)','Deep Forcing','Frame diversity (additive)','Frame diversity + cap')
    cases=((34001, (8,363,456)),(24017,(8,444,456)))
    page=Image.new('RGB',(185+4*247,45+6*145),'white')
    draw=ImageDraw.Draw(page)
    for column,name in enumerate(methods):
        draw.text((190+247*column,15),name,fill='black')
    for row,(seed,indices) in enumerate(cases):
        ident=seed%1000
        archive='segment-pilot-20261124' if seed//1000==24 else 'quota-replication-20261124'
        quota='quota-study-20261124' if seed//1000==24 else archive
        inputs=[run/'output'/'videos'/f'{seed}.mp4',
                run.parent/archive/'output'/'videos'/f'{ident:03d}-attention.mp4',
                run.parent/archive/'output'/'videos'/f'{ident:03d}-mixture.mp4',
                run.parent/quota/'output'/'videos'/f'{ident:03d}-capped.mp4']
        for line,i in enumerate(indices):
            y=45+(3*row+line)*145
            draw.text((6,y+38),('Workshop' if row==0 else 'Stairs')+f'  {seed}',fill='black')
            draw.text((6,y+65),f'{i/16:.2f} s',fill='black')
        for column,video in enumerate(inputs):
            with av.open(str(video)) as stream:
                for i,frame in enumerate(stream.decode(video=0)):
                    if i in indices:
                        y=45+(3*row+indices.index(i))*145
                        pic=frame.to_image().resize((242,139),Image.Resampling.LANCZOS)
                        page.paste(pic,(185+247*column,y))
    page.save(run/'output'/'paired-examples.jpg',quality=94)


if __name__=='__main__':
    main()
