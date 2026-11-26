"""Align reviewer-identified moments to the nearest preceding cache eviction; reuse existing videos."""
import csv
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

from baseline_review import font, pictures, table


# Start of a six-picture strip at 0.25 s intervals, and the event to align.
EVENTS = [(24017,17,27.75,28.5),(34001,1,22.25,22.75),(24001,1,23.0,23.5),
          (34011,11,25.0,25.5),(24011,11,22.75,23.5),(34017,17,27.75,28.5)]


def luminance(stream_file):
    sample = []
    with av.open(str(stream_file)) as stream:
        for i,frame in enumerate(stream.decode(video=0)):
            if i>=480: break
            if i%4: continue
            image=frame.to_image().convert("L").resize((208,120),Image.Resampling.BOX)
            level=np.asarray(image)
            sample.append((level.mean(),100*(level<32).mean()))
    if len(sample)!=120: raise ValueError("Incomplete 4-fps luminance sample")
    return np.array(sample).reshape(3,40,2).mean(axis=1)


def main():
    dest=Path("baseline")
    with (dest/"cache-events.csv").open() as stream: records=list(csv.DictReader(stream))
    tw,th=200,116
    sheet=Image.new("RGB",(6*(tw+5)+10,45+len(EVENTS)*(th+42)),"white")
    pen=ImageDraw.Draw(sheet)
    pen.text((8,7),"Late-window events and the alternate staircase seed",font=font(20,True),fill="#203945")
    matches=[]
    for pos,(seed,ident,start,mark) in enumerate(EVENTS):
        src="segment-pilot-20261124" if seed<30000 else "quota-replication-20261124"
        video=Path("..")/src/"output/videos"/f"{ident:03d}-attention.mp4"
        times=[start+j/4 for j in range(6)]
        imgs=pictures(video,{round(t*16) for t in times},(tw,th))
        y=41+pos*(th+42)
        for j,t in enumerate(times):
            x=5+j*(tw+5)
            sheet.paste(imgs[round(t*16)],(x,y))
            pen.text((x,y+th+1),f"{seed}   /   {t:.2f} s",font=font(13),fill="#253942")
        timeline=[r for r in records if int(r["seed"])==seed and float(r["decision_s"])<=mark]
        decision=max(float(r["decision_s"]) for r in timeline)
        light=luminance(video)
        for event in timeline:
            if float(event["decision_s"])!=decision: continue
            matches.append(dict(id=ident,seed=seed,visible_event_s=mark,decision_s=decision,
                                layer=event["layer"],effective=event["effective"],peak_pct=event["peak_pct"],
                                source_1_s=event["source_1_s"],older_than_5s_pct=event["older_than_5s_pct"],
                                mean_luma_early=light[0,0],mean_luma_late=light[2,0],
                                dark_pct_early=light[0,1],dark_pct_late=light[2,1]))
    sheet.save(dest/"observed-events.jpg",quality=93)
    table(dest/"observed-events.csv",matches)


if __name__=="__main__":
    main()
