"""Plot measured layer and time profiles, and project source-91 tokens onto the image."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output'
SEEDS = (34001,24017,34017,34079)
LABELS = ('Workshop 34001','Stairs 24017','Stairs 34017','Cyclist 34079')
COLORS = ('#9b4a13','#b33648','#2369a0','#45816c')


def numeric(name):
    with (OUT/name).open() as f:
        return [{k:float(v) for k,v in r.items() if v} for r in csv.DictReader(f)]


def main():
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    profile=numeric('late-layer-profile.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,7.4),layout='constrained')
    for seed,label,color in zip(SEEDS,LABELS,COLORS):
        rows=sorted((r for r in profile if r['seed']==seed),key=lambda r:r['layer'])
        for ax,metric in zip(axes[0],('entropy_normalized','selected_effective')):
            ax.plot([r['layer'] for r in rows],[r[metric] for r in rows],color=color,label=label,lw=1.65)
    for ax in axes[0]:
        ax.set(xlabel='Transformer layer',xlim=(0,29),xticks=(0,5,10,15,20,25,29))
        ax.grid(axis='y',alpha=.18)
    axes[0,0].set(title='Token attention entropy, 20–30 s',ylabel='Entropy / log(number of keys)')
    axes[0,0].legend(ncol=2,fontsize=8,loc='lower left',frameon=False)
    axes[0,1].set(title='Source frames retained, 20–30 s',ylabel='Effective number of source frames')
    stairs=numeric('staircase-sink.csv')
    ax=axes[1,0]
    for seed,label,color in zip(SEEDS[1:3],LABELS[1:3],COLORS[1:3]):
        rows=[r for r in stairs if r['seed']==seed and r['t0_s']>=20]
        ax.plot([(r['t0_s']+r['t1_s'])/2 for r in rows],[r['sink_pct'] for r in rows],
                marker='o',markersize=3,lw=1.6,label=label,color=color)
    ax.axvspan(28.125,28.5,color='#d5cabf',alpha=.6,label='Window frame disappears')
    ax.set(title='Sink attention around the staircase transition',xlabel='Decoded time (s)',
           ylabel='Mean attention to original Sink (%)',xlim=(20,30.5))
    ax.legend(frameon=False,fontsize=8)
    ax.grid(axis='y',alpha=.18)
    ax=axes[1,1]
    img_file=ROOT/'workshop-363.png'
    if not img_file.exists():
        img_file=OUT/'workshop-363.png'
    ax.imshow(Image.open(img_file))
    patches=json.loads((OUT/'34001'/'patches.json').read_text())
    kept=np.array(patches['kept'])
    ax.scatter(16*(kept%52)+8,16*(kept//52)+8,s=1.9,c='#a2f0ff',edgecolor='none',alpha=.68)
    ax.add_patch(Rectangle((448,144),96,144,fill=False,linewidth=2,edgecolor='#fff021'))
    ax.set(title='Workshop, 22.69 s · 1,155 retained patches at layer 0')
    ax.axis('off')
    fig.savefig(OUT/'diagnostics.png',dpi=160)
    plt.close(fig)
    matrices=[]
    for seed in SEEDS:
        with (OUT/str(seed)/'attention.csv').open() as f:
            rows=list(csv.DictReader(f))
        mat=np.full((30,41),np.nan)
        for r in rows:
            mat[int(r['layer']),int(r['chunk'])//3]=100*float(r['sink_mass'])
        matrices.append(mat)
    fig,axs=plt.subplots(2,2,figsize=(11,6),sharex=True,sharey=True,layout='constrained')
    # Pixel-frame cell edges: the first compressed latent accounts for one decoded frame.
    edges=[0]+[(4*(3*i)-3)/16 for i in range(1,41)]+[489/16]
    for ax,mat,title in zip(axs.flat,matrices,LABELS):
        im=ax.pcolormesh(edges,np.arange(31)-.5,mat,cmap='viridis',vmin=0,vmax=100,shading='flat')
        ax.set(title=title,xlim=(0,489/16),ylim=(-.5,29.5),yticks=(0,7,14,21,29))
    for ax in axs[1]:
        ax.set_xlabel('Decoded time (s)')
    for ax in axs[:,0]:
        ax.set_ylabel('Transformer layer')
    fig.colorbar(im,ax=axs,label='Attention to original Sink (%)')
    fig.savefig(OUT/'sink-all-layers.png',dpi=155)
    plt.close(fig)


if __name__=='__main__':
    main()
