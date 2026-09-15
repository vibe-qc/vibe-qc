#!/usr/bin/env python3
"""Generate the website's outlined SVG brand artwork, without network access.

Requires fontTools (see ../artwork/requirements.txt). Raster exports are made
by gen-social-card.mjs. Do not edit generated files under public/logo by hand.
"""
from html import escape
from pathlib import Path
from functools import lru_cache
import math

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from fontTools.pens.svgPathPen import SVGPathPen

SITE = Path(__file__).resolve().parents[1]
OUT = SITE / 'public/logo'
OUT.mkdir(parents=True, exist_ok=True)
PAPER, INK, LIGHT, MUTED = '#08120F', '#0F172A', '#ECF3F1', '#9DB1AB'
COLORS = {'vibe-qc': ('#0F766E', '#2DB9A8'), 'vibe-view': ('#7151B8', '#BAA4FA'), 'vibe-queue': ('#A55325', '#F1AD80')}
# The product name, which is what a wordmark shows. `vq` stays the shell
# command, Python distribution and import package (see the vibe-queue
# independence note in src/data/products.mjs); it is not the product name.
NAMES = {'vibe-qc': 'vibe-qc', 'vibe-view': 'vibe-view', 'vibe-queue': 'vibe-queue'}

@lru_cache(None)
def font(weight):
    f = instantiateVariableFont(TTFont(SITE / 'artwork/fonts/Manrope.ttf'), {'wght': weight}, inplace=True)
    return f.getGlyphSet(), f.getBestCmap(), f['head'].unitsPerEm

def lettering(label, x, y, size, color=LIGHT, weight=650, tracking=0, limit=None):
    glyphs, cmap, upm = font(weight)
    advance = sum(glyphs[cmap[ord(c)]].width for c in label) * size / upm + max(0, len(label)-1) * tracking
    if limit is not None and advance > limit:
        raise ValueError(f'Text exceeds its layout: {label!r}: {advance:.1f} > {limit}')
    body, cursor = [], 0
    for char in label:
        glyph = glyphs[cmap[ord(char)]]
        pen = SVGPathPen(glyphs, ntos=lambda v: f'{v:.2f}'.rstrip('0').rstrip('.') if v else '0')
        glyph.draw(pen)
        if pen.getCommands():
            body.append(f'<path transform="translate({cursor:.3f},0)" d="{pen.getCommands()}"/>')
        cursor += glyph.width + tracking * upm / size
    return f'<g aria-label="{escape(label)}" fill="{color}" transform="translate({x},{y}) scale({size/upm},-{size/upm})">'+''.join(body)+'</g>', advance

def text(*args, **kwargs): return lettering(*args, **kwargs)[0]

def symbol(kind='vibe-qc', color='#0F766E', background=None):
    frame = f'<rect x="2" y="2" width="44" height="44" rx="11" fill="{background or "none"}" stroke="{color}" stroke-width="2.5"/>'
    if kind == 'vibe-qc':
        drawing = '<path d="M11 24C15 10 20 10 24 24S33 38 37 24"/><circle cx="11" cy="24" r="2.8"/><circle cx="37" cy="24" r="2.8"/>'
    elif kind == 'vibe-view':
        drawing = '<path d="M8 24Q24 5 40 24Q24 43 8 24Z"/><circle cx="24" cy="24" r="6"/>'
    else:
        drawing = '<path d="M12 14H24V34H36"/><circle cx="12" cy="14" r="3.2"/><circle cx="24" cy="24" r="3.2"/><circle cx="36" cy="34" r="3.2"/>'
    return frame + f'<g stroke="{color}" stroke-width="2.7" stroke-linecap="round" stroke-linejoin="round" fill="{background or "none"}">{drawing}</g>'

def svg(body, width, height, title, desc):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>{body}</svg>\n'

def wordmark(kind, dark=False):
    color = COLORS[kind][int(dark)]
    paths, width = lettering(NAMES[kind], 63, 37, 43, LIGHT if dark else INK, 650, -1.0)
    return symbol(kind,color) + paths, math.ceil(66+width)

for kind in NAMES:
    for dark in (False, True):
        body, width = wordmark(kind,dark)
        (OUT/f'{kind}-wordmark-{"dark" if dark else "light"}.svg').write_text(svg(body,width,48,NAMES[kind],f'{NAMES[kind]} product wordmark with outlined Manrope lettering and a framed scientific symbol.'))
    (OUT/f'{kind}-favicon.svg').write_text(svg(symbol(kind, '#FFFFFF', COLORS[kind][0]),48,48,NAMES[kind],f'{NAMES[kind]} product icon.'))

def tile(kind, x, y, scale=1):
    color=COLORS[kind][1]
    return f'<g transform="translate({x},{y}) scale({scale})"><rect width="112" height="112" rx="25" fill="#10251F" stroke="{color}" stroke-opacity=".4"/>'+f'<g transform="translate(26,24) scale(1.25)">{symbol(kind,color)}</g></g>'

def family_diagram():
    return '''<g fill="none" stroke="#31584D" stroke-width="2"><path d="M901 248C960 248 1005 258 1018 319"/><path d="M1018 375C1018 425 917 451 887 404"/><path d="M843 353C825 310 815 261 854 241"/></g>'''+tile('vibe-qc',824,156)+tile('vibe-view',982,296)+tile('vibe-queue',794,355)

def lattice_diagram():
    # A projected unit-cell network; a visual motif, not a physical structure.
    points={(x,y,z):(836+x*93+y*48,211-y*39+z*105) for x in range(3) for y in range(2) for z in range(2)}
    result=[]
    for p,(x,y) in points.items():
        for d in ((1,0,0),(0,1,0),(0,0,1)):
            q=tuple(a+b for a,b in zip(p,d))
            if q in points:
                tx,ty=points[q];result.append(f'<path d="M{x} {y}L{tx} {ty}" stroke="#2DB9A8" stroke-opacity=".45" stroke-width="2"/>')
    for x,y in points.values():result.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#75DBCA"/>')
    result.append('<ellipse cx="925" cy="270" rx="71" ry="28" transform="rotate(-32 925 270)" fill="url(#teal)"/><ellipse cx="1020" cy="313" rx="62" ry="26" transform="rotate(-32 1020 313)" fill="url(#coral)"/>')
    return ''.join(result)

def viewer_diagram():
    return '''<ellipse cx="889" cy="269" rx="80" ry="47" transform="rotate(-28 889 269)" fill="url(#teal)" stroke="#2DB9A8"/>
<ellipse cx="1030" cy="327" rx="80" ry="47" transform="rotate(-28 1030 327)" fill="url(#violet)" stroke="#BAA4FA"/>
<g fill="none" stroke="#BAA4FA" stroke-opacity=".55"><ellipse cx="958" cy="298" rx="152" ry="85" transform="rotate(-28 958 298)"/><ellipse cx="958" cy="298" rx="152" ry="35" transform="rotate(-28 958 298)"/><path d="M839 215L1079 382M846 384L1068 204"/></g>
<circle cx="958" cy="298" r="11" fill="#ECF3F1"/><circle cx="882" cy="352" r="6" fill="#2DB9A8"/><circle cx="1030" cy="244" r="6" fill="#BAA4FA"/>'''

def queue_diagram():
    result=[]
    for i,(x,y,w) in enumerate(((824,177,233),(878,263,215),(812,349,239))):
        result.append(f'<rect x="{x}" y="{y}" width="{w}" height="66" rx="16" fill="#15261E" stroke="#F1AD80" stroke-opacity=".45"/><circle cx="{x+28}" cy="{y+33}" r="7" fill="#F1AD80"/><path d="M{x+52} {y+25}H{x+w-25}M{x+52} {y+41}H{x+w-65}" stroke="#8A9E94" stroke-width="3" stroke-linecap="round"/>')
    return '<path d="M830 210H790V296H878M1080 296H1125V382H1050" fill="none" stroke="#F1AD80" stroke-opacity=".6" stroke-width="2"/>'+''.join(result)

cards={
 'vibe-qc-social': ('family','THREE INDEPENDENT PRODUCTS',('Calculate. Explore.','Keep work moving.'),'One scientific workflow. Separate tools.',family_diagram),
 'vibe-qc-product-social': ('vibe-qc','CALCULATE',('Quantum chemistry.','Molecules to solids.'),'Molecular + periodic · Python + C++17',lattice_diagram),
 'vibe-view-social': ('vibe-view','EXPLORE',('See the structure.','Understand the result.'),'Browser · desktop · terminal · headless',viewer_diagram),
 'vibe-queue-social': ('vibe-queue','SCHEDULE',('Queue the work.','Keep the results.'),'Local machines · remote hosts · cluster schedulers',queue_diagram),
}
for filename,(kind,eyebrow,lines,caption,diagram) in cards.items():
    key='vibe-qc' if kind=='family' else kind
    accent=COLORS[key][1]
    logo,width=wordmark(key,True)
    body='''<defs><radialGradient id="halo"><stop stop-color="#174B3E" stop-opacity=".48"/><stop offset="1" stop-color="#08120F" stop-opacity="0"/></radialGradient><linearGradient id="teal" x2="1" y2="1"><stop stop-color="#2DB9A8" stop-opacity=".42"/><stop offset="1" stop-color="#2DB9A8" stop-opacity=".08"/></linearGradient><linearGradient id="coral" x2="1" y2="1"><stop stop-color="#FF7057" stop-opacity=".40"/><stop offset="1" stop-color="#FF7057" stop-opacity=".06"/></linearGradient><linearGradient id="violet" x2="1" y2="1"><stop stop-color="#BAA4FA" stop-opacity=".45"/><stop offset="1" stop-color="#BAA4FA" stop-opacity=".08"/></linearGradient></defs>'''
    body+=f'<rect width="1200" height="630" fill="{PAPER}"/><circle cx="980" cy="290" r="360" fill="url(#halo)"/>'
    body+='<g stroke="#294A3E" stroke-opacity=".22">'+''.join(f'<path d="M{x} 130V455"/>' for x in range(780,1160,40))+''.join(f'<path d="M780 {y}H1160"/>' for y in range(135,456,40))+'</g>'
    body+=f'<g transform="translate(64,48) scale(.9)">{logo}</g>'
    body+=text('vibe-qc.com',976,78,18,MUTED,500)
    body+=text(eyebrow,66,181,14,accent,700,2.3,690)
    body+=text(lines[0],62,268,58,LIGHT,700,-1.8,710)
    body+=text(lines[1],62,342,58,accent,700,-1.8,710)
    body+=text(caption,65,402,20,MUTED,450,0,700)
    body+=diagram()
    body+='<path d="M64 482H1136" stroke="#29473E"/>'
    if kind=='family':
        for (name,label),x,color in zip((('vibe-qc','Calculate'),('vibe-view','Explore'),('vibe-queue','Schedule')),(64,432,800),('#2DB9A8','#BAA4FA','#F1AD80')):
            body+=f'<circle cx="{x+5}" cy="530" r="4" fill="{color}"/>'+text(name,x+24,541,26,LIGHT,650)+text(label,x+24,580,17,MUTED,450)
    else:
        body+=text('Independent by design.',64,548,24,LIGHT,550)
        body+=text('Part of the vibe scientific toolset',64,582,17,MUTED,450)
        body+=text('/products/'+key+'/',845,562,17,accent,500,0,290)
    title='vibe scientific toolset' if kind=='family' else NAMES[key]
    (OUT/f'{filename}.svg').write_text(svg(body,1200,630,title,' '.join(lines)+' '+caption))
print('Generated six wordmarks, three SVG icons and four social cards.')
