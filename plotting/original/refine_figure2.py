from pathlib import Path
from copy import deepcopy
from lxml import etree as E
import json, hashlib
from citation_tools import B
WORK = Path(__file__).resolve().parent
SVG = {'s': 'http://www.w3.org/2000/svg'}
source = B / '12_Figure2_完整图.svg'
tree = E.parse(str(source))
root = tree.getroot()
audit = []
for aid, letter in [('axes_1', 'a'), ('axes_2', 'b'), ('axes_4', 'c'), ('axes_5', 'd')]:
    ax = root.xpath(f'.//s:g[@id="{aid}"]', namespaces=SVG)[0]
    t = ax.xpath(f'.//s:text[text()="{letter}"]', namespaces=SVG)[0]
    title = [x for x in ax.xpath('./s:g/s:text', namespaces=SVG) if 'font-size: 9px' in x.get('style', '')][0]
    x = float(title.get('x')) - 22.559097
    y = float(title.get('y'))
    audit.append({'panel': letter, 'old': dict(t.attrib), 'new_x': x, 'new_y': y})
    t.set('x', f'{x:.6f}')
    t.set('y', f'{y:.6f}')
    t.set('transform', f'rotate(-0 {x:.6f} {y:.6f})')
    t.set('style', t.get('style').replace('font-size: 11px', 'font-size: 10.5px'))
for aid in ['axes_4', 'axes_5']:
    ax = root.xpath(f'.//s:g[@id="{aid}"]', namespaces=SVG)[0]
    t = ax.xpath('.//s:text[text()="Rebuilt buildings classified as upgraded (%)"]', namespaces=SVG)[0]
    y = float(t.get('y'))
    x = f"{float(t.get('x')) - 2.5:.6f}"
    t.text = None
    t.set('x', x)
    t.set('transform', f'rotate(-90 {x} {y:.6f})')
    for dy, label in [(-4.5, 'Rebuilt buildings classified'), (4.5, 'as upgraded (%)')]:
        span = E.SubElement(t, '{' + SVG['s'] + '}tspan', x=x, y=f'{y + dy:.6f}')
        span.text = label
    audit.append({'panel': aid, 'vertical_axis_caption': 'same wording wrapped into two lines'})
legend = root.xpath('.//s:g[@id="legend_1"]', namespaces=SVG)[0]
legend.set('transform', 'translate(0 -4)')
audit.append({'legend': 'panel a', 'vertical_shift_pt': -4})
out = B / '23_Figure2_分图位置精调_20261122.svg'
tree.write(str(out), xml_declaration=True, encoding='utf-8')
print(out)
