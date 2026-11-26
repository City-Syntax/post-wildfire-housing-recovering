"""Retain the author's Figure 2 text layout around recomputed chart geometry."""
from pathlib import Path
import re, json
from lxml import etree as E

NS={'s':'http://www.w3.org/2000/svg'}
SVG='{'+NS['s']+'}'

def align_vector_text(computed: Path, template: Path, audit: Path):
    drawing=E.parse(str(computed));reference=E.parse(str(template))
    current_text=drawing.xpath('//s:text',namespaces=NS)
    source_text=reference.xpath('//s:text',namespaces=NS)
    assert len(current_text)==len(source_text)
    assert all(''.join(a.itertext())==''.join(b.itertext()) for a,b in zip(current_text,source_text)), 'Computed labels differ from the authored layout'
    def anchor(tree):
        path=tree.xpath('//s:g[@id="axes_1"]/s:g[starts-with(@id,"patch_")]/s:path',namespaces=NS)[0]
        return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?',path.get('d'))[:2]]
    old=anchor(reference);new=anchor(drawing);dx,dy=old[0]-new[0],old[1]-new[1]
    figure=drawing.xpath('//s:g[@id="figure_1"]',namespaces=NS)[0]
    assert not figure.get('transform')
    figure.set('transform',f'translate({dx:.9f} {dy:.9f})')
    for current,source in zip(current_text,source_text):
        current.attrib.clear();current.attrib.update(source.attrib)
        a=current.findall(SVG+'tspan');b=source.findall(SVG+'tspan')
        assert len(a)==len(b)
        for aa,bb in zip(a,b):aa.attrib.clear();aa.attrib.update(bb.attrib)
        parent=current.getparent();index=parent.index(current)
        parent.remove(current);wrapper=E.Element(SVG+'g',transform=f'translate({-dx:.9f} {-dy:.9f})');wrapper.append(current);parent.insert(index,wrapper)
    for key in ['width','height','viewBox']:drawing.getroot().set(key,reference.getroot().get(key))
    drawing.write(str(computed),encoding='utf8',xml_declaration=True)
    import os
    destination=os.environ.get('RECOVERY_AUDIT_DIR')
    if destination:
        target=Path(destination);target.mkdir(parents=True,exist_ok=True)
        (target/audit.name).write_text(json.dumps({'computed_text_values_checked':len(current_text),'original_statistical_paths_copied':False,'geometry_translation_pt':[dx,dy]},indent=2),encoding='utf8')

align_figure2=align_vector_text

def save_reference_canvas(fig,path,dimensions,dpi=300,**kwargs):
    """Use original export bounds, retaining recomputed paths and whitespace."""
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    bounds=fig.get_tightbbox(fig.canvas.get_renderer())
    width,height=[(value+.01)/dpi for value in dimensions]
    padx=(width-bounds.width)/2;pady=(height-bounds.height)/2
    assert padx>=0 and pady>=0, 'Original canvas would clip the computed labels'
    box=Bbox.from_bounds(bounds.x0-padx,bounds.y0-pady,width,height)
    kwargs.update({'dpi':dpi,'bbox_inches':box,'pad_inches':0})
    fig.savefig(path,**kwargs)

def publish(staging: Path, output: Path, stems):
    import shutil
    output.mkdir(exist_ok=True,parents=True)
    for stem in stems:
        for ext in ['png','pdf','svg']:
            path=staging/(stem+'.'+ext)
            if path.is_file():shutil.copy2(path,output/path.name)
