"""Configure local segmentation illustrations for the photographic Figure 3 panel."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('image_assets',type=Path);args=parser.parse_args()
    assets=args.image_assets.resolve()
    if not (assets/'data/seg_visuals').is_dir():parser.error('Expected data/seg_visuals under the local asset directory.')
    (ROOT/'local_assets.json').write_text(json.dumps({'image_asset_root':str(assets)},indent=2),encoding='utf8')
    print('Local illustration assets configured.')
if __name__=='__main__':main()
