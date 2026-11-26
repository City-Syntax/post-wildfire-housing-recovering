"""Run supplementary item FigureS6 and its related numerical cohort."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    sys.exit(subprocess.call([sys.executable,str(root/'run.py'),'supplementary','--item','FigureS6','--figures'],cwd=root))
