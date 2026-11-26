"""Run 05 Result4 Figure4."""
from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    sys.exit(subprocess.call([sys.executable,str(root/'run.py'),'4','--figures'],cwd=root))
