"""Run bounded, deterministic teaching tests; optional PyTorch tests are explicit."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--torch',action='store_true',help='also run optional neural implementation tests')
    args=parser.parse_args()
    files=sorted((ROOT/'examples').glob('*_lab.py'))
    files += [ROOT/'examples/mechanisms.py', ROOT/'examples/formula_checks.py']
    if args.torch:
        files.append(ROOT/'examples/deep_textbook_train.py')
    for file in files:
        # The original knowledge demo predates the embedded unittest interface.
        args_for_file = {'formula_checks.py': [], 'knowledge_lab.py': ['all', '--steps', '2000']}.get(file.name, ['test'])
        command=[sys.executable,str(file)]+args_for_file
        print('Testing '+file.name,flush=True)
        subprocess.run(command,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},check=True,timeout=180)
    print(f'Completed {len(files)} teaching test programs. This is not a benchmark efficacy result.')

if __name__=='__main__':
    main()
