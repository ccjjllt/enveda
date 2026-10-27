import os
h = os.path.dirname(os.path.abspath(__file__))
f, m, b = (open(f'{h}/{n}').read() for n in ['feats.py', 'model.py', 'train_hn.py'])
emit = ("import os as _os\nif _os.path.isdir('/kaggle/working'):\n    for _n, _s in {'feats.py': %r, 'model.py': %r}.items():\n"
        "        open('/kaggle/working/' + _n, 'w').write(_s)\n") % (f, m)
os.makedirs(f'{h}/../../notebooks/044', exist_ok=True)
pre = ("import subprocess as _sp, sys as _sys\n"
       "try:\n    import rdkit\nexcept ImportError:\n    _sp.run([_sys.executable, '-m', 'pip', 'install', '-q', 'rdkit==2026.3.3'], check=True)\n"
       "import rdkit; print('rdkit', rdkit.__version__, flush=True)\n")
open(f'{h}/../../notebooks/044/own_spec_hn.py', 'w').write(pre + f + '\n' + m + '\n' + emit + '\n' + b)
