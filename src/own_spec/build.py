import os
h = os.path.dirname(os.path.abspath(__file__))
f, m, b = (open(f'{h}/{n}').read() for n in ['feats.py', 'model.py', 'train_body.py'])
emit = ("import os as _os\nif _os.path.isdir('/kaggle/working'):\n    for _n, _s in {'feats.py': %r, 'model.py': %r}.items():\n"
        "        open('/kaggle/working/' + _n, 'w').write(_s)\n") % (f, m)
os.makedirs(f'{h}/../../notebooks/039', exist_ok=True)
open(f'{h}/../../notebooks/039/own_spec_train.py', 'w').write(f + '\n' + m + '\n' + emit + '\n' + b)
