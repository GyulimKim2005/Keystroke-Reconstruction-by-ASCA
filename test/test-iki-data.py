import numpy as np
from pathlib import Path

npz_dir = Path('../npz_sequences')
for p in sorted(npz_dir.glob('*.npz')):
    data = np.load(p)
    x = data['x']
    iki = data['iki']
    print(p.name, 'x', x.shape, 'iki', iki.shape)
    print('iki:', iki)
    print('N:', x.shape[0], 'IKI len:', len(iki))