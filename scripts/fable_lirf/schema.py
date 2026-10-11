import os
import pandas as pd, numpy as np, pyarrow.parquet as pq
D=(os.environ['DC26_DATA'] + '/')
for f in ['training_2025-01-01_2025-02-01.parquet','ranking.parquet','submitting.parquet']:
    s=pq.read_schema(D+f)
    print(f, pq.read_metadata(D+f).num_rows)
    for n,t in zip(s.names,s.types): print('   ',n,t)
z=np.load(os.path.join(os.environ['DC26_WORK'], 'adsblol', 'cand', 'v121_window.npz'),allow_pickle=True)
print({k:(z[k].shape,z[k].dtype) for k in z.files})
print(z['win_ids'][:5], z['newW'][:5])
sub=pd.read_parquet(os.environ.get('FABLE_V121', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions', 'quick-boat_v121.parquet')))
print(sub.shape, sub.dtypes, sub.head())
