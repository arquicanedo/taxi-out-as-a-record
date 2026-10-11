"""Content hash of an ADS-B feature table: the rows sorted by movement id and the archived columns sorted by name,
hashed by value, so the hash does not depend on parquet encoding or writer metadata. A regenerated table may carry
columns added to the feature builder later; only the archived columns are hashed, as regen_adsb.sh compares them. The public release does not redistribute the
derived adsb.lol tables (ODbL: it ships the method instead); scripts/regen_adsb.sh regenerates them from the archive and
checks them against artifacts/adsb_push_feat.content.sha256, which this script writes from the archived tables.
  python scripts/adsb_feat_hash.py --write                   write artifacts/adsb_push_feat.content.sha256
  python scripts/adsb_feat_hash.py --check KEY TABLE.parquet exit 0 if TABLE matches the recorded hash for KEY
  python scripts/adsb_feat_hash.py --place KEY TABLE.parquet OUT.parquet
                                                             check, then write TABLE restricted to the archived columns,
                                                             in their order, to OUT. The stage harness uses whatever
                                                             detection columns a table carries, so a rebuilt table must
                                                             be placed with exactly the archived columns.
"""
import hashlib, os, sys
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = {v: f'artifacts/adsb_push_feat_{"full" if v == "full" else "v" + v}.parquet' for v in ('102', '103', '104', '105', 'full')}

def content_hash(path, cols=None):
    d = pd.read_parquet(path).set_index('MVT_ID_mvt').sort_index(); d = d[sorted(cols or d.columns)]
    h = hashlib.sha256(pd.util.hash_pandas_object(d, index=True).values.tobytes())
    h.update(','.join(f'{c}:{d[c].dtype}' for c in d.columns).encode()); return h.hexdigest()

if __name__ == '__main__':
    REC = os.path.join(REPO, 'artifacts/adsb_push_feat.content.sha256')
    if sys.argv[1:] == ['--write']:
        lines = []
        for v, p in TABLES.items():
            cols = [c for c in pd.read_parquet(os.path.join(REPO, p)).columns if c != 'MVT_ID_mvt']
            lines.append(f'{v} {content_hash(os.path.join(REPO, p))} {",".join(cols)}')
        open(REC, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(l[:90] for l in lines))
    elif sys.argv[1] in ('--check', '--place'):
        key, path = sys.argv[2], sys.argv[3]
        rec = {l.split()[0]: l.split()[1:] for l in open(REC).read().splitlines()}; h, cols = rec[key][0], rec[key][1].split(',')
        got = content_hash(path, cols); print(f'  {key}: {"MATCHES" if got == h else "DIFFERS FROM"} the recorded content hash')
        if got != h: sys.exit(1)
        if sys.argv[1] == '--place':
            pd.read_parquet(path)[['MVT_ID_mvt'] + cols].to_parquet(sys.argv[4], index=False)
            assert content_hash(sys.argv[4]) == h; print(f'  {key}: placed {sys.argv[4]} with the {len(cols)} archived columns')
