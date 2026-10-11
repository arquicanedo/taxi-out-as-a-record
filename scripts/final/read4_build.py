"""Read 4: the Istanbul candidate's January and July rows as a leaderboard file (quick-boat_v131.parquet).

  env -u UV_EXCLUDE_NEWER uv run --quiet --with pandas --with numpy --with pyarrow==26.0.0 python scripts/final/read4_build.py \
      --candidate <the output of scripts/final/ltfm_local.py> --old-data <2026-09-04 extract> --out quick-boat_v131.parquet

The uploaded file was written with pyarrow 26.0.0 (built without run_final.sh's version pin). With the pinned 25.0.1 the
values are identical and the bytes differ (parquet writer metadata), so the script checks the values against the
shipped file always, and its sha256 when pyarrow is 26.0.0.

The leaderboard scores January and July against the 2026-09-04 extract's labels, so the file takes the candidate's
predictions for those 344,841 departures in that extract's submitting order and schema, as read3_build.py did for v130.
It differs from v130 on Istanbul (LTFM) rows only, which is asserted.

Pre-registered before the upload: 233.2730 [233.5101, 233.0357] from the window replay, 233.7464 (v130) meaning no
effect. Scored 2026-10-11 13:47:52Z: 234.2287, worse than v130 by 0.4823 s. The candidate was not adopted.
"""
import argparse, hashlib, os
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
for k in ('--candidate', '--old-data', '--out'): ap.add_argument(k, required=True)
a = ap.parse_args()
SUB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'submissions')
SHA131 = '7b1eb53dc76513140bac3edfd336c32ba8a42ebbcef312fb246ba9c5b6792c11'

tpl = pq.read_table(f'{a.old_data}/submitting.parquet'); ids = tpl.to_pandas().MVT_ID_mvt.values; assert len(ids) == 344841
C = pq.read_table(a.candidate).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(ids).to_numpy(float)
assert np.isfinite(C).all(), 'the candidate must cover every January and July departure'
V = pq.read_table(f'{SUB}/quick-boat_v130.parquet').to_pandas()
assert (V.MVT_ID_mvt.values == ids).all(); v = V.TAXITIME_SEC_mvt.to_numpy(float)
R = pq.read_table(f'{a.old_data}/ranking.parquet', columns=['MVT_ID_mvt', 'PHASE_mvt', 'ADEP_mvt']).to_pandas()
apt = R[R.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt').ADEP_mvt.reindex(ids).values
d = C - v; ch = d != 0
assert (apt[ch] == 'LTFM').all(), 'a non-Istanbul row differs from v130'
print(f'against v130: {int(ch.sum()):,} rows differ, all LTFM (of {int((apt == "LTFM").sum()):,}); Q {float((d ** 2).sum()):.4e}; '
      f'more than 60 s {int((np.abs(d) > 60).sum()):,}')
t = pa.Table.from_pandas(pd.DataFrame({'MVT_ID_mvt': ids, 'TAXITIME_SEC_mvt': C.astype(tpl.schema.field('TAXITIME_SEC_mvt').type.to_pandas_dtype())}),
                         schema=tpl.schema, preserve_index=False)
pq.write_table(t, a.out)
chk = pq.read_table(a.out); assert chk.schema.equals(tpl.schema) and chk.num_rows == len(ids)
sha = hashlib.sha256(open(a.out, 'rb').read()).hexdigest()
print(f'wrote {a.out}  sha256 {sha}')
assert chk.equals(pq.read_table(f'{SUB}/quick-boat_v131.parquet')), 'values differ from the shipped quick-boat_v131.parquet'
print('values identical to the shipped quick-boat_v131.parquet')
if pa.__version__ == '26.0.0':
    assert sha == SHA131, f'not the shipped quick-boat_v131.parquet ({SHA131})'
    print('bytes identical to the shipped file')
else:
    print(f'pyarrow {pa.__version__}: bytes differ from the shipped file by writer metadata only (it was written by 26.0.0)')
