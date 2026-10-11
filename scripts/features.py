"""Feature build for DC2026 taxi-out. Uses only columns present in ranking.parquet."""
import glob, os
import numpy as np, pandas as pd, pyarrow.parquet as pq

# DC26_DATA repoints the whole chain at a different challenge extract; DC26_TAG
# keeps that build's /tmp intermediates from colliding with the default one, so
# the old-baseline scripts keep reproducing their shipped submissions.
DATA = os.environ["DC26_DATA"]
TAG = os.environ.get("DC26_TAG", "")

# DC26_DATA must point at the 2026-09-04 reissue (689,534 ranking rows). The pre-reissue extract has the
# same twelve training files but a DEAD 215,876-row scored set, and a script that reads the scored side
# from it gets a silent wrong answer, so ranking_path() makes that case loud.
_PRE = "09-01-2026-09-35-06_files_list"
def ranking_path():
    """The scored-set path, with a loud complaint if it is the dead one."""
    p = os.path.join(DATA, "ranking.parquet")
    n = pq.read_metadata(p).num_rows
    if n != 689534:
        import sys
        print(f"*** WARNING: {p} has {n:,} rows, not the reissue's 689,534. "
              f"{'This is the DEAD pre-reissue scored set. ' if _PRE in DATA else ''}"
              f"Set DC26_DATA to the 2026-09-04 reissue for any scored-set work.",
              file=sys.stderr, flush=True)
    return p

# DC26_TMP relocates every /tmp intermediate (2026-09-23): a sandboxed or clean-room run cannot write /tmp,
# and macOS purges it. The default is unchanged.
TMP = os.environ.get("DC26_TMP", "/tmp")
def wk(stem, ext="parquet"):
    return f"{TMP}/{stem}{TAG}.{ext}"
USE = ['MVT_ID_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt',
       'TAXITIME_SEC_mvt','AIRCRAFT_TYPE_mvt','RUNWAY_mvt','STAND_mvt','LOBT_flt','IOBT_flt',
       'MARKET_SEGMENT_flt','FLIGHT_TYPE_flt','WK_TBL_CAT_flt','AIRCRAFT_OPERATOR_flt',
       'EOBT_1_flt','ARVT_1_flt','AOBT_3_flt','ARVT_3_flt','CALLSIGN_flt']

def load_training():
    parts = [pq.read_table(f, columns=USE).to_pandas() for f in sorted(glob.glob(f"{DATA}/training_*.parquet"))]
    d = pd.concat(parts, ignore_index=True)
    return d.drop_duplicates(subset='MVT_ID_mvt')

def secs(a, b):
    return (a - b).dt.total_seconds()

def build(df):
    """df must contain both DEP and ARR rows for the airports/period, so arrival
    pressure can be computed. Returns departures only, with features."""
    df = df.copy()
    df['apt'] = df.ADEP_mvt.where(df.PHASE_mvt == 'DEP', df.ADES_mvt)
    dep = df[df.PHASE_mvt == 'DEP'].copy()
    arr = df[df.PHASE_mvt == 'ARR']

    # the strong baseline, as a feature
    dep['naive'] = secs(dep.MVT_TIME_UTC_mvt, dep.AOBT_3_flt)
    dep.loc[(dep.naive < 0) | (dep.naive > 7200), 'naive'] = np.nan

    # planning / delay signals (all from unblanked columns)
    dep['aobt_minus_eobt'] = secs(dep.AOBT_3_flt, dep.EOBT_1_flt)
    dep['aobt_minus_sched'] = secs(dep.AOBT_3_flt, dep.SCHED_TIME_UTC_mvt)
    dep['lobt_minus_iobt'] = secs(dep.LOBT_flt, dep.IOBT_flt)
    dep['filed_block'] = secs(dep.ARVT_1_flt, dep.EOBT_1_flt)
    dep['flown_block'] = secs(dep.ARVT_3_flt, dep.AOBT_3_flt)

    # calendar
    t = dep.MVT_TIME_UTC_mvt
    dep['hour'] = t.dt.hour + t.dt.minute / 60
    dep['dow'] = t.dt.dayofweek
    dep['month'] = t.dt.month
    dep['doy'] = t.dt.dayofyear
    dep['hr_sin'] = np.sin(2*np.pi*dep.hour/24); dep['hr_cos'] = np.cos(2*np.pi*dep.hour/24)
    dep['day'] = t.dt.floor('D')

    # ---- traffic state, computed per airport on epoch seconds ----
    # epoch SECONDS. pandas gives datetime64[us] here, so convert explicitly
    # rather than dividing raw int64 by a hard-coded power of ten.
    def epoch_s(s):
        v = s.values.astype('datetime64[s]').astype('float64')
        v[pd.isna(s.values)] = np.nan
        return v
    dep['t_off'] = epoch_s(dep.MVT_TIME_UTC_mvt)
    dep['t_obt'] = epoch_s(dep.AOBT_3_flt)
    arr_t = {k: np.sort(epoch_s(g.MVT_TIME_UTC_mvt)) for k, g in arr.groupby('apt')}

    out = []
    for apt, g in dep.groupby('apt', sort=False):
        g = g.sort_values('t_off').copy()
        # surface count: both arrays MUST describe the same set of flights, else
        # the counts do not cancel and N drifts with dataframe size.
        both = (~np.isnan(g.t_obt.values)) & (~np.isnan(g.t_off.values))
        off = np.sort(g.t_obt.values[both])
        tko = np.sort(g.t_off.values[both])
        g['N_push'] = np.searchsorted(off, g.t_obt.values, 'right') - np.searchsorted(tko, g.t_obt.values, 'right')
        g['N_off']  = np.searchsorted(off, g.t_off.values, 'right') - np.searchsorted(tko, g.t_off.values, 'right')
        # airport-wide departure throughput in windows before own take-off
        for w in (900, 1800, 3600):
            g[f'dep_{w}'] = np.searchsorted(tko, g.t_off.values, 'right') - np.searchsorted(tko, g.t_off.values - w, 'left')
        # arrivals landed in the window before own take-off
        at = arr_t.get(apt, np.array([], dtype=np.int64))
        for w in (900, 1800):
            g[f'arr_{w}'] = np.searchsorted(at, g.t_off.values, 'right') - np.searchsorted(at, g.t_off.values - w, 'left')
        # same-runway pressure and gap to previous departure on that runway
        g['rwy_900'] = 0.0; g['rwy_gap'] = np.nan
        for rw, gg in g.groupby('RUNWAY_mvt'):
            s = np.sort(gg.t_off.values); i = gg.index
            g.loc[i, 'rwy_900'] = np.searchsorted(s, gg.t_off.values, 'right') - np.searchsorted(s, gg.t_off.values - 900, 'left')
            pos = np.searchsorted(s, gg.t_off.values, 'left')
            gap = np.where(pos > 0, gg.t_off.values - s[np.maximum(pos-1, 0)], np.nan)
            g.loc[i, 'rwy_gap'] = np.where(gap > 86400, np.nan, gap)   # cap month-boundary gaps
        out.append(g)
    dep = pd.concat(out, ignore_index=True)

    # ---- day-level and hour-level naive aggregates ----
    # legitimate at inference: the ranking set contains whole months, so all
    # AOBT_3 and take-off times for a given airport-day are available.
    # This is the de-icing / disrupted-day proxy.
    dep['apt_day_naive'] = dep.groupby(['apt','day']).naive.transform('median')
    dep['apt_day_naive_p90'] = dep.groupby(['apt','day']).naive.transform(lambda s: s.quantile(.90))
    dep['apt_hr_naive'] = dep.groupby(['apt','day',dep.MVT_TIME_UTC_mvt.dt.hour]).naive.transform('median')
    dep['naive_vs_day'] = dep.naive - dep.apt_day_naive

    dep['dest_region'] = dep.ADES_mvt.str[:2]
    return dep
