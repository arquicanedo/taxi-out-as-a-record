"""One colour system for the seven figures of the paper. Every fig_*.py script imports its colours from here, by role,
never as hex.

Two hues carry the thesis, placed apart in lightness as well as hue so they survive greyscale printing:
the RECORD (what an airport's system wrote down) is a vivid red-orange at L* 47, and the MEASUREMENT
(what actually happened) is a deep navy at L* 33, the ink-like colour the journal uses for its own headings.
Both pass 4.5:1 on white as text. Everything else is a lighter step of one of those two hues, or a grey.

Measured (colorspacious, Machado 2009 at severity 100, CAM02-UCS distance):
    RECORD / MEASURED   dE 61.7 normal, 53.4 deutan, 43.9 protan, 62.0 tritan,  L* gap 14.1 (greys #707070 / #4E4E4E)
    RECORD text 4.98:1, MEASURED text 8.37:1 on white
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK = '#000000'            # axis text, tick labels, titles, ink marks (take-off, the identity line); never changes
# ---- the record
RECORD = '#C94001'         # the record and its anomalies: schedule copies, extremes, the recorded off-block, rollovers, bound violations
RECORD_MID = '#E1794F'     # a lighter step of the record hue, for a second record-side category (the join-rules step of the waterfall)
RECORD_TINT = '#EFC6B6'    # light fill of the record hue: error bars of the record class, blanked rows predicted as copies
# ---- the measurement
MEASURED = '#085182'       # the measurement and the ordinary: ADS-B pushback, true seconds, ordinary rows, the value needed
MEASURED_MID = '#648EBF'   # a lighter step of the measurement hue: the bulk prediction's dot, tree members, the more-seeds step
MEASURED_TINT = '#C2D0E5'  # faint fill of the measurement hue: the ordinary population (histogram, plain rows, row counts)
# ---- neutrals (context only; grey never carries meaning)
REF = '#8c8c8c'            # reference lines (chance, "if every row cost the same") and neutral marks (the other extremes, SCHED)
REF_LIGHT = '#b4b4b4'      # error bars of the neutral class, the schematic day axes in Figure 3
CONTEXT = '#cfcfcf'        # connectors and dumbbell bars, identity diagonals, month gridlines
GROUND = '#9a9a94'         # Figure 4: every adsb.lol ground sample (drawn at alpha 0.35)
BAND = ('#f3f3ef', '#f9f9f6')   # Figure 4: alternating band backgrounds, and the text halo
SHADE = '#e6e6e2'          # Figure 4: the scored-month span
# ---- categorical: the nine members of the blend (Figure 4), three families as three steps of the measurement hue
MEMBER_NET = MEASURED      # neighbour nets
MEMBER_TREE = MEASURED_MID # pooled and per-airport trees
MEMBER_LINEAR = '#97B3D8'  # the third family (independent build, identity categoricals, runway neighbours): L* 72
# ---- the score waterfall (Figure 4): base, trajectory stage, more seeds, feed rules, join rules
STEP = [INK, MEASURED, MEASURED_MID, RECORD, RECORD_MID]   # measurement-side gains in blue, record-side gains in red-orange
# ---- colormaps
SEQ = LinearSegmentedColormap.from_list('measured_seq', ['#B7C8E0', '#648EBF', '#085182', '#0F2941'])   # time along an ADS-B track, early (light) to late (dark)
DENSITY = LinearSegmentedColormap.from_list('measured_density', ['#FFFFFF', '#C2D0E5', '#648EBF', '#085182', '#04203A'])   # hexbin density of ordinary rows, white to deep navy
