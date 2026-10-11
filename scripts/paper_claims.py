"""Claim ledger and checker for the paper (analysis/joas/main.tex).

A revision must not invent a number, and must not lose one silently. This makes both mechanical. The ledger holds every
number of the paper's first version (v1, git tag `paper-v1` of the private research repository; the `ledger` and
`paragraphs` commands need it); the paper being checked is v2.

  python3 scripts/paper_claims.py ledger          v1 -> analysis/paper-v2/ledger.tsv (every number, with its sentence)
  python3 scripts/paper_claims.py paragraphs      v1 -> analysis/paper-v2/paragraphs.tsv (block inventory for the mapping)
  python3 scripts/paper_claims.py check [FILE]   default analysis/joas/main.tex; exit 1 on either failure:
      NEW    a number in V2 that is not in the ledger (or a rounding of a ledger number) and not in added.tsv
      LOST   a ledger number absent from V2 (exactly or rounded) and not in dropped.tsv
    ROUNDED matches are reported for review, not failed: rounding is not invention, but it can hide a coincidence.

What counts as a number: anything with a decimal point, an exponent, a thousands separator or a percent sign, any
integer of 10 or more, dates and clock times. Section references ("Section 4.3"), citation brackets and heading
numbers are not claims and are removed first. Values are compared by value (1,037,037 == 1037037; 6.969e+09 ==
6.969e9; 0.40 == 0.4), sign included.

added.tsv   value<TAB>source      a number V2 may use that v1 does not have; source says which script or tool made it
dropped.tsv value<TAB>reason      a v1 number V2 deliberately leaves out
"""
import os, re, subprocess, sys
from decimal import Decimal, InvalidOperation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR = os.path.join(REPO, 'analysis', 'paper-v2')
V1_REF = 'paper-v1:analysis/PAPER.md'

STRIP = [re.compile(p) for p in (
    r'<!--.*?-->',                                                     # edit-pass notes, not rendered
    r'Sections? \d+(?:\.\d+)?(?:(?:,| and| to| or) \d+(?:\.\d+)?)*',   # cross-references
    r'(?:Tables?|Figures?) \d+',                                        # table and figure numbers
    r'\[\d+(?:,\s*\d+)*\]',                                           # citations
    r'^#+\s*\d+(?:\.\d+)*\.?',                                         # heading numbers
    r'^\d+\.\s',                                                       # numbered-list markers (references)
    r'figures/[\w\-]+\.\w+',                                           # figure paths
)]
TOKEN = re.compile(
    r'(?P<date>\b\d{4}-\d{2}-\d{2}\b)'
    r'|(?P<time>\b\d{1,2}:\d{2}(?::\d{2})?\b)'
    r'|(?P<num>(?:(?<=[\s(=~<>:/])|^)[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?'
    r'|(?<![\w.\-+])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?)')

def canon(raw):
    """Canonical value string, or None for a number too small to be a claim (a bare integer under 10)."""
    s = raw.replace(',', '').rstrip('%').lstrip('+')
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    plain = re.fullmatch(r'-?\d+', s) is not None
    if plain and abs(d) < 10:
        return None
    return format(d.normalize(), 'f') if 'e' not in s.lower() else format(d.normalize(), 'E').replace('E+', 'e').replace('E', 'e')

def numbers(text):
    """(canonical, raw, line_no, line) for every claim-number in a markdown text, code blocks included."""
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        work = line
        for p in STRIP:
            work = p.sub(' ', work)
        for m in TOKEN.finditer(work):
            if m.group('date') or m.group('time'):
                out.append((m.group(0), m.group(0), i, line)); continue
            c = canon(m.group(0))
            if c is not None:
                out.append((c, m.group(0), i, line))
    return out

def sections(text):
    """Section label for every line: Abstract, Intro, 1.3, 4.1, Conclusion, ..."""
    lab, labs = 'Title', []
    for line in text.splitlines():
        m = re.match(r'^(#+)\s*(.*)', line)
        if m:
            h = m.group(2).strip(); n = re.match(r'(\d+(?:\.\d+)*)', h)
            lab = n.group(1) if n else h.split()[0]
        elif line.startswith('**Abstract'):
            lab = 'Abstract'
        labs.append(lab)
    return labs

def v1_text():
    return subprocess.run(['git', '-C', REPO, 'show', V1_REF], capture_output=True, text=True, check=True).stdout

def read_tsv(name):
    p = os.path.join(DIR, name)
    if not os.path.exists(p):
        return {}
    rows = [l.split('\t') for l in open(p).read().splitlines() if l and not l.startswith('#')]
    return {r[0]: (r[1] if len(r) > 1 else '') for r in rows}

def rounded_match(v, pool):
    """Ledger values that round to v at v's own precision (v must carry a decimal point or an exponent)."""
    if re.fullmatch(r'-?\d+', v) or ':' in v or v.count('-') > 1:
        return []
    dv = Decimal(v); q = Decimal(1).scaleb(dv.as_tuple().exponent)
    hits = []
    for w in pool:
        if ':' in w or w.count('-') > 1:
            continue
        try:
            dw = Decimal(w)
        except InvalidOperation:
            continue
        if dw != dv and dw.quantize(q) == dv and len(w.replace('-', '').replace('.', '')) > len(v.replace('-', '').replace('.', '')):
            hits.append(w)
    return hits

def cmd_ledger():
    t = v1_text(); labs = sections(t); os.makedirs(DIR, exist_ok=True)
    rows = numbers(t)
    with open(os.path.join(DIR, 'ledger.tsv'), 'w') as f:
        f.write('# every number in the paper at tag paper-v1: value, as written, section, line, the line itself\n')
        for c, raw, i, line in rows:
            f.write(f'{c}\t{raw}\t{labs[i-1]}\t{i}\t{" ".join(line.split())[:220]}\n')
    print(f'ledger: {len(rows)} numbers, {len({r[0] for r in rows})} distinct values -> analysis/paper-v2/ledger.tsv')

def cmd_paragraphs():
    t = v1_text(); labs = sections(t); lines = t.splitlines(); os.makedirs(DIR, exist_ok=True)
    blocks, cur, start, fence = [], [], None, False
    for i, line in enumerate(lines, 1):
        if line.startswith('```'):
            if not fence:
                if cur: blocks.append((start, cur)); cur = []
                fence, start, cur = True, i, [line]; continue
            cur.append(line); blocks.append((start, cur)); cur, fence = [], False; continue
        if fence:
            cur.append(line); continue
        if not line.strip():
            if cur: blocks.append((start, cur)); cur = []
            continue
        if not cur: start = i
        cur.append(line)
    if cur: blocks.append((start, cur))
    n_by = {}
    with open(os.path.join(DIR, 'paragraphs.tsv'), 'w') as f:
        f.write('# id\tkind\tv1 lines\tnumbers\tfirst words\n')
        for s, b in blocks:
            sec = labs[s - 1]; n_by[sec] = n_by.get(sec, 0) + 1; pid = f'{sec}.p{n_by[sec]}'
            kind = 'code' if b[0].startswith('```') else 'heading' if b[0].startswith('#') else 'figure' if b[0].startswith('![') else 'text'
            nn = len(numbers('\n'.join(b)))
            f.write(f'{pid}\t{kind}\t{s}-{s + len(b) - 1}\t{nn}\t{" ".join(" ".join(b).split())[:90]}\n')
    print(f'paragraphs: {len(blocks)} blocks -> analysis/paper-v2/paragraphs.tsv')

def cmd_check(path):
    led = [l.split('\t') for l in open(os.path.join(DIR, 'ledger.tsv')).read().splitlines() if not l.startswith('#')]
    pool = {r[0] for r in led}; ctx = {r[0]: f'v1 {r[2]}: {r[4][:110]}' for r in led}
    added, dropped = read_tsv('added.tsv'), read_tsv('dropped.tsv')
    t = open(path).read()
    if path.endswith('.tex'):                              # the JOAS source, from 2026-10-05; its references are in the .bib
        t = tex_to_text(t)
        bib = os.path.join(os.path.dirname(path), 'reference.bib')
        if os.path.exists(bib): t += '\n## References\n' + re.sub(r'[{}]', ' ', open(bib).read())
    labs = sections(t); v2 = numbers(t); v2vals = {c for c, *_ in v2}
    new, rounded = [], []
    for c, raw, i, line in v2:
        if c in pool or c in added:
            continue
        hits = rounded_match(c, pool)
        (rounded if hits else new).append((c, raw, labs[i - 1], i, line, hits))
    lost = []
    v2_only = [w for w in v2vals if w not in pool and w not in added]   # only a number v1 does not have can stand in
    for v in sorted(pool):                                             # for a rounded v1 number (0.93 in v1 already
        if v in v2vals or v in dropped:                                # says nothing about 0.9253)
            continue
        if any(v in rounded_match(w, [v]) for w in v2_only):
            continue
        lost.append(v)
    for c, raw, sec, i, line, _ in new:
        print(f'NEW      {raw:>14}  v2 {sec} line {i}: {" ".join(line.split())[:120]}')
    for v in lost:
        print(f'LOST     {v:>14}  {ctx[v]}')
    for c, raw, sec, i, line, hits in rounded:
        print(f'ROUNDED  {raw:>14}  from {", ".join(hits[:3])}  v2 {sec} line {i}')
    print(f'\n{len(v2)} numbers in {os.path.basename(path)}: {len(new)} NEW, {len(lost)} LOST, {len(rounded)} rounded '
          f'(ledger {len(pool)} values, added {len(added)}, dropped {len(dropped)})')
    sys.exit(1 if new or lost else 0)

def _drop_cmd(t, name, nargs):
    """Remove every \name[opt]{arg}...{arg} with balanced braces (nargs braced arguments)."""
    out, i = [], 0
    pat = re.compile(r'\\' + name + r'\b\*?')
    while True:
        m = pat.search(t, i)
        if not m: out.append(t[i:]); break
        out.append(t[i:m.start()]); j = m.end()
        if j < len(t) and t[j] == '[':
            j = t.index(']', j) + 1
        for _ in range(nargs):
            while j < len(t) and t[j] in ' \n': j += 1
            if j >= len(t) or t[j] != '{': break
            depth = 0
            while True:
                depth += {'{': 1, '}': -1}.get(t[j], 0); j += 1
                if depth == 0: break
        i = j
    return ''.join(out)

def tex_to_text(t):
    """The prose of a LaTeX paper, for the number check: the document body only, no labels, references, citations, column
    specs or layout settings; headings rewritten as Markdown so findings name their section."""
    t = t[t.index('\\begin{document}') + 16:t.index('\\end{document}')]
    t = re.sub(r'(?<!\\)%.*', '', t)
    t = re.sub(r'\\(sub)?section\*?\{(.*?)\}\\label\{sec:([^}]*)\}',
               lambda m: '\n' + ('### ' if m.group(1) else '## ') + m.group(3) + ('. ' if not m.group(1) else ' ') + m.group(2) + '\n', t)
    for name, n in (('label', 1), ('ref', 1), ('cite', 1), ('setlength', 2), ('renewcommand', 2), ('includegraphics', 1),
                    ('begin', 2 if False else 1), ('end', 1), ('phantom', 1)):
        t = _drop_cmd(t, name, n)
    t = re.sub(r'@\{\}[rlc]@\{\}', ' ', t)                       # nested header tabular specs
    t = re.sub(r'\\hsize=[0-9.]+\\hsize', ' ', t)                 # tabularx column weights
    t = re.sub(r'>\{\\raggedright\\arraybackslash\}p\{[^}]*\}|\{(?:[lrc]|p\{[^}]*\})+\}', ' ', t)   # column specs
    t = t.replace('{,}', ',').replace('\\%', '%').replace('\\_', '_').replace('{[}', '[').replace('{]}', ']')
    t = t.replace('\\\\', '\n').replace('&', ' | ').replace('~', ' ').replace('\\ ', ' ')
    t = re.sub(r'\\[a-zA-Z]+\*?', ' ', t)
    t = re.sub(r'[{}$^]', ' ', t)
    return t

if __name__ == '__main__':
    a = sys.argv[1:] or ['check']
    if a[0] == 'ledger': cmd_ledger()
    elif a[0] == 'paragraphs': cmd_paragraphs()
    elif a[0] == 'check': cmd_check(a[1] if len(a) > 1 else os.path.join(REPO, 'analysis', 'joas', 'main.tex'))
    else: sys.exit(__doc__)
