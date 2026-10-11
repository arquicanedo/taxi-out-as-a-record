"""Pull EUROCONTROL PRU open datasets (ansperformance.eu / eurocontrol.int)."""
import urllib.request, pandas as pd, io, os   # needs --with openpyxl for the xlsx
BASE="https://www.eurocontrol.int/performance/data/download/xls/"
FILES=["Airport_Arrival_ATFM_Delay.xlsx","ATFM_Slot_Adherence.xlsx",
       "Taxi-Out_Additional_Time.xlsx","Airport_Traffic.xlsx",
       # s88: found via the AIU portal manifest at
       # github.com/euctrl-pru/aiu-portal -> data/dwnld.yml
       "ATC_Pre-Departure_Delay.xlsx",   # Jan 2016 - Jun 2026, delay at the stand
       "All_Pre-Departure_Delay.xlsx",   # Jan 2020 - Jun 2026, all-cause
       "Taxi-In_Additional_Time.xlsx",
       "ASMA_Additional_Time.xlsx"]
# repo-relative, so the download lands correctly wherever the repository is checked out
PRU=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'reference','pru')
os.makedirs(PRU,exist_ok=True)
PARQUET={"Airport_Arrival_ATFM_Delay.xlsx","ATFM_Slot_Adherence.xlsx"}   # the two the model reads
for f in FILES:
    try:
        raw=urllib.request.urlopen(BASE+f,timeout=180).read()
        p=os.path.join(PRU,f)
        open(p,'wb').write(raw)
        xl=pd.ExcelFile(io.BytesIO(raw))
        print(f'{f}  {len(raw)/1e6:.1f} MB  sheets={xl.sheet_names}')
        for s in xl.sheet_names[:3]:
            df=xl.parse(s,nrows=3)
            print(f'    [{s}] cols={list(df.columns)[:12]}')
        # the model reads the DATA sheet as parquet (v17_features.py). This conversion reproduces the
        # archived 2026-09-01 parquets exactly (DataFrame.equals, checked 2026-09-23), but PRU republishes
        # monthly and revises, so a fresh pull is a NEWER RELEASE: the shipped model used release V.1.0 of
        # 2026-08-14 (META sheet). Keep the archived snapshot to reproduce v100; see EXTERNAL-DATA.md.
        if f in PARQUET:
            q=os.path.join(PRU,f.replace('.xlsx','.parquet'))
            if os.path.exists(q) and not os.environ.get('PRU_OVERWRITE'):
                print(f'    kept {q} (archived snapshot; PRU_OVERWRITE=1 replaces it)')
            else:
                xl.parse('DATA').to_parquet(q,index=False); print(f'    wrote {q}')
    except Exception as e:
        print(f'{f}  FAILED {type(e).__name__}: {str(e)[:90]}')
