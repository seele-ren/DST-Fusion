"""Verify complete prediction families and export reproducible summary tables."""
import sys,csv,json,statistics,hashlib
from pathlib import Path
from collections import defaultdict
from run import PROTOCOL, TUNING_VALIDATION_YEARS, atomic_json, metric, tuning_family

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    root=Path(sys.argv[1]);out=root/'summary';out.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text());rs=[];predictions=[]
    if manifest.get('protocol') != PROTOCOL:
        raise ValueError('Expected the independent-test manifest; use the historical result reader for archives')
    panel_sha256=hashlib.sha256((root/'prepared/panel.npz').read_bytes()).hexdigest()
    expected={(variant,year,seed) for variant in manifest['variants'] for year in manifest['years']
              for seed in ([42] if variant=='ema_only' else manifest['seeds'])}
    observed={}
    fixed_tuning={}
    for family in {tuning_family(v) for v in manifest['variants'] if v!='ema_only'}:
        raw=(root/'tuning'/family/'tuning.json').read_bytes()
        tuned=json.loads(raw)
        assert tuned['protocol']==PROTOCOL and tuned['variant']==family
        assert tuned['panel_sha256']==panel_sha256 and tuned['validation_years']==TUNING_VALIDATION_YEARS
        fixed_tuning[family]=(tuned.get('selected_lr'),hashlib.sha256(raw).hexdigest())
    for p in sorted((root/'experiments').rglob('result.json')):
        r=json.loads(p.read_text());raw=p.with_name('predictions.csv').read_bytes()
        key=(r['variant'],r['year'],r['seed'])
        assert key in expected, ('Unexpected or repeated experimental fold', key)
        expected.remove(key)
        assert r['protocol']==PROTOCOL and r['panel_sha256']==panel_sha256
        if r['variant']!='ema_only':
            assert (r['initial_lr'],r['tuning_sha256'])==fixed_tuning[tuning_family(r['variant'])]
        assert hashlib.sha256(raw).hexdigest()==r['prediction_sha256']
        rows=list(csv.DictReader(p.with_name('predictions.csv').open()))
        m=metric([float(z['y_true']) for z in rows],[float(z['y_pred']) for z in rows])
        assert abs(m['rmse']-r['rmse'])<1e-8 and len(rows)==r['n']
        assert all(int(z['year'])==r['year'] and int(z['seed'])==r['seed'] and z['variant']==r['variant'] for z in rows)
        labels={z['county']:float(z['y_true']) for z in rows}
        assert len(labels)==len(rows) and observed.setdefault(r['year'],labels)==labels, key
        rs.append(r);predictions+=rows
    assert not expected and len(rs)==manifest['experiment_tasks']
    by=defaultdict(list)
    for r in rs:by[(r['variant'],r['seed'])].append(r)
    means=[]
    for (v,s),values in sorted(by.items()):
        assert sorted(x['year'] for x in values)==manifest['years']
        means.append({'variant':v,'seed':s,**{f'annual_mean_{k}':statistics.mean(r[k] for r in values) for k in ['rmse','mae','bias','r2']}})
    summary=[]
    for v in manifest['variants']:
        vv=[r for r in means if r['variant']==v]
        summary.append({'variant':v,'seeds':len(vv),'annual_mean_rmse':statistics.mean(r['annual_mean_rmse'] for r in vv),
          'between_seed_sd':statistics.stdev(r['annual_mean_rmse'] for r in vv) if len(vv)>1 else None,
          'annual_mean_mae':statistics.mean(r['annual_mean_mae'] for r in vv),'annual_mean_r2':statistics.mean(r['annual_mean_r2'] for r in vv),
          'protocol':PROTOCOL,'panel_sha256':panel_sha256})
    write_csv(out/'all_results.csv',rs);write_csv(out/'all_predictions.csv',predictions);write_csv(out/'per_seed_means.csv',means);write_csv(out/'model_summary.csv',summary)
    atomic_json(out/'validation.json',{'passed':True,'jobs':len(rs),'prediction_rows':len(predictions),'all_checkpoint_reload_checks_passed':all(r['checkpoint_reload_max_abs_difference']<=1e-4 for r in rs)})
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
