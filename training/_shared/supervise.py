"""Persistent two-GPU queue. No task selection based on outer-test results."""
import argparse,os,sys,json,time,subprocess,hashlib,datetime,fcntl
from pathlib import Path
import torch
from run import PROTOCOL, SEARCH_RATES, TEST_YEARS, TUNING_SEED, TUNING_VALIDATION_YEARS, atomic_json, tuning_family
from protocol import SEEDS, VARIANTS

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--variants',nargs='+',choices=VARIANTS,default=VARIANTS)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    variants=list(dict.fromkeys(args.variants))
    families=sorted({tuning_family(v) for v in variants if v!='ema_only'})
    root=args.root.resolve();root.mkdir(exist_ok=True)
    lock=(root/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=root/'manifest.json'
    old=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old is not None and old.get('protocol') != PROTOCOL:
        raise ValueError('Use a new run directory for the fixed initial learning rate protocol')
    verification=json.loads((root/'verification.json').read_text())
    assert verification['passed'] and verification.get('protocol')==PROTOCOL, 'Run verify.py for this protocol first'
    (root/'logs').mkdir(exist_ok=True)
    script=Path(__file__).with_name('run.py').resolve()
    common=[sys.executable,'-u',str(script),'--data-dir',str(root/'data'),'--panel',str(root/'prepared/panel.npz'),'--epochs','500','--patience','30','--device',args.device]
    slots=range(min(2,torch.cuda.device_count())) if args.device=='cuda' else range(1)
    if not slots:
        parser.error('CUDA requested but no CUDA device is visible')
    tunes=[];runs=[]
    tuning_paths={}
    for family in families:
        out=root/'tuning'/family
        tuning_paths[family]=out/'tuning.json'
        tunes.append({'id':f'tune_{family}','cmd':common+['tune','--variant',family,'--output',str(out)],'done':str(tuning_paths[family])})
    for variant in variants:
        for year in TEST_YEARS:
            for seed in ([42] if variant=='ema_only' else SEEDS):
                out=root/'experiments'/variant/str(year)/str(seed)
                cmd=common+['run','--variant',variant,'--year',str(year),'--seed',str(seed),'--output',str(out)]
                if variant!='ema_only':
                    cmd+=['--tuning',str(tuning_paths[tuning_family(variant)])]
                runs.append({'id':f'{variant}_{year}_{seed}','cmd':cmd,'done':str(out/'result.json')})
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'protocol':PROTOCOL,
      'years':list(TEST_YEARS),'seeds':SEEDS,'variants':variants,
      'tuning_tasks':len(tunes),'experiment_tasks':len(runs),'inner_validation_years':2,'search_rates':list(SEARCH_RATES),
      'initial_lr_validation_years':TUNING_VALIDATION_YEARS,'initial_lr_search_seed':TUNING_SEED,
      'search_policy':'one historical search per family; neural models share the selected initial LR and RF shares the selected parameters',
      'early_stopping_policy':'run separately per fold, variant and seed with the fixed initial LR; validate on the two years before the test year',
      'refit_policy':'reset model seed; refit all years before target for frozen number of epochs and frozen historical LR schedule',
      'parameter_selection_metric':'RMSE on the two historical validation years pooled by county-year',
      'objective':'Huber; loss over valid counties only; accumulated gradient over all fitting-year graphs',
      'source_sha256':{str(p.relative_to(script.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in script.parents[1].rglob('*.py')},
      'blocked':json.loads((root/'prepared/provenance.json').read_text())['blocked_experiments']}
    if old is None:atomic_json(manifest_path,manifest)
    else:
        assert old['source_sha256']==manifest['source_sha256'],'Source changed within run'
        assert old['variants']==variants,'Variant list changed within run'
    for phase,tasks in [('tuning',tunes),('experiments',runs)]:
        pending=[t for t in tasks if not Path(t['done']).exists()];active={};completed=len(tasks)-len(pending);failed=[]
        while pending or active:
            for gpu in slots:
                if gpu in active or not pending:continue
                task=pending.pop(0);env=dict(os.environ,PYTHONPATH=os.pathsep.join(filter(None,(str(root/'deps'),os.environ.get('PYTHONPATH')))),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
                if args.device=='cuda':
                    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
                    env['CUDA_VISIBLE_DEVICES']=(visible.split(',')[gpu] if visible is not None else str(gpu))
                log=(root/'logs'/f"{task['id']}.log").open('a')
                proc=subprocess.Popen(task['cmd'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
                active[gpu]=(task,proc,log,time.time())
            for gpu,(task,proc,log,start) in list(active.items()):
                rc=proc.poll()
                if rc is None:continue
                log.close();del active[gpu]
                if rc==0 and Path(task['done']).exists():completed+=1
                else:failed.append({'task':task['id'],'exit_code':rc})
            atomic_json(root/'status.json',{'state':'running' if not failed else 'failed','phase':phase,'completed':completed,'total':len(tasks),
                'pending':len(pending),'active':[{'gpu':gpu,'task':t['id'],'pid':p.pid,'elapsed_seconds':time.time()-start} for gpu,(t,p,l,start) in active.items()],
                'failed':failed,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            if failed:
                # Finish currently running jobs, but do not launch further tasks after a failure.
                for task,proc,log,start in active.values():proc.wait();log.close()
                raise RuntimeError(f'Job failure: {failed}; inspect logs')
            time.sleep(5)
    subprocess.run([sys.executable,str(Path(__file__).with_name('summarize.py')),str(root)],check=True)
    atomic_json(root/'status.json',{'state':'complete','experiments':len(runs),'tuning_tasks':len(tunes),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
if __name__=='__main__':main()
