"""Non-tautological leakage, masking, and train/export regression checks."""
import argparse,sys,json,tempfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dataset import load,split_indices,fit_stats,graph,digest
from models import build_model
from protocol import NON_NEURAL, VARIANTS
from run import PROTOCOL, TEST_YEARS, TUNING_VALIDATION_YEARS, TUNING_YEAR, atomic_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    root=args.root;panel=root/'prepared/panel.npz';data_dir=root/'data'
    p=load(panel);torch.set_num_threads(2)
    for year in TEST_YEARS:
        tr,va,te=split_indices(p,year)
        assert max(p['years'][tr])<min(p['years'][va])<=max(p['years'][va])<year
        assert (p['phenology_max_year']<p['years']).all() and (p['ema_max_year']<p['years']).all()
        q={k:v.copy() for k,v in p.items()};future=q['years']>=year
        for key in ['x','s','w','y','emas']:q[key][future]=1e6
        for variant in ['full','no_ema','ema_03','graph_environment']:
            a=fit_stats(p,tr,variant);b=fit_stats(q,tr,variant)
            assert all(np.array_equal(a[k],b[k]) for k in a)
            assert np.array_equal(graph(p,tr,variant,42,data_dir),graph(q,tr,variant,42,data_dir))
    # Unavailable nodes cannot affect valid-node outputs, even with arbitrary huge features.
    torch.manual_seed(42);x=torch.randn(1,3,175,19);s=torch.randn(1,3,8);w=torch.randn(1,3,25,2);w[...,1]=0
    m=torch.tensor([[1.,1.,0.]]);adj=torch.ones(3,3)-torch.eye(3)
    variants=[]
    for variant in VARIANTS:
        if variant in NON_NEURAL:continue
        model=build_model(variant).eval()
        with torch.no_grad():
            a=model(x,s,w,m,adj)
            xx=x.clone();ss=s.clone();ww=w.clone();xx[:,2]=1e6;ss[:,2]=1e6;ww[:,2]=1e6
            b=model(xx,ss,ww,m,adj)
        assert torch.allclose(a[:,:2],b[:,:2],atol=1e-6),variant
        assert torch.isfinite(a).all(),variant
        model.train();model(x,s,w,m,adj)[:,:2].sum().backward();variants.append(variant)
    # The first two test folds must reuse the same tuning file and select epochs
    # on their own historical validation years.
    with tempfile.TemporaryDirectory(dir=root) as folder:
        d=Path(folder)
        poisoned=d/'poisoned.npz'
        q={k:v.copy() for k,v in p.items()}
        q['y'][q['years']==TUNING_YEAR]+=10000
        np.savez_compressed(poisoned,**q)
        command=[sys.executable,str(Path(__file__).with_name('run.py')),
          '--data-dir',str(data_dir),'--variant','full','--epochs','2','--patience','2','--device',args.device]
        outputs=[]
        for name,input_panel,years in [('normal',panel,TEST_YEARS[:2]),('poisoned',poisoned,TEST_YEARS[:1])]:
            out=d/name
            subprocess.run(command+['tune','--panel',str(input_panel),'--output',str(out),'--rates','.001'],check=True)
            tuning=out/'tuning.json'
            fixed=json.loads(tuning.read_text())
            assert fixed['validation_years']==TUNING_VALIDATION_YEARS
            for year in years:
                job=out/str(year)
                subprocess.run(command+['run','--panel',str(input_panel),'--output',str(job),
                  '--year',str(year),'--tuning',str(tuning)],check=True)
                report=json.loads((job/'result.json').read_text())
                assert report['initial_lr']==fixed['selected_lr'] and report['tuning_sha256']==digest(tuning)
                state=torch.load(job/'model.pt',map_location='cpu',weights_only=False)
                assert state['selection']['validation_years']==[year-2,year-1]
                assert state['selection']['lr']==fixed['selected_lr']
                if year==TUNING_YEAR:outputs.append(state)
        a,b=outputs
        assert a['selection']['best_epoch']==b['selection']['best_epoch']
        assert a['selection']['refit_lr_schedule']==b['selection']['refit_lr_schedule']
        assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    result={'passed':True,'protocol':PROTOCOL,'tested_variants':variants,'chronological_folds':len(TEST_YEARS),
      'shared_initial_lr_verified':True,'test_label_poisoning_model_state_identical':True,
      'normalization_future_perturbation_invariant':True,'graph_future_perturbation_invariant':True,'invalid_node_invariance':True}
    atomic_json(root/'verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
