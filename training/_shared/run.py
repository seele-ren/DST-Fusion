"""Select the initial learning rate once, then early-stop and refit each fold."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,json,time,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from rf import refit_rf, tune_rf
from ema_only import predict_history
from models import build_model
from dataset import load,prepare,split_indices,fit_stats,tensors,graph,digest
from protocol import (
    NEURAL_BASELINES, NON_NEURAL, PROTOCOL, SEARCH_RATES, TEST_YEARS,
    TUNING_SEED, TUNING_VALIDATION_YEARS, TUNING_YEAR, VARIANTS,
)

def tuning_family(variant):
    if variant in (*NEURAL_BASELINES, 'rf'):
        return variant
    return 'transformer_matched' if variant.startswith('transformer') else 'full'

def load_tuning(path, variant, panel_sha256):
    if not path:
        raise ValueError('Neural runs require the fixed initial learning rate file (--tuning)')
    tuned = json.loads(Path(path).read_text())
    if (tuned.get('protocol') != PROTOCOL or tuned.get('variant') != tuning_family(variant)
        or tuned.get('validation_years') != TUNING_VALIDATION_YEARS
        or tuned.get('panel_sha256') != panel_sha256):
        raise ValueError(f'Tuning file does not match the fixed protocol, model family or input panel: {path}')
    return tuned

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=False)

def atomic_json(p,obj):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def metric(y,p):
    e=np.asarray(p,dtype=float)-np.asarray(y,dtype=float);y=np.asarray(y,dtype=float)
    return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),'bias':float(np.mean(e)),
      'mape':float(np.mean(abs(e)/np.maximum(abs(y),1e-6))*100),'nrmse_mean':float(np.sqrt(np.mean(e*e))/abs(y.mean())),
      'r2':float(1-(e*e).sum()/((y-y.mean())**2).sum()) if ((y-y.mean())**2).sum()>0 else None}

def forward(model,batch,adj):
    x,s,w,y,m=batch;return model(x,s,w,m,adj)
def step(model,batch,adj,opt,chunk=4):
    model.train();opt.zero_grad(set_to_none=True);total=float(batch[-1].sum());loss_sum=0.
    for start in range(0,len(batch[0]),chunk):
        b=tuple(a[start:start+chunk] for a in batch);pred=forward(model,b,adj);valid=b[-1].bool()
        loss=nn.functional.huber_loss(pred[valid],b[-2][valid],reduction='sum')/total
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        loss.backward();loss_sum+=float(loss.detach())
    nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();return loss_sum
@torch.no_grad()
def predict(model,batch,adj):
    model.eval();return torch.cat([forward(model,tuple(a[i:i+4] for a in batch),adj) for i in range(0,len(batch[0]),4)])
def optimizer(model,lr):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(no_decay if name=='alpha' else decay).append(p)
    return torch.optim.Adam([{'params':decay,'weight_decay':1e-4},{'params':no_decay,'weight_decay':0.}],lr=lr)

def select(p,variant,target,seed,lr,max_epochs,patience,device,data_dir):
    tr,va,_=split_indices(p,target);stats=fit_stats(p,tr,variant)
    bt,_,_=tensors(p,tr,stats,variant,device);bv,_,_=tensors(p,va,stats,variant,device)
    adj=torch.as_tensor(graph(p,tr,variant,seed,data_dir),device=device)
    seed_all(seed);model=build_model(variant).to(device);opt=optimizer(model,lr)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=8,cooldown=2,min_lr=1e-6)
    best=float('inf');best_epoch=0;bad=0;history=[]
    started=time.monotonic()
    for epoch in range(1,max_epochs+1):
        used_lr=float(opt.param_groups[0]['lr']);loss=step(model,bt,adj,opt)
        pv=predict(model,bv,adj);valid=bv[-1].bool()
        rmse=float(torch.sqrt(torch.mean((pv[valid]-bv[-2][valid])**2)))*stats['ys']
        history.append({'epoch':epoch,'lr':used_lr,'train_huber':loss,'historical_validation_rmse':rmse})
        if rmse<best-1e-4:best=rmse;best_epoch=epoch;bad=0
        else:bad+=1
        sch.step(rmse)
        if bad>=patience:break
    result={'variant':variant,'target_year':target,'seed':seed,'lr':lr,'best_epoch':best_epoch,'validation_rmse':best,
     'history':history,'refit_lr_schedule':[h['lr'] for h in history[:best_epoch]],'seconds':time.monotonic()-started,
     'train_years':sorted(set(p['years'][tr].tolist())),'validation_years':sorted(set(p['years'][va].tolist())),
     'test_labels_used':False}
    del model,opt,bt,bv;torch.cuda.empty_cache();return result





def tune(args):
    if args.year != TUNING_YEAR:
        raise ValueError(f'Initial learning rate selection is run once, with validation years {TUNING_VALIDATION_YEARS}')
    if args.variant == 'ema_only' or args.variant != tuning_family(args.variant):
        raise ValueError('Select a model family for initial tuning')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    dest=out/'tuning.json'
    if dest.exists():raise FileExistsError(dest)
    p=load(args.panel)
    if args.variant == 'rf':
        selection = tune_rf(p, metric)
    else:
        trials=[select(p,args.variant,TUNING_YEAR,TUNING_SEED,lr,args.epochs,args.patience,args.device,args.data_dir)
                for lr in args.rates]
        winner=min(trials,key=lambda r:r['validation_rmse'])
        selection={'selected_lr':winner['lr'],'validation_years':winner['validation_years'],'trials':trials}
    if selection['validation_years'] != TUNING_VALIDATION_YEARS:
        raise ValueError(f'Both validation years are required: {TUNING_VALIDATION_YEARS}')
    atomic_json(dest,{'protocol':PROTOCOL,'variant':args.variant,'criterion':'historical validation RMSE only',
      'panel_sha256':digest(args.panel),**selection})



def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError('Completed experiment exists')
    panel_sha256=digest(args.panel)
    tuned=load_tuning(args.tuning,args.variant,panel_sha256) if args.variant!='ema_only' else None
    p=load(args.panel);tr,va,te=split_indices(p,args.year);refit=np.r_[tr,va]
    applies_fold_selection = args.variant not in NON_NEURAL
    split={'fit_years':sorted(set(p['years'][tr].tolist())) if applies_fold_selection else [],
      'validation_years':sorted(set(p['years'][va].tolist())) if applies_fold_selection else [],
      'refit_years':sorted(set(p['years'][refit].tolist())),'test_year':args.year,
      'selection_uses_test':False,'statistics_use_test':False,'test_evaluation':'after final model frozen',
      'initial_lr_validation_years':tuned['validation_years'] if applies_fold_selection else [],
      'historical_features':'strictly earlier than each sample year','data_provenance_limitations':'See provenance.json; fold-specific upstream masks unavailable.'}
    atomic_json(out/'split.json',split)
    if args.variant=='ema_only':
        pred, y, names = predict_history(p, te)
        selection=None;params=0;train_seconds=0.;max_difference=0.;tune_hash=None
    elif args.variant=='rf':
        pred, train_seconds = refit_rf(p, refit, te, tuned['selected_config'], args.seed, out/'model.pkl')
        y, names = p['y'][te], p['counties'][te]
        selection=None;params=None;max_difference=0.;tune_hash=digest(args.tuning)
    else:
        selection=select(p,args.variant,args.year,args.seed,tuned['selected_lr'],args.epochs,args.patience,args.device,args.data_dir)
        atomic_json(out/'selection.json',selection)
        stats=fit_stats(p,refit,args.variant);bt,_,_=tensors(p,refit,stats,args.variant,args.device)
        a=graph(p,refit,args.variant,args.seed,args.data_dir);adj=torch.as_tensor(a,device=args.device)
        seed_all(args.seed);model=build_model(args.variant).to(args.device);opt=optimizer(model,selection['lr'])
        begin=time.monotonic()
        # Frozen historical-validation learning-rate schedule. There is no validation/test callback here.
        for lr in selection['refit_lr_schedule']:
            for group in opt.param_groups:group['lr']=lr
            step(model,bt,adj,opt)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)
        train_seconds=time.monotonic()-begin
        params=sum(v.numel() for v in model.parameters() if v.requires_grad)
        cp={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'stats':stats,'adj':a,
         'variant':args.variant,'seed':args.seed,'target_year':args.year,'selection':selection,'split':split,
         'names':p['names'].tolist(),'panel_sha256':panel_sha256}
        torch.save(cp,out/'model.pt')
        # First use of held-out test tensors and labels, after final fit/checkpoint is complete.
        test,years,all_names=tensors(p,te,stats,args.variant,args.device)
        pred_tensor=predict(model,test,adj);mask=test[-1].bool()
        y=(test[-2][mask]*stats['ys']+stats['ym']).cpu().numpy()
        pred=(pred_tensor[mask]*stats['ys']+stats['ym']).cpu().numpy()
        names=np.array(all_names)[mask.cpu().numpy()[0]]
        del model,opt;torch.cuda.empty_cache()
        loaded=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
        check=build_model(args.variant).to(args.device);check.load_state_dict(loaded['model'],strict=True)
        restored=(predict(check,test,torch.as_tensor(loaded['adj'],device=args.device))[mask]*stats['ys']+stats['ym']).cpu().numpy()
        max_difference=float(np.max(abs(restored-pred)))
        assert np.allclose(restored,pred,atol=1e-5,rtol=1e-5),'Checkpoint reload mismatch'
        tune_hash=digest(args.tuning)
    source={str(p['counties'][i]):float(p['y'][i]) for i in te}
    assert len(names)==len(set(names))==len(te)
    assert np.allclose(y,[source[str(n)] for n in names],atol=1e-4,rtol=1e-7),'County-label mismatch'
    y=np.asarray([source[str(n)] for n in names],dtype=np.float64)
    rows=[{'variant':args.variant,'seed':args.seed,'year':args.year,'county':str(n),'y_true':float(t),'y_pred':float(v),'error':float(v-t)} for n,t,v in zip(names,y,pred)]
    with (out/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'variant':args.variant,'year':args.year,'seed':args.seed,**metric(y,pred),'best_epoch':selection['best_epoch'] if selection else None,
     'parameter_count_trainable':params,'selection_seconds':selection['seconds'] if selection else 0.,'refit_seconds':train_seconds,
     'checkpoint_reload_max_abs_difference':max_difference,'county_labels_verified':True,'panel_sha256':panel_sha256,
     'initial_lr':tuned.get('selected_lr') if tuned else None,'tuning_sha256':tune_hash,'prediction_sha256':digest(out/'predictions.csv'),'protocol':PROTOCOL,
     'torch_version':str(torch.__version__),'gpu':torch.cuda.get_device_name(args.device)
     if args.variant not in NON_NEURAL and torch.device(args.device).type=='cuda' else 'cpu'}
    atomic_json(out/'result.json',result);print(json.dumps(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','tune','run']);ap.add_argument('--data-dir',required=True);ap.add_argument('--panel');ap.add_argument('--output',required=True)
    ap.add_argument('--variant',choices=VARIANTS,default='full');ap.add_argument('--year',type=int,choices=TEST_YEARS,default=TUNING_YEAR);ap.add_argument('--seed',type=int,default=TUNING_SEED)
    ap.add_argument('--tuning');ap.add_argument('--epochs',type=int,default=500);ap.add_argument('--patience',type=int,default=30)
    ap.add_argument('--rates',type=float,nargs='+',default=SEARCH_RATES);ap.add_argument('--device',default='cuda');args=ap.parse_args()
    torch.set_num_threads(2)
    if args.action=='prepare':print(json.dumps(prepare(args.data_dir,args.output),ensure_ascii=False))
    elif args.action=='tune':tune(args)
    else:run(args)
if __name__=='__main__':main()
