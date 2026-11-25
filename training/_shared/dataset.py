"""Historical feature construction and strict chronological model-selection splits."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from _shared.data import CsvYieldDataset,build_county_adjacency,_load_yield_table
from importlib import import_module
FEATURES='WDRVI_median,GCI_median,EVI_median,NDWI_median,NIRv_median,GDD,KDD,PRCP,VPD,WDRVI_p90,WDRVI_p10,GCI_p90,GCI_p10,EVI_p90,EVI_p10,NDWI_p90,NDWI_p10,NIRv_p90,NIRv_p10'.split(',')
STATIC='SOS,VGS,RGS,sand_median,silt_median,clay_median'.split(',')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(data_dir,out):
    d=Path(data_dir);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ds=CsvYieldDataset(str(d/'All_Data.csv'),FEATURES,yield_csv=str(d/'Yield_Data.xlsx'),yield_col='yield',yield_county_col='name',yield_year_col='year',
      weekly_csv=str(d/'SIF_Weekly.csv'),weekly_cols=['SIF_median','sif_missing_flag'],start_doy=128,end_doy=302,static_cols=STATIC,
      static_csv=str(d/'jilin_phenology_2001-2020.csv'),soil_csv=str(d/'Jilin_Soil_Texture_Combined_Static.csv'),soil_cols=STATIC[3:])
    # Always reconstruct phenology from pre-sample-year records, never same-year phenology.
    with (d/'jilin_phenology_2001-2020.csv').open(encoding='utf-8-sig') as f:pr=list(csv.DictReader(f))
    history={}
    for r in pr:history.setdefault(r['name'],[]).append((int(r['year']),[float(r[k]) for k in STATIC[:3]]))
    targets=_load_yield_table(str(d/'Yield_Data.xlsx'),'yield','name','year')
    # Rebuild EMA for arbitrary coefficients and persist max source years for auditing.
    emas=np.zeros((len(ds.y),3),np.float32);ema_miss=np.ones(len(ds.y),np.float32)
    py=np.zeros(len(ds.y),np.int32);ey=np.zeros(len(ds.y),np.int32)
    static=ds.static.copy();phen_missing=[]
    for i,(county,year) in enumerate(zip(ds.counties,ds.years)):
        prev=[(t,v) for t,v in history.get(str(county),[]) if 2001<=t<int(year)]
        if prev:static[i,:3]=np.mean([v for t,v in prev],axis=0);py[i]=max(t for t,v in prev)
        else:static[i,:3]=0;phen_missing.append(i)
        prev_y=sorted((t,v) for (c,t),v in targets.items() if c==county and t<year)
        if prev_y:
            for j,alpha in enumerate([.3,.5,.7]):
                value=prev_y[0][1]
                for t,v in prev_y[1:]:value=alpha*v+(1-alpha)*value
                emas[i,j]=value
            ema_miss[i]=0;ey[i]=prev_y[-1][0]
    static[:,-2]=emas[:,1];static[:,-1]=ema_miss
    assert np.all(py<ds.years) and np.all(ey<ds.years)
    names,adj=build_county_adjacency(str(d/'吉林省.json'))
    active=set(map(str,ds.counties));ids=[i for i,n in enumerate(names) if n in active]
    names=[names[i] for i in ids];adj=adj[np.ix_(ids,ids)]
    assert active==set(names),'Unmatched county identifiers'
    # Explicitly discard years after 2019; they cannot affect new evaluation.
    keep=(ds.years>=2001)&(ds.years<=2019)
    np.savez_compressed(out/'panel.npz',x=ds.x[keep],y=ds.y[keep],s=static[keep],w=ds.weekly[keep],years=ds.years[keep],
      counties=np.asarray(ds.counties[keep],dtype=str),names=np.asarray(names,dtype=str),adj=adj,emas=emas[keep],
      phenology_max_year=py[keep],ema_max_year=ey[keep])
    provenance={'inputs':{p.name:digest(p) for p in d.iterdir() if p.is_file() and p.suffix in ['.csv','.xlsx','.json']},
      'features':FEATURES,'static':STATIC+['historical_yield_EMA','EMA_missing'],
      'phenology':'mean of available 2001..sample_year-1, zeros when unavailable',
      'ema':'available yield records strictly before sample year; earliest supplied year may be 2000',
      'missing_phenology_samples':len(phen_missing),'input_limitations':['Existing processed county time series reused. Fold-specific maize-mask provenance is unavailable.','HCSIF is a retrospective reconstructed product; this is not an as-of-date operational data availability experiment.'],
      'blocked_experiments':{'mask_threshold':'Requires alternate-threshold county extractions and per-fold provenance; no raster masks present.',
      'raw_filter_sensitivity':'Requires pre-interpolation/pre-smoothing series and quality masks; smoothing the processed table again would not test the original preprocessing.'}}
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    return provenance

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def split_indices(panel,target):
    y=panel['years'];train=np.where((y>=2001)&(y<=target-3))[0];val=np.where((y>=target-2)&(y<target))[0];test=np.where(y==target)[0]
    assert len(train) and len(val) and len(test)
    assert set(y[val])=={target-2,target-1}, 'Both historical validation years are required'
    assert max(y[train])<min(y[val]) and max(y[val])<target
    assert not set(train)&set(val) and not set(train)&set(test) and not set(val)&set(test)
    return train,val,test

def fit_stats(panel,ids,variant):
    s=static_values(panel,variant)[ids]
    def stats(x,axes):
        mu=x.mean(axis=axes);sd=x.std(axis=axes);return mu,np.where(sd<1e-6,1.,sd)
    xm,xs=stats(panel['x'][ids],(0,1));sm,ss=stats(s,0)
    y=panel['y'][ids];w=panel['w'][ids];valid=(w[...,1]<=0)&np.isfinite(w[...,0])
    wm=float(w[...,0][valid].mean()) if valid.any() else 0.;ws=float(w[...,0][valid].std()) if valid.any() else 1.
    return dict(xm=xm,xs=xs,sm=sm,ss=ss,ym=float(y.mean()),ys=max(float(y.std()),1e-6),wm=wm,ws=max(ws,1e-6))
def static_values(p,variant):
    s=p['s'].copy()
    coefficient = getattr(import_module(variant), 'EMA_FEATURE', 0.5)
    if coefficient is None:
        s[:, -2:] = 0
    elif coefficient in (0.3, 0.7):
        s[:, -2] = p['emas'][:, {0.3: 0, 0.7: 2}[coefficient]]
    return s

def graph(p,fit_ids,variant,seed,data_dir):
    a=p['adj'].copy();n=len(a)
    topology = getattr(import_module(variant), 'GRAPH_TOPOLOGY', 'adjacency')
    if topology == 'self_loop':
        return np.zeros_like(a)
    if topology == 'random':
        # Permutation preserves degree sequence, number of edges and components.
        perm=np.random.default_rng(seed).permutation(n);return a[np.ix_(perm,perm)]
    if topology == 'adjacency':
        return a
    if topology == 'environment':
        # Node climate/soil descriptors use only the fitting years, never validation/test inputs.
        feats=[]
        for name in p['names']:
            ids=[i for i in fit_ids if p['counties'][i]==name]
            assert ids,'Missing training county for similarity graph'
            feats.append(np.r_[p['x'][ids,:,5:9].mean((0,1)),p['s'][ids,3:6].mean(0)])
        f=np.asarray(feats);f=(f-f.mean(0))/np.maximum(f.std(0),1e-6)
        dist=((f[:,None]-f[None,:])**2).sum(-1)
    else:
        import math
        def points(x):
            if isinstance(x,list) and len(x)>=2 and isinstance(x[0],(int,float)):yield x[:2]
            elif isinstance(x,list):
                for item in x:yield from points(item)
        geo=json.loads((Path(data_dir)/'吉林省.json').read_text());centers={}
        for feat in geo['features']:
            centers[feat['properties'].get('name')]=np.mean(list(points(feat['geometry']['coordinates'])),axis=0)
        f=np.radians([centers[name] for name in p['names']]);lon=f[:,0];lat=f[:,1]
        dist=np.sin((lat[:,None]-lat[None,:])/2)**2+np.cos(lat[:,None])*np.cos(lat[None,:])*np.sin((lon[:,None]-lon[None,:])/2)**2
    # Global nearest pairs with exactly the same edge count; isolates are reported, not hidden.
    edges=int(a.sum()/2);r,c=np.triu_indices(n,1);order=np.argsort(dist[r,c],kind='stable')[:edges]
    a=np.zeros_like(a);a[r[order],c[order]]=1;a[c[order],r[order]]=1;return a

def tensors(p,ids,stats,variant,device):
    import torch
    years=sorted(set(map(int,p['years'][ids])));names=list(p['names']);ni={name:i for i,name in enumerate(names)}
    arrays=[np.zeros((len(years),len(names),*shape),np.float32) for shape in [(175,19),(8,),(25,2),(),()]]
    x,s,w,y,mask=arrays;sv=static_values(p,variant);yi={year:i for i,year in enumerate(years)}
    for i in ids:
        a,b=yi[int(p['years'][i])],ni[p['counties'][i]]
        assert mask[a,b]==0,'duplicate county/year'
        x[a,b]=(p['x'][i]-stats['xm'])/stats['xs'];s[a,b]=(sv[i]-stats['sm'])/stats['ss']
        w[a,b]=p['w'][i];w[a,b,:,0]=(w[a,b,:,0]-stats['wm'])/stats['ws']
        w[a,b,w[a,b,:,1]>0,0]=0
        y[a,b]=(p['y'][i]-stats['ym'])/stats['ys'];mask[a,b]=1
    assert all(np.isfinite(a).all() for a in arrays)
    return tuple(torch.as_tensor(a,device=device) for a in arrays),years,names
