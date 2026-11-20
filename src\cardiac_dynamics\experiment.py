import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge

def signal(n=900, seed=27):
    rng=np.random.default_rng(seed); x=np.zeros(n); x[:2]=[0.2,0.5]
    for t in range(2,n):
        x[t]=1.72*x[t-1]-0.78*x[t-2]+0.06*np.sin(x[t-1]*3)+rng.normal(0,0.025)
    return x

def rollout(model, start, steps, physics=False, hybrid=False):
    out=list(start)
    for _ in range(steps):
        prior=1.68*out[-1]-0.76*out[-2]
        learned=float(model.predict([[out[-1],out[-2],prior]])[0])
        out.append(prior if physics else (0.5*prior+0.5*learned if hybrid else learned))
    return np.array(out[2:])

def evaluate(seed=27):
    x=signal(seed=seed); X=[]; y=[]
    for t in range(2,650):
        prior=1.68*x[t-1]-0.76*x[t-2]; X.append([x[t-1],x[t-2],prior]); y.append(x[t])
    model=Ridge(alpha=.4).fit(X,y); truth=x[650:]
    start=x[648:650]
    preds={"physics_only":rollout(model,start,len(truth),physics=True),"data_only":rollout(model,start,len(truth)),"hybrid":rollout(model,start,len(truth),hybrid=True)}
    return {k:{"rollout_rmse":float(np.sqrt(np.mean((v-truth)**2)))} for k,v in preds.items()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,default=Path("outputs/demo")); p.add_argument("--seed",type=int,default=27); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); r=evaluate(a.seed); (a.output/"metrics.json").write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=="__main__": main()
