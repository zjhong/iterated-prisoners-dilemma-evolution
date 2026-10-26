#!/usr/bin/env python3
"""Exact expected iterated-PD payoffs and deterministic replicator dynamics.
Run: python experiment.py --output results
Requires Python 3.10+, numpy, matplotlib. No network or random seed needed.
"""
import argparse, itertools, json, os, tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'ipd-matplotlib'))
import numpy as np
NAMES=['Always cooperate','Always defect','Grim trigger','Tit for tat','Random','Generous tit for tat']
IDS=['always_c','always_d','grim','tft','random','generous_tft']
STATES=list(itertools.product((0,1), repeat=4)) # last A/B, ever-defected A/B
INDEX={s:i for i,s in enumerate(STATES)}
PAY=np.array([[3.,0.],[5.,1.]])

def defect_probability(strategy, last_opponent=0, opponent_ever=0, first=False):
    if strategy==0: return 0.
    if strategy==1: return 1.
    if strategy==4: return .5
    if first: return 0.
    if strategy==2: return float(opponent_ever)
    if strategy==3: return float(last_opponent)
    if strategy==5: return (2/3)*last_opponent
    raise ValueError(strategy)

def action_prob(strategy,last,ever,first,noise):
    q=defect_probability(strategy,last,ever,first)
    q=noise+(1-2*noise)*q
    return [1-q,q]

def outcomes(a,b,state,noise,first=False):
    la,lb,ea,eb=state
    pa=action_prob(a,lb,eb,first,noise)
    pb=action_prob(b,la,ea,first,noise)
    for x,y in itertools.product((0,1),repeat=2):
        yield INDEX[(x,y,int(ea or x),int(eb or y))], pa[x]*pb[y], PAY[x,y],PAY[y,x]

def pair_payoffs(a,b,noise,lengths=(20,200,1000)):
    """Propagate all possible histories through sufficient 16-state memory.
    This sums probability mass exactly (up to floating-point), not Monte Carlo.
    Each game starts with fresh memory. Players act simultaneously.
    """
    transition=np.zeros((16,16)); rewards=np.zeros((16,2))
    for k,state in enumerate(STATES):
        for j,p,ra,rb in outcomes(a,b,state,noise):
            transition[k,j]+=p; rewards[k]+=p*np.array([ra,rb])
    assert np.allclose(transition.sum(1),1)
    mass=np.zeros(16); total=np.zeros(2)
    for j,p,ra,rb in outcomes(a,b,(0,0,0,0),noise,first=True):
        mass[j]+=p; total+=p*np.array([ra,rb])
    saved={}
    if 1 in lengths: saved[1]=total.copy()
    for t in range(2,max(lengths)+1):
        total+=mass@rewards
        mass=mass@transition
        if t in lengths: saved[t]=total.copy()
    assert np.isclose(mass.sum(),1)
    return saved

def payoff_matrices(noise,lengths=(20,200,1000)):
    matrices={t:np.zeros((6,6)) for t in lengths}
    for a in range(6):
        for b in range(a,6):
            scores=pair_payoffs(a,b,noise,lengths)
            for t,(sa,sb) in scores.items():
                matrices[t][a,b]=sa if a!=b else (sa+sb)/2
                matrices[t][b,a]=sb if a!=b else (sa+sb)/2
    for t,m in matrices.items(): assert np.all((m>=0)&(m<=5*t+1e-8))
    return matrices

def evolve(payoffs,generations=5000,mutation=0.):
    """Infinite well-mixed population, fitness-proportional reproduction.
    log-space avoids numerical extinction; no integer rounding or drift.
    Mutation, when enabled: mix mu of newborn frequencies with uniform.
    """
    k=len(payoffs); logp=np.full(k,-np.log(k)); history=[]
    for g in range(generations+1):
        p=np.exp(logp); history.append(p)
        if g==generations: break
        fitness=payoffs@p
        logp+=np.log(fitness)
        z=np.max(logp); logp-=z+np.log(np.exp(logp-z).sum())
        if mutation:
            p=(1-mutation)*np.exp(logp)+mutation/k; logp=np.log(p)
    history=np.array(history)
    assert np.allclose(history.sum(1),1) and np.all(history>=0)
    return history

def tests():
    assert defect_probability(2,0,1)==1
    assert defect_probability(3,0,1)==0
    assert defect_probability(5,1,1)==2/3
    m=payoff_matrices(0,(200,))[200]
    expected=np.array([[600,0,600,600],[1000,200,204,204],[600,199,600,600],[600,199,600,600]])
    assert np.allclose(m[:4,:4],expected)
    assert np.allclose(m[:4,:4].sum(1),[1800,1608,1999,1999])
    assert np.isclose(m[4,4],450)
    assert np.isclose(pair_payoffs(0,1,1,(1,))[1][0],5)
    uniform=evolve(np.full((4,4),3),10)
    assert np.allclose(uniform,.25)
    half=payoff_matrices(.5,(200,))[200]
    assert np.allclose(half,450)
    assert np.allclose(evolve(half,10),1/6)
    return {'status':'passed','checks':['baseline full 4x4 payoff matrix','baseline totals','grim persistent memory','TFT last-action memory','generous forgiveness probability','random self-play payoff','100% flips edge case','transition probability conservation','payoff bounds','population normalization','equal-fitness neutral evolution','50% flips erase strategy differences']}

def main(output):
    output.mkdir(parents=True,exist_ok=True)
    test_report=tests(); results={'method':'Exact finite-state expected payoffs; deterministic infinite-population replicator dynamics','names':NAMES,'ids':IDS,'initial_population':'equal strategy proportions','generations':5000,'mutation_default':0,'forgiveness_probability':1/3,'noise':'independent per-player action flips; actual actions observed','tests':test_report,'scenarios':[]}
    curves={}
    for noise in [0,.01,.05]:
        matrices=payoff_matrices(noise)
        for rounds,matrix in matrices.items():
            for k in [4,6]:
                m=matrix[:k,:k]; h=evolve(m)
                record={'strategies':k,'rounds':rounds,'noise':noise,'payoff_matrix':m.tolist(),'equal_weight_tournament_totals':m.sum(1).tolist(),'population_0':h[0].tolist(),'population_50':h[50].tolist(),'population_100':h[100].tolist(),'population_500':h[500].tolist(),'population_5000':h[5000].tolist(),'max_change_last_100_generations':float(np.max(np.abs(h[-1]-h[-101]))),'trajectory_0_to_500':h[:501].tolist()}
                results['scenarios'].append(record)
                curves[(k,rounds,noise)]=h
                if rounds==200:
                    mut=evolve(m,mutation=.001)
                    record['mutation_0.001_population_500']=mut[500].tolist()
                    record['mutation_0.001_population_5000']=mut[-1].tolist()
    (output/'results.json').write_text(json.dumps(results,ensure_ascii=False,separators=(',',':')), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#1b9e77','#d95f02','#7570b3','#e7298a','#666666','#0072b2']
    for k in [4,6]:
        fig,axes=plt.subplots(3,1,figsize=(9,12),sharey=True)
        for ax,noise in zip(axes,[0,.01,.05]):
            h=curves[(k,200,noise)][:501]
            for i in range(k): ax.plot(np.arange(501),100*h[:,i],color=colors[i],label=NAMES[i],linewidth=2)
            ax.set(title=f'Action noise: {noise:.0%}',xlabel='Generation',ylim=(0,100),xlim=(0,500))
            ax.grid(alpha=.18)
        for ax in axes: ax.set_ylabel('Population share (%)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.035))
        fig.suptitle(f'{k} strategies evolving: 200 rounds per encounter',fontsize=16,y=.99)
        fig.text(.5,.005,'Equal initial shares | No mutation | Infinite well-mixed population\nExact expected payoffs | Fitness-proportional reproduction',ha='center',fontsize=10,color='#555555')
        fig.tight_layout(rect=[0,.10,1,.96]);fig.savefig(output/f'evolution_{k}_strategies.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,9),sharey=True)
    for ax,noise in zip(axes,[.01,.05]):
        h=curves[(4,200,noise)]
        for i in range(4): ax.plot(np.arange(5001),100*h[:,i],color=colors[i],label=NAMES[i],linewidth=1.6)
        ax.axvline(500,color='#999999',linestyle='--',linewidth=1)
        ax.set(title=f'Original four strategies | Action noise: {noise:.0%}',xlabel='Generation',ylabel='Population share (%)',ylim=(0,100),xlim=(0,5000))
        ax.grid(alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.035))
    fig.suptitle('Longer observation: generation 500 is not the final outcome',fontsize=16,y=.99)
    fig.text(.5,.005,'Dashed line: generation 500 | 200 rounds per encounter\nEqual initial shares | No mutation',ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=[0,.09,1,.96]); fig.savefig(output/'evolution_long_diagnostic.png',dpi=180); plt.close(fig)
    print(json.dumps({'tests':test_report,'main_results':[ {key:r[key] for key in ['strategies','rounds','noise','population_500','population_5000','mutation_0.001_population_5000']} for r in results['scenarios'] if r['rounds']==200]},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('results'));args=p.parse_args();main(args.output)
