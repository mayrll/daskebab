"""Select captures that force the opponent to spend more turns recovering."""
from initiative_agent import choose_action as finish
from sequence_agent import adjacency


def choose_action(s):
    t=s['actions_remaining']
    if t <= 32:
        return finish(s)
    p=s['seat']
    q=1-p
    o,w,c=s['owners'],s['weights'],s['costs']
    e,r=s['energy'][p],s['energy'][q]
    adj=adjacency(s['size'])
    en=[sum(o[j]==q for j in ns) for ns in adj]
    ours=[sum(o[j]==p for j in ns) for ns in adj]
    base=0
    danger=[]
    for i in range(len(o)):
        if o[i]==p and w[i] and en[i] and c[i]+2<=r:
            danger.append((2*w[i],i))
            base=max(base,2*w[i])
    limit = 6 if t > 100 else 7
    best=(.9 if e<=limit else -4,None)
    for i in s['legal_actions']:
        if i is None:
            continue
        cap=o[i]==q
        spend=c[i]+2*cap
        score=w[i]*(1+cap)-2.5*spend
        reward=0
        for j in adj[i]:
            if o[j]==-3 and not ours[j]:
                reward+=3.5 if t>165 else .85
            if o[j]==q and w[j] and not ours[j]:
                reward+=max(0,2*w[j]-2*(c[j]+2))*.52
            if o[j]==-1 and not ours[j]:
                reward+=max(0,w[j]-1.6*c[j])*.48
        score+=reward
        risk=max((v for v,j in danger if not cap or i not in adj[j] or en[j]>1),default=0)
        if en[i] and r>=c[i]+2:
            risk=max(risk,2*w[i])
            score -= .45*w[i] if not cap else .05*w[i]
            if cap and r-(c[i]+2)<3 and e-spend>=3:
                score+=2.4
        elif cap:
            score+=3.8
        score-=.9*(risk-base)
        if score>best[0]:
            best=score,i
    return best[1]
