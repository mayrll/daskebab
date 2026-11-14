from functools import lru_cache


@lru_cache(None)
def adjacency(size):
    return tuple(tuple(j for j in (i-size,i+size,i-1,i+1)
                       if 0 <= j < size*size and abs(i//size-j//size)+abs(i%size-j%size)==1)
                 for i in range(size*size))


def choose_action(s):
    me = s['seat']
    them = 1-me
    o, w, c = s['owners'], s['weights'], s['costs']
    energy, rival = s['energy'][me], s['energy'][them]
    turns = s['actions_remaining']
    adj = adjacency(s['size'])
    hostile = {i for i, owner in enumerate(o) if owner == them}
    attacks = {i: 2*w[i] for i, owner in enumerate(o)
               if owner == me and w[i] and c[i]+2 <= rival
               and any(j in hostile for j in adj[i])}
    baseline = max(attacks.values(), default=0)
    best = (1.0 if energy <= 6 and turns > 2 else -1e9, None)
    for i in s['legal_actions']:
        if i is None:
            continue
        capture = o[i] == them
        cost = c[i] + 2*capture
        enemy_neighbors = [j for j in adj[i] if o[j] == them]
        threat = bool(enemy_neighbors)
        if turns <= 2:
            score = w[i]*(1+capture)
        else:
            score = w[i]*(1+capture) - (2.4 if turns > 180 else 3.4)*cost
            score += (3.1 if turns > 180 else .5)*sum(o[j] == -3 for j in adj[i])
            score += .25*sum(max(0,w[j]-c[j]) for j in adj[i] if o[j]==-1)
            if threat and not capture:
                score -= (2.05 if rival >= c[i]+2 else .65)*w[i]
            elif capture:
                score += 2.5
            if turns <= 12:
                score += .9*w[i]*capture
        if turns > 2:
            worst = max((value for target, value in attacks.items()
                         if i not in adj[target] or any(j != i and j in hostile for j in adj[target])),
                        default=0)
            if threat and rival >= c[i]+2:
                worst = max(worst, 2*w[i])
            score -= (worst-baseline) * (1 if turns < 18 else .72)
        if score > best[0]:
            best = score, i
    return best[1]





