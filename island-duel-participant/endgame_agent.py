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
    best = (0.4 if energy <= 6 and turns > 2 else -1e9, None)
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
            score = w[i]*(1+capture) - 2.7*cost
            score += 0.6*sum(o[j] == -3 for j in adj[i])
            score += .40*sum(max(0,w[j]-c[j]) for j in adj[i] if o[j]==-1)
            if threat and not capture:
                score -= (2.05 if rival >= c[i]+2 else .65)*w[i]
            elif capture:
                score += 3.0
            if turns <= 16:
                score += (2.6 if turns <= 4 else 1.15)*w[i]*capture - (4 if threat and rival >= c[i]+2 and turns > 2 else 0)
        if score > best[0]:
            best = score, i
    return best[1]




