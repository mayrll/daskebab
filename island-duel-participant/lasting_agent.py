"""Prioritize lasting territory and maximize the final score swing."""
from access_agent import choose_action as opening
from sequence_agent import adjacency


def choose_action(s):
    turns = s['actions_remaining']
    if turns > 65:
        return opening(s)
    me = s['seat']
    them = 1-me
    owners, weights, costs = (s[k] for k in ('owners','weights','costs'))
    energy, rival = (s['energy'][p] for p in (me, them))
    links = adjacency(s['size'])
    threats = [(2*weights[i], i) for i, o in enumerate(owners)
               if o==me and weights[i] and costs[i]+2 <= rival
               and any(owners[j]==them for j in links[i])]
    baseline = max((v for v, _ in threats), default=0) if turns > 1 else 0
    price = 2.5 if turns > 8 else 1.45 if turns > 2 else 0
    best = (min(3,9-energy) * price * .35 - baseline, None)
    for tile in s['legal_actions']:
        if tile is None:
            continue
        capture = owners[tile] == them
        cost = costs[tile] + 2*capture
        score = weights[tile]*(1+capture)
        if turns > 1:
            response = max((v for v,j in threats if tile not in links[j]
                           or any(owners[k] == them and k != tile for k in links[j])), default=0)
            exposed = any(owners[j]==them for j in links[tile])
            if exposed and costs[tile]+2 <= rival:
                response = max(response,2*weights[tile])
            if turns > 2:
                score -= price*cost
                score += 2.0*capture
                if not exposed and capture:
                    score += 1.8
            score -= response
        if score > best[0]:
            best = score, tile
    return best[1]
