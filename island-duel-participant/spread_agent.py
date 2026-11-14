from closing_agent import choose_action as closing
from sequence_agent import adjacency


def choose_action(s):
    turns = s['actions_remaining']
    if turns <= 45:
        return closing(s)
    me = s['seat']
    them = 1-me
    owners, weights, costs = (s[k] for k in ('owners','weights','costs'))
    energy, enemy_energy = (s['energy'][p] for p in (me, them))
    adj = adjacency(s['size'])
    friendly = [sum(owners[j] == me for j in ns) for ns in adj]
    enemy = [sum(owners[j] == them for j in ns) for ns in adj]
    pressure = [2 * weights[i] for i in range(len(owners))
                if owners[i] == me and weights[i] and enemy[i]
                and costs[i]+2 <= enemy_energy]
    baseline = max(pressure, default=0)
    best = (2.8 if energy <= 6 else -2.5, None)
    for i in s['legal_actions']:
        if i is None:
            continue
        capture = owners[i] == them
        spend = costs[i] + 2*capture
        score = weights[i]*(1+capture) - 1.55*spend + .7*capture
        new_targets = []
        for j in adj[i]:
            if friendly[j]:
                continue
            owner = owners[j]
            if owner == -3:
                new_targets.append(3.7 if turns > 200 else 1.25)
            elif owner in (-1, them) and weights[j]:
                gain = weights[j]*(1+(owner == them))
                new_targets.append(max(0, gain-1.7*(costs[j]+2*(owner==them))))
        score += 1.25*max(new_targets, default=0) + .55*sum(new_targets)
        if enemy[i]:
            if costs[i]+2 <= enemy_energy:
                score -= .9*max(0, 2*weights[i]-baseline) + .32*weights[i]
            elif not capture:
                score -= .55*weights[i]
        elif capture:
            score += 2.4
        if capture:
            score += .8*sum(weights[j] for j in adj[i]
                            if owners[j]==me and enemy[j]==1 and weights[j])
        if energy-spend <= 2:
            score -= .35
        if score > best[0]:
            best = score, i
    return best[1]


