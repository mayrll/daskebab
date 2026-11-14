"""Use one-turn energy initiative to time contested captures."""
from sequence_agent import adjacency


def choose_action(s):
    me = s['seat']
    foe = 1-me
    energy, reserve = s['energy'][me], s['energy'][foe]
    own, val, cost = s['owners'], s['weights'], s['costs']
    turns = s['actions_remaining']
    adj = adjacency(s['size'])
    enemy_count = [sum(own[j] == foe for j in near) for near in adj]
    threatened = sorted(((2*val[i], i) for i in range(len(own))
                         if own[i] == me and val[i] and enemy_count[i] and cost[i]+2 <= reserve),
                        reverse=True)
    current_risk = threatened[0][0] if threatened else 0
    best = ((1.3 if energy <= 6 else -5) - current_risk, None)
    if turns <= 2:
        best = (0 if turns == 1 else -current_risk, None)
    for i in s['legal_actions']:
        if i is None:
            continue
        capture = own[i] == foe
        expense = cost[i] + 2*capture
        score = val[i]*(1+capture)
        if turns == 1:
            if score > best[0]:
                best = score, i
            continue
        reply = next((v for v,j in threatened
                      if not capture or i not in adj[j] or enemy_count[j]>1), 0)
        if enemy_count[i] and cost[i]+2 <= reserve:
            reply = max(reply, 2*val[i])
        if turns > 2:
            score -= (2.45 if turns > 175 else 2.75 if turns > 25 else 1.25)*expense
            score += (2.65 if turns > 175 else .5)*sum(own[j] == -3 for j in adj[i])
            if capture and reserve < cost[i]+2:
                score += 3.0
            if capture and not enemy_count[i]:
                score += 2.0
            if capture and reserve >= cost[i]+2 and enemy_count[i]:
                score += 1.9*(cost[i]+2) - 2
        score -= reply * (1 if turns <= 10 else .88)
        if score > best[0]:
            best = score,i
    return best[1]
