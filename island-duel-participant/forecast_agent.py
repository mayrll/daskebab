"""Forecast the opponent's visible moves and value connected expansion."""
from sequence_agent import adjacency


def choose_action(s):
    me = s['seat']
    them = 1 - me
    energy, rival = (s['energy'][p] for p in (me, them))
    owners, values, costs = (s[k] for k in ('owners', 'weights', 'costs'))
    turns = s['actions_remaining']
    links = adjacency(s['size'])
    friendly = [sum(owners[j] == me for j in ns) for ns in links]
    hostile = [sum(owners[j] == them for j in ns) for ns in links]
    my_targets = [i for i, o in enumerate(owners) if o in (-1, them) and friendly[i] and values[i]]
    their_targets = [i for i, o in enumerate(owners) if o in (-1, me) and hostile[i] and values[i]]
    early = turns > 182
    price = 2.6 if early else 3.35
    if turns < 20:
        price = 1.9
    if turns <= 2:
        price = 0
    answers = sorted(
        [(values[i] * (1 + (owners[i] == me)), i) for i in their_targets
         if costs[i] + 2 * (owners[i] == me) <= rival], reverse=True)
    best = (-float('inf'), None)
    for i in s['legal_actions']:
        capture = i is not None and owners[i] == them
        cost = 0 if i is None else costs[i] + 2 * capture
        next_energy = min(9, energy + 3) if i is None else energy - cost
        score = price * (next_energy - energy) if i is None else values[i] * (1 + capture) - price * cost
        if i is not None and turns > 2:
            if capture:
                score += 2.8
            for j in links[i]:
                if owners[j] == -3:
                    score += 2.1 if early else .35
                elif owners[j] == -1 and not friendly[j]:
                    score += .45 * max(0, values[j] - 1.8 * costs[j])
                elif owners[j] == them and not friendly[j] and values[j]:
                    score += .30 * max(0, 2 * values[j] - 2.4 * (costs[j] + 2))
        if turns > 1:
            reply = next((gain for gain, j in answers if j != i and
                          (not capture or i not in links[j] or hostile[j] > 1)), 0)
            if i is not None and hostile[i] and costs[i] + 2 <= rival:
                reply = max(reply, 2 * values[i])
            score -= (1 if turns < 20 else .8) * reply
        if turns > 2:
            follow = max((values[j] * (1 + (owners[j] == them)) - price * (costs[j] + 2 * (owners[j] == them))
                          for j in my_targets if j != i and costs[j] + 2 * (owners[j] == them) <= next_energy), default=0)
            score += .45 * max(0, follow)
        if score > best[0]:
            best = score, i
    return best[1]
