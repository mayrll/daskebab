"""Three-ply search over visible moves, with a static expansion policy."""
from sequence_agent import adjacency
from closing_agent import choose_action as opening


def choose_action(s):
    turns = s["actions_remaining"]
    if turns > 72:
        return opening(s)
    me = s["seat"]
    them = 1 - me
    owners = s["owners"].copy()
    values, costs = s["weights"], s["costs"]
    energy = s["energy"]
    links = adjacency(s["size"])
    frontier = [{i for i, v in enumerate(values) if v and owners[i] in (-1, 1-p)
                 and any(owners[j] == p for j in links[i])} for p in range(2)]
    price = 2.5 if turns > 15 else 1.8 if turns > 5 else 0.7
    limits = (12, 7, 1)

    def candidates(p, power, tiles, remaining, limit):
        actions = [(price * min(3, 9-power) * .18 if remaining > 1 else 0, None, min(9, power+3))]
        for i in tiles:
            capture = owners[i] == 1-p
            cost = costs[i] + 2*capture
            if cost <= power:
                rank = values[i] * (1 + capture)
                if remaining > 1:
                    rank -= price * cost
                    rank += 1.7 * capture
                actions.append((rank, i, power-cost))
        actions.sort(key=lambda x: x[0], reverse=True)
        return actions[:limit]

    def search(p, power, other_power, tiles, other_tiles, depth):
        remaining = turns - depth
        best = (-float("inf"), None)
        for rank, i, new_power in candidates(p, power, tiles, remaining, limits[depth]):
            if depth == 2 or remaining <= 1:
                if rank > best[0]:
                    best = rank, i
                continue
            if i is None:
                reply, _ = search(1-p, other_power, new_power, other_tiles, tiles, depth+1)
            else:
                previous = owners[i]
                owners[i] = p
                own = tiles - {i}
                enemy = other_tiles - {i}
                for j in links[i]:
                    if values[j] and owners[j] in (-1, 1-p):
                        own.add(j)
                    if previous == 1-p and j in enemy and not any(owners[k] == 1-p for k in links[j]):
                        enemy.discard(j)
                if any(owners[j] == 1-p for j in links[i]):
                    enemy.add(i)
                reply, _ = search(1-p, other_power, new_power, enemy, own, depth+1)
                owners[i] = previous
            score = rank - (1 if remaining <= 4 else .86) * reply
            if score > best[0]:
                best = score, i
        return best

    return search(me, energy[me], energy[them], frontier[me], frontier[them], 0)[1]

