from frugal_opening import choose_action as base, adjacency


def choose_action(state):
    me = state["seat"]
    them = 1 - me
    owners = state["owners"]
    weights = state["weights"]
    costs = state["costs"]
    adj = adjacency(state["size"])
    energy, enemy_energy = (state["energy"][i] for i in (me, them))
    turns = state["actions_remaining"]
    margin = state["scores"][me] - state["scores"][them]
    if turns > 36 or turns == 1:
        return base(state)
    options = []
    for tile in state["legal_actions"]:
        if tile is None:
            continue
        capture = owners[tile] == them
        cost = costs[tile] + 2 * capture
        counter = any(owners[j] == them for j in adj[tile])
        gain = weights[tile] * (1 + capture)
        targets = [
            weights[j] * 2 for j in range(len(owners))
            if owners[j] == me and weights[j] and costs[j] + 2 <= enemy_energy
            and any(k != tile and owners[k] == them for k in adj[j])
        ]
        if counter and costs[tile] + 2 <= enemy_energy:
            targets.append(2 * weights[tile])
        score = gain - max(targets, default=0)
        if turns > 2:
            score -= 1.5 * cost
            score += .15 * (energy - cost)
        options.append((score, tile))
    resting = -max((2 * weights[j] for j in range(len(owners))
                    if owners[j] == me and weights[j] and costs[j] + 2 <= enemy_energy
                    and any(owners[k] == them for k in adj[j])), default=0)
    if turns > 2:
        resting += 1.5 * min(3, 9 - energy)
    score, action = max(options, default=(-float("inf"), None))
    if score > resting or (turns == 2 and margin <= -resting and margin + score > 0):
        return action
    return None

