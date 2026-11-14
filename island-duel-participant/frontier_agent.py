"""Fog-of-war policy for Island Duel."""


def neighbors(cell, size):
    row, col = divmod(cell, size)
    return [
        index for valid, index in (
            (row > 0, cell - size),
            (row + 1 < size, cell + size),
            (col > 0, cell - 1),
            (col + 1 < size, cell + 1),
        ) if valid
    ]


def choose_action(state):
    seat = state["seat"]
    enemy = 1 - seat
    energy = state["energy"][seat]
    enemy_energy = state["energy"][enemy]
    owners, weights, costs = (state[key] for key in ("owners", "weights", "costs"))
    size = state["size"]
    remaining = state["actions_remaining"]
    best = None
    best_score = 0.8 if energy <= 6 and remaining > 2 else float("-inf")
    for cell in state["legal_actions"]:
        if cell is None:
            continue
        capture = owners[cell] == enemy
        cost = costs[cell] + 2 * capture
        adjacent = neighbors(cell, size)
        exposed = any(owners[i] == enemy for i in adjacent)
        score = weights[cell] * (1 + capture)
        if remaining > 2:
            score -= 1.15 * cost
            score += 2.4 * sum(owners[i] == -3 for i in adjacent)
            score += 0.16 * sum(max(0, weights[i] - costs[i]) for i in adjacent if owners[i] == -1)
            if exposed and enemy_energy >= costs[cell] + 2:
                score -= 1.25 * weights[cell]
            if capture:
                score += 1.4 * sum(owners[i] == enemy for i in adjacent)
        if score > best_score:
            best_score, best = score, cell
    return best

