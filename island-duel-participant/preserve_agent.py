"""Preserve a lead by avoiding vulnerable moves and use energy for late captures."""
from closing_agent import choose_action as base
from sequence_agent import adjacency


def choose_action(s):
    me = s["seat"]
    them = 1-me
    turns = s["actions_remaining"]
    if turns > 65 or turns <= 2:
        return base(s)
    o, w, c = s["owners"], s["weights"], s["costs"]
    a = adjacency(s["size"])
    e, r = s["energy"][me], s["energy"][them]
    lead = s["scores"][me]-s["scores"][them]
    threats = sorted(((2*w[i], i) for i in range(len(o)) if o[i]==me and w[i]
                      and c[i]+2 <= r and any(o[j]==them for j in a[i])), reverse=True)
    best = (-threats[0][0] if threats else 0) + (min(3,9-e)*(2 if turns <= 8 else 1))
    chosen = None
    for i in s["legal_actions"]:
        if i is None:
            continue
        capture = o[i] == them
        cost = c[i]+2*capture
        loss = next((v for v,j in threats if i not in a[j] or
                     any(k != i and o[k]==them for k in a[j])),0)
        if c[i]+2 <= r and any(o[j]==them for j in a[i]):
            loss = max(loss,2*w[i])
        score = w[i]*(1+capture) - loss - (2.2 if lead > 0 else 1.3)*cost
        score += (2 if turns < 10 else .5)*capture
        score += .35*sum(o[j]==-1 for j in a[i])
        if score > best:
            best, chosen = score, i
    return chosen
