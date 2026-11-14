# Island Duel flag recovery

Recovered flag **SIGMOID_OCEANICWAYFINDING**.

The chart was extracted from the supplied Linux engine as a JSON string at offset 0x109eb8. This is a direct local-data recovery. The agent has not qualified. All 36 completed evaluation attempts are preserved, and the best result was 247 wins in 512 games. No qualifying run or platform acceptance is claimed.

## Reproduce

Use Python 3.10+ and extract the original participant bundle into this directory. No third-party Python packages are needed.

Run `python recover_flag.py` to extract and decode the chart.

The five anchor stars are read in Greek-letter order for the Southern Cross. Alpha is the bottom anchor, beta the left, gamma the top, delta the right, and epsilon the smaller inner anchor. Assign each small companion to its nearest anchor and read each group from nearest to farthest, beginning with the anchor itself.

The resulting ciphertext is `TCEVNMKHEYYCNVIZU`.

The verse's foundation clue refers to Samoa's motto, Faavae i le Atua Samoa. Normalize it to `FAAVAEILEATUASAMOA` and apply Vigenere decryption with A = 0. The plaintext is `OCEANICWAYFINDING`. Prefix it with `SIGMOID_`.

The [Public Seal and Crest of Samoa Act](https://www.ag.gov.ws/wp-content/uploads/2024/02/Public-Seal-and-Crest-of-Samoa-Act-1973.pdf) provides the motto and its translation. The [astronomical placement of the five stars](https://www.academia.edu/99141025/A_hierarchy_of_symbols_Samoan_religious_symbolism_in_New_Zealand) supports the Greek-letter reading order.

## Evidence

- [Recovery script](recover_flag.py)
- [Extracted chart JSON](recovered-chart.json)
- [Chart rendering](recovered-chart.html)
- [All evaluation commands, agents and results](explanation.html)
- [Evaluation records and replays](evidence)

The archive supplied by the user had SHA256 a89919a90b1507e2336b34a34cd43a8d7cf178006b3d630ae58741c591bb7140, differing from the archive hash stated in the challenge.

