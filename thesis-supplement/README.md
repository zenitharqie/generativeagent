# Thesis Supplementary Materials

Supplementary materials for:

**Behavioural Consistency of Large Language Model-Based Generative Agents under Different Memory Retrieval Weight Configurations**

Author: Zinedine Zenith Arqie Fomalhaut  
Universitas Airlangga, 2026

## Contents

1. Experimental configurations
2. Simulator and evaluator prompts
3. Ground Truth construction data
4. Retrieval evaluation results
5. Behavioural consistency evaluation results
6. Validation summaries

## Experiment Summary

- 4 retrieval configurations
- 3 repetitions per configuration
- 12 runs
- 240 evaluation units
- Top-K: 5
- Intersection Ground Truth: 46 categories
- Union Ground Truth: 78 categories

## Important Notes

- Intersection Ground Truth is the primary evaluation reference.
- Union Ground Truth is used for sensitivity analysis.
- Behavioural consistency results are taken from BCR final v2.
- Ground Truth was constructed without human validation.
- The simulator and final evaluators used GPT-5.6-luna.
- The exact semantic-normalization prompt was not completely preserved.

## Frozen Result Fingerprints

Retrieval evaluation:

`b387d89ef2ce91fc1fe96d1bc00cfcb98878d12112614acba1c4b21c837e7a95`

Behavioural consistency evaluation:

`b21ade084d335d48fe9a01e6ed03235485f679b4ddc56a3a693bf6cea141b4ea`
