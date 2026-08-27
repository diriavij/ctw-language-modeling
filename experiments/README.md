# Experiments

The experiment directory separates executable code from generated artifacts:

```text
experiments/
  *.py       Experiment and plotting scripts
  results/   Machine-readable JSON outputs
  figures/   Generated PNG and PDF figures
```

Scripts use `results/` and `figures/` by default. Explicit `--out` paths are
still respected, which is useful for one-off runs outside the repository.

Start with the lightweight synthetic validation:

```bash
python experiments/validate_markov.py
```

The WikiText-2 and GPT-2 experiments require the optional experiment
dependencies described in the repository-level README.
