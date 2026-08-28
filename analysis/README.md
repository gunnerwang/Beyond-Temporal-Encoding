# Recorded search results

Each file is the outcome of one hyperparameter search: the operating point it selected and
the score it reached. The analysis scripts in `scripts/` read these back so that a
measurement holds every hyperparameter fixed at a recorded value and varies only the one
thing it is measuring.

```
analysis/<dataset>/full-pipeline.json            prior weights and online correction parameters for the
                                                 whole pipeline; read by paired_mechanism_ablation.py
analysis/<dataset>/embedding-adaptation.json     the search over the low-rank embedding adaptation;
                                                 read by adapter_effect_study.py
analysis/<dataset>/classification.json           tree-ensemble weights and kNN parameters for the
                                                 classification instantiation; read by delong_auc_ci.py
analysis/<dataset>/xgb-tuning.json               XGBoost hyperparameters for the leaf retrieval space;
                                                 read by build_leaf_embeddings.py
```

`xgb-tuning.json` ships for `homesite-insurance` and `homecredit-default`, and
`build_leaf_embeddings.py` picks it up automatically; elsewhere it uses library defaults
and names the source it used.

These are records of a search, not configurations to run. The configurations are in
`configs/`; see `docs/configurations.md`.
