# Configuration Versioning

When changing the training configuration system:

- Increment `SCHEMA_VERSION` in `config/train_config.py` and
  `schema_version` in `train_config.yaml` when the YAML structure, required
  fields, inheritance behavior, or interpretation of existing fields changes.
- Increment `CONFIG_REVISION` in `config/train_config.py` and
  `config_revision` in `train_config.yaml` when behavior changes in any
  registered feature, statistic, target, normalizer, missing-value handler,
  retriever, model, loss, metric, optimizer, or component factory.
- Do not increment either value for refactors, comments, formatting, or tests
  that preserve configuration structure and runtime behavior.
- Keep the code constant and root YAML value equal. Experiment YAML files
  inherit these values unless they explicitly override them.
