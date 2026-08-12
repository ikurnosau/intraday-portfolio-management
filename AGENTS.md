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

# Model Package Versioning

When changing the persisted deployment format in `modeling/model_package.py`:

- Increment `MODEL_PACKAGE_FORMAT_VERSION` when the package layout, required
  files, `metadata.json` or `COMPLETED` schema, integrity rules, or package-wide
  loading semantics change incompatibly.
- Increment `CHECKPOINT_FORMAT_VERSION` when the structure, serialization, or
  interpretation of `model.pt` changes. Do not increment it merely because a
  new training run produces different model weights.
- Increment `ALLOCATOR_CONFIG_VERSION` when the fields, types, allocator
  identifier, or interpretation of `allocator_config.json` change. Do not
  increment it when tuning produces different parameter values.
- Increment only the version whose persisted contract changed. Do not increment
  these values for refactors, comments, formatting, or tests that preserve the
  stored representation and runtime interpretation.
- Package loaders must validate these versions and explicitly reject unsupported
  formats rather than silently guessing how to deserialize them.
