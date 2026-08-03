# artifacts/trained_models/

Trained model weights produced by `src/models/train.py` (Module D). Kept
separate from `src/` deliberately: generated output, not hand-written code.

Regenerate with:

```
python -m src.models.train
```

## `ddi_gnn.pt`

A checkpoint dictionary containing:

| Key | Meaning |
|---|---|
| `state_dict` | Trained weights |
| `model_config` | Architecture settings used, so the model rebuilds identically |
| `in_channels` | Number of node features (12 molecular descriptors) |
| `best_val_auc` | Validation AUC of the saved epoch |
| `best_epoch` | Epoch the weights were taken from |

The checkpoint stores its own `model_config`, so reloading never depends on
`config/config.yaml` still holding the settings used at training time.

Training keeps the weights from the best validation AUC (not the final
epoch) and stops early once validation stops improving, so the saved model
is the best observed rather than the most overfit.
