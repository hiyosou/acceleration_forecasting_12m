# acceleration_forecasting_12m

100 m区間の上下加速度絶対値最大値を、過去5か月と現在値から翌月以降12か月生成する独立実装です。

- 入力採用条件: 現在値必須、過去5か月中3か月以上有効
- ガイド: 既存256次元波形embeddingから異なる3日を検索
- 生成: Softmaxガイド基準に対するAttentionなしResidual Diffusion
- 出力: DDIMで12か月系列を100本生成し、中央値・p10・p90を保存

既存の`acceleration_retrieval/artifacts_dataset_split`を読み取り専用で参照します。

```powershell
uv sync --extra dev
uv run python -m acceleration_forecasting_12m.cli --help
```

## 実行順

```powershell
uv run python -m acceleration_forecasting_12m.cli build-retrieval
uv run python -m acceleration_forecasting_12m.cli prepare-datasets --device cuda
uv run python -m acceleration_forecasting_12m.cli train --device cuda
uv run python -m acceleration_forecasting_12m.cli validate --device cuda
uv run python -m acceleration_forecasting_12m.cli predict --num-samples 100 --device cuda
uv run python -m acceleration_forecasting_12m.cli evaluate --bootstrap 1000 --plot
```

Cross-Attention付き絶対値拡散版は既存Residual版と別成果物で実行します。

```powershell
uv run python -m acceleration_forecasting_12m.cli prepare-absolute --device cuda
uv run python -m acceleration_forecasting_12m.cli train --dataset-dir artifacts/datasets_absolute_attention --output-dir artifacts/models_absolute_attention/unet --device cuda --no-resume
uv run python -m acceleration_forecasting_12m.cli select-sampling --device cuda
uv run python -m acceleration_forecasting_12m.cli predict --dataset-dir artifacts/datasets_absolute_attention --checkpoint artifacts/models_absolute_attention/unet/best_model.pt --output-dir artifacts/predictions_absolute_attention --device cuda --num-samples 100 --sampling-steps 100 --initial-noise-scale 1.0
uv run python -m acceleration_forecasting_12m.cli evaluate --dataset-dir artifacts/datasets_absolute_attention --prediction-dir artifacts/predictions_absolute_attention --output-dir artifacts/evaluation_absolute_attention --bootstrap 1000 --plot --plot-max-targets 195
```

成果物は`artifacts/`以下へ保存され、Git管理対象には含まれません。各コマンドは進捗をstderr、最終結果JSONをstdoutへ出力します。`--no-progress`で進捗表示を無効化できます。
