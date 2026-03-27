"""
final_src — Combined TropiCycloneNet + Track Forecasting Library
================================================================
Combines Arshad's TropiCycloneNet codebase (intensity/direction classification,
DG methods, basin metrics) with the track forecasting additions (sliding-window
LSTM seq2seq, ADE/FDE evaluation).

Structure
---------
  dataset/
    dataset.py          ← Arshad's TCNDDataset (per-timestep, all 3 modalities)
    track_dataset.py    ← TrackForecastDataset (sliding-window, track + env)

  models/
    backbone.py         ← Arshad's TropiCycloneModel (SpatialEnc + TaskHeads)
    track_seq2seq.py    ← TrackSeq2Seq (LSTM encoder-decoder + attention)

  methods/
    dg_methods.py       ← Arshad's DG: ERM, IRM, VREx, CORAL, DANN, MAML, PhysIRM
    track_dg.py         ← Track DG: same API, MSE loss instead of CE

  metrics/
    basin_metrics.py    ← Arshad's BasinEvaluator, BTG, BNTE, RI-F1
    track_metrics.py    ← TrackEvaluator, ADE/FDE, haversine

  configs/
    ablations.py        ← Arshad's method hparams + ablation configs
    track_hparams.py    ← Track-specific hparams, LOBO ablations

  scripts/
    train_track.py      ← LOBO training script for track forecasting

  train.py              ← Arshad's LOBO training script (classification task)
"""
