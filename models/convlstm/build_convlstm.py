"""
models/convlstm/build_convlstm.py
===================================
STEP 2 — ConvLSTM encoder-decoder architecture factory.

Architecture (encoder-decoder with BatchNorm):
  Input  : (batch, T=12, H=35, W=45, 1)
  Encode : ConvLSTM2D(32) → BN → ConvLSTM2D(64) → BN → ConvLSTM2D(64) → BN
  Expand : Lambda reshape → (batch, 1, H, W, 64)
  Decode : ConvLSTM2D(64) → BN → ConvLSTM2D(32) → BN
  Output : TimeDistributed(Conv2D(1, kernel=1, sigmoid)) → squeeze to (batch, H, W, 1)

  padding='same' throughout — preserves 35×45 spatial dims, no power-of-2 padding needed.
  sigmoid output matches [0,1] normalised targets.
  Loss: MAE  |  Optimiser: Adam  |  Metric: RMSE (pixel-wise)

Usage:
    from models.convlstm.build_convlstm import build_convlstm, DEFAULTS
    model = build_convlstm()          # uses DEFAULTS
    model = build_convlstm(filters_enc=(16,32,32), lr=5e-4)
    model.summary()
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ─── Defaults (mirrors LSTM's DEFAULTS dict for consistency) ──────────────────
# Filter counts kept small (16,32,32 / 32,16) for CPU feasibility:
# ~150K params → ~2-3s/epoch vs ~14s/epoch with (32,64,64).
# Still well-specified for a 35×45 spatial field.
DEFAULTS = {
    "input_shape":  (12, 35, 45, 1),    # (T, H, W, C)
    "filters_enc":  (16, 32, 32),        # encoder ConvLSTM filter counts
    "filters_dec":  (32, 16),            # decoder ConvLSTM filter counts
    "kernel_size":  3,
    "dropout":      0.0,                 # SpatialDropout2D rate (0 = disabled at train time)
    "lr":           1e-3,
}


# ─── Custom RMSE metric ───────────────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="convlstm")
def pixel_rmse(y_true, y_pred):
    """Pixel-wise RMSE across the full spatial frame."""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# ─── Model factory ────────────────────────────────────────────────────────────

def build_convlstm(
    input_shape: tuple = DEFAULTS["input_shape"],
    filters_enc: tuple = DEFAULTS["filters_enc"],
    filters_dec: tuple = DEFAULTS["filters_dec"],
    kernel_size:  int  = DEFAULTS["kernel_size"],
    dropout:     float = DEFAULTS["dropout"],
    lr:          float = DEFAULTS["lr"],
) -> keras.Model:
    """
    Build and compile the ConvLSTM encoder-decoder.

    Parameters
    ----------
    input_shape  : (T, H, W, C)  — temporal depth, height, width, channels
    filters_enc  : tuple of 3 ints — filter counts for encoder ConvLSTM layers
    filters_dec  : tuple of 2 ints — filter counts for decoder ConvLSTM layers
    kernel_size  : spatial kernel size (same for all ConvLSTM layers)
    dropout      : SpatialDropout2D rate (applied after each BN in decoder);
                   set > 0 to enable MC Dropout at forecast time
    lr           : Adam learning rate

    Returns
    -------
    Compiled keras.Model
    """
    T, H, W, C = input_shape
    ks = (kernel_size, kernel_size)

    inputs = keras.Input(shape=input_shape, name="input_sequence")

    # ── Encoder ──────────────────────────────────────────────────────────────
    # Layer E1
    x = layers.ConvLSTM2D(
        filters=filters_enc[0], kernel_size=ks,
        padding="same", return_sequences=True,
        name="enc_convlstm_1",
    )(inputs)
    x = layers.BatchNormalization(name="enc_bn_1")(x)

    # Layer E2
    x = layers.ConvLSTM2D(
        filters=filters_enc[1], kernel_size=ks,
        padding="same", return_sequences=True,
        name="enc_convlstm_2",
    )(x)
    x = layers.BatchNormalization(name="enc_bn_2")(x)

    # Layer E3 — return_sequences=False → collapses time axis → (batch, H, W, F)
    x = layers.ConvLSTM2D(
        filters=filters_enc[2], kernel_size=ks,
        padding="same", return_sequences=False,
        name="enc_convlstm_3",
    )(x)
    x = layers.BatchNormalization(name="enc_bn_3")(x)

    # ── Bridge: expand time axis back to length 1 ─────────────────────────────
    # x shape: (batch, H, W, F)  →  (batch, 1, H, W, F)
    # Use Reshape (not Lambda) so the model serializes safely without Lambda
    x = layers.Reshape((1, H, W, filters_enc[2]), name="expand_time")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    # Layer D1
    x = layers.ConvLSTM2D(
        filters=filters_dec[0], kernel_size=ks,
        padding="same", return_sequences=True,
        name="dec_convlstm_1",
    )(x)
    x = layers.BatchNormalization(name="dec_bn_1")(x)
    if dropout > 0:
        # TimeDistributed SpatialDropout2D — drops entire feature maps spatially
        # Keeps training=True active at inference for MC Dropout uncertainty
        x = layers.TimeDistributed(
            layers.SpatialDropout2D(rate=dropout, name="spatial_dropout_1"),
            name="td_dropout_1",
        )(x)

    # Layer D2
    x = layers.ConvLSTM2D(
        filters=filters_dec[1], kernel_size=ks,
        padding="same", return_sequences=True,
        name="dec_convlstm_2",
    )(x)
    x = layers.BatchNormalization(name="dec_bn_2")(x)
    if dropout > 0:
        x = layers.TimeDistributed(
            layers.SpatialDropout2D(rate=dropout, name="spatial_dropout_2"),
            name="td_dropout_2",
        )(x)

    # ── Output head ──────────────────────────────────────────────────────────
    # TimeDistributed Conv2D maps to 1 channel, sigmoid → [0, 1]
    # Output shape: (batch, 1, H, W, 1)
    x = layers.TimeDistributed(
        layers.Conv2D(1, kernel_size=(1, 1), activation="sigmoid", padding="same",
                      name="output_conv"),
        name="td_output",
    )(x)

    # Squeeze the time axis → (batch, H, W, 1)  — matches y shape from prepare_frames
    # Use Reshape (not Lambda) for safe serialization
    outputs = layers.Reshape((H, W, 1), name="squeeze_time")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="ConvLSTM_NTL")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mae",
        metrics=[pixel_rmse],
    )
    return model


# ─── Quick verification ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    print("\n[ConvLSTM] Step 2 — build_convlstm.py\n")

    model = build_convlstm()
    model.summary()

    # Shape smoke-test: one batch of 4 sequences
    T, H, W, C = DEFAULTS["input_shape"]
    dummy_X = np.random.rand(4, T, H, W, C).astype(np.float32)
    dummy_y = model.predict(dummy_X, verbose=0)

    print(f"\n[INFO] Input  shape : {dummy_X.shape}")
    print(f"[INFO] Output shape : {dummy_y.shape}   (expected: (4, {H}, {W}, 1))")
    assert dummy_y.shape == (4, H, W, 1), \
        f"Shape mismatch: got {dummy_y.shape}, expected (4, {H}, {W}, 1)"

    # Also verify with dropout enabled
    model_mc = build_convlstm(dropout=0.2)
    dummy_y2 = model_mc.predict(dummy_X, verbose=0)
    assert dummy_y2.shape == (4, H, W, 1), "MC Dropout model shape mismatch"

    print(f"\n[INFO] Total trainable params : "
          f"{model.count_params():,}")
    print("\n[ConvLSTM] Step 2 complete — architecture verified.")
    print("[ConvLSTM] Next: python models/convlstm/train_convlstm.py\n")
