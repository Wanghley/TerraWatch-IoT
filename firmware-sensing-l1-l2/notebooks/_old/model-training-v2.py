# train_and_convert_tf.py
import os, json, argparse, sys
import numpy as np
import tensorflow as tf

# -----------------------
# CONFIG
# -----------------------
MAX_SEQ_LEN = 64   # fixed sequence length for export/deployment (pick >= your typical length)
BATCH = 8
EPOCHS = 20
REPRESENTATIVE_SAMPLES = 200  # for quantization
MODEL_OUTPUT = "model_int8.tflite"
STATS_JSON = "norm_stats.json"

THERM_SHAPE = (8,8,3)   # per-frame thermal (H,W,channels)
RADAR_DIM = 12

# -----------------------
# Helpers: read dataset and compute stats
# -----------------------
def read_session_jsonl(folder):
    frames = []
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jsonl")])
    for fname in files:
        with open(os.path.join(folder, fname),'r') as fh:
            for line in fh:
                data = json.loads(line)
                left  = np.array(data["thermal"]["left"]).reshape(8,8).astype(np.float32)
                center= np.array(data["thermal"]["center"]).reshape(8,8).astype(np.float32)
                right = np.array(data["thermal"]["right"]).reshape(8,8).astype(np.float32)
                thermal = np.stack([left, center, right], axis=-1)  # H,W,C
                r1 = data["mmWave"]["R1"]; r2 = data["mmWave"]["R2"]
                radar = np.array([
                    r1["numTargets"], r1["range"], r1["speed"], r1["energy"], float(r1["valid"]),
                    r2["numTargets"], r2["range"], r2["speed"], r2["energy"], float(r2["valid"])
                ], dtype=np.float32)
                mic = np.array([data["mic"]["left"], data["mic"]["right"]], dtype=np.float32)
                radar_mic = np.concatenate([radar, mic])  # (12,)
                frames.append((thermal, radar_mic))
    return frames

def collect_all_sessions(root_dir):
    sessions = []
    labels = []
    for sub in sorted(os.listdir(root_dir)):
        full = os.path.join(root_dir, sub)
        if not os.path.isdir(full): continue
        label = 1 if 'animal' in sub.lower() else 0
        frames = read_session_jsonl(full)
        if frames:
            sessions.append(frames)
            labels.append(label)
    return sessions, labels

def compute_global_stats(sessions):
    # compute mean/std across ALL thermal pixels and all radar features
    thermals = []
    radars = []
    for sess in sessions:
        for t, r in sess:
            thermals.append(t.reshape(-1))  # flatten HWC
            radars.append(r)
    thermals = np.concatenate(thermals).astype(np.float32)
    radars = np.concatenate(radars).astype(np.float32)
    t_mean = float(np.mean(thermals)); t_std = float(np.std(thermals) + 1e-6)
    r_mean = float(np.mean(radars));  r_std = float(np.std(radars) + 1e-6)
    return {"thermal_mean": t_mean, "thermal_std": t_std, "radar_mean": r_mean, "radar_std": r_std}

# -----------------------
# Build TF datasets (padding variable sequences to MAX_SEQ_LEN)
# -----------------------
def pad_or_truncate_session(sess, stats, max_len=MAX_SEQ_LEN):
    """sess: list of (thermal (H,W,C), radar (12,))
       returns arrays: thermals (max_len, H,W,C), radars (max_len,12), length
    """
    L = len(sess)
    therm_p = np.zeros((max_len,)+THERM_SHAPE, dtype=np.float32)
    radar_p  = np.zeros((max_len, RADAR_DIM), dtype=np.float32)
    # normalize using precomputed stats
    t_mean, t_std = stats["thermal_mean"], stats["thermal_std"]
    r_mean, r_std = stats["radar_mean"], stats["radar_std"]
    for i in range(min(L, max_len)):
        t, r = sess[i]
        therm_p[i] = (t - t_mean) / t_std
        radar_p[i]  = (r - r_mean) / r_std
    if L < max_len:
        # pad by repeating last valid frame (helps models)
        if L > 0:
            therm_p[L:] = therm_p[L-1]
            radar_p[L:] = radar_p[L-1]
    return therm_p, radar_p, min(L, max_len)

def make_tf_dataset(sessions, labels, stats, max_len=MAX_SEQ_LEN, batch=BATCH, shuffle=True):
    X1=[]; X2=[]; Y=[]; L=[]
    for sess, lab in zip(sessions, labels):
        t_p, r_p, length = pad_or_truncate_session(sess, stats, max_len)
        X1.append(t_p)  # (max_len,H,W,C)
        X2.append(r_p)
        Y.append(lab)
        L.append(length)
    X1 = np.array(X1, dtype=np.float32)
    X2 = np.array(X2, dtype=np.float32)
    Y  = np.array(Y, dtype=np.int64)
    L  = np.array(L, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices(((X1, X2, L), Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(2048, len(Y)))
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------
# Keras model: per-frame small CNN -> per-frame features -> temporal masked avg pooling -> classification
# -----------------------
def make_model(max_seq=MAX_SEQ_LEN):
    # Inputs: thermals (B, max_seq, H,W,C), radars (B, max_seq, 12), lengths (B,)
    t_in = tf.keras.Input(shape=(max_seq,)+THERM_SHAPE, name='thermal')
    r_in = tf.keras.Input(shape=(max_seq, RADAR_DIM), name='radar')
    len_in = tf.keras.Input(shape=(), dtype=tf.int32, name='length')

    # process frames: use TimeDistributed conv
    x = t_in
    # small frame encoder
    td = tf.keras.layers.TimeDistributed
    x = td(tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu'))(x)
    x = td(tf.keras.layers.MaxPool2D(2))(x)  # 4x4
    x = td(tf.keras.layers.Conv2D(12, 3, padding='same', activation='relu'))(x)
    x = td(tf.keras.layers.Flatten())(x)
    frame_feats = td(tf.keras.layers.Dense(32, activation='relu'))(x)  # (B, seq, 32)

    # radar projector
    radar_feats = td(tf.keras.layers.Dense(16, activation='relu'))(r_in)  # (B, seq, 16)

    fused = tf.keras.layers.Concatenate(axis=-1)([frame_feats, radar_feats])  # (B, seq, 48)

    # masked temporal average pooling
    # build mask from lengths
    seq_len = tf.shape(fused)[1]
    # mask shape (B, seq_len)
    rng = tf.range(start=0, limit=seq_len, dtype=tf.int32)
    mask = tf.cast(tf.less(rng[None, :], len_in[:, None]), tf.float32)  # (B, seq_len)
    mask = tf.expand_dims(mask, axis=-1)  # (B, seq_len,1)
    fused_masked = fused * mask
    sum_feats = tf.reduce_sum(fused_masked, axis=1)  # (B, feat)
    lengths_f = tf.cast(tf.maximum(len_in, 1), tf.float32)[:, None]
    avg_feats = sum_feats / lengths_f

    x = tf.keras.layers.Dense(24, activation='relu')(avg_feats)
    out = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)

    model = tf.keras.Model(inputs=[t_in, r_in, len_in], outputs=out)
    return model

# -----------------------
# Representative dataset generator for TFLite conversion
# -----------------------
def rep_data_gen_from_arrays(X1, X2, L, max_samples=REPRESENTATIVE_SAMPLES):
    # yield list of inputs as float32 arrays
    N = X1.shape[0]
    ns = min(N, max_samples)
    for i in range(ns):
        t = X1[i:i+1].astype(np.float32)
        r = X2[i:i+1].astype(np.float32)
        l = L[i:i+1].astype(np.int32)
        # TFLite converter expects list of input arrays in same order as model inputs
        yield [t, r, l]

# -----------------------
# Main training & conversion
# -----------------------
def main(args):
    # 1) load sessions
    sessions, labels = collect_all_sessions(args.data)
    if len(sessions) == 0:
        print("No sessions found in", args.data); sys.exit(1)
    # 2) compute stats and save
    stats = compute_global_stats(sessions)
    with open(STATS_JSON, "w") as fh:
        json.dump(stats, fh, indent=2)
    print("Saved normalization stats to", STATS_JSON, stats)

    # 3) train/val split
    N = len(sessions)
    idx = np.random.permutation(N)
    split = int(0.8 * N)
    train_idx, val_idx = idx[:split], idx[split:]
    train_sessions = [sessions[i] for i in train_idx]
    train_labels   = [labels[i] for i in train_idx]
    val_sessions = [sessions[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    train_ds = make_tf_dataset(train_sessions, train_labels, stats, max_len=MAX_SEQ_LEN, batch=args.batch, shuffle=True)
    val_ds   = make_tf_dataset(val_sessions, val_labels, stats, max_len=MAX_SEQ_LEN, batch=args.batch, shuffle=False)

    # 4) model
    model = make_model(max_seq=MAX_SEQ_LEN)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # optional callbacks
    cb = [tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
          tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)]
    model.fit(train_ds, epochs=args.epochs, validation_data=val_ds, callbacks=cb)

    # save Keras model (SavedModel) - optional
    model.save("saved_model_tf")

    # 5) Convert to int8 TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Force full integer
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Prepare representative arrays (raw numpy used in make_tf_dataset)
    # Build arrays for rep data from train set
    X1=[]; X2=[]; L=[]
    for sess,lab in zip(train_sessions, train_labels):
        t_p, r_p, length = pad_or_truncate_session(sess, stats, MAX_SEQ_LEN)
        X1.append(t_p); X2.append(r_p); L.append(length)
    X1 = np.array(X1, dtype=np.float32); X2 = np.array(X2, dtype=np.float32); L = np.array(L, dtype=np.int32)

    def representative_gen():
        for t, r, l in rep_data_gen_from_arrays(X1, X2, L):
            # TFLite rep gen yields list of input arrays matching model signature
            yield [t, r, l]

    converter.representative_dataset = representative_gen

    tflite_model = converter.convert()
    with open(MODEL_OUTPUT, "wb") as f:
        f.write(tflite_model)
    print("Saved TFLite model:", MODEL_OUTPUT)
    # print input quantization params for device use (we can probe interpreter)
    interpreter = tf.lite.Interpreter(model_path=MODEL_OUTPUT)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("Input details:")
    for d in input_details:
        print(d["name"], "shape", d["shape"], "dtype", d["dtype"], "quant", d["quantization"])
    print("Output details:")
    for d in output_details:
        print(d["name"], "shape", d["shape"], "dtype", d["dtype"], "quant", d["quantization"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="root dataset directory")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    main(args)
