# ===============================
# MTL Training - PHASE 3: Fine-tuning from Phase 2
# ===============================
import os
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import gc

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ===============================
# GPU + Memory Config
# ===============================
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gc.collect()
tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {len(gpus)} device(s)")

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision enabled")
print(f"TensorFlow: {tf.__version__}")

# ===============================
# 1. Load Phase 2 Best Model
# ===============================
print("\n" + "="*70)
print("📥 LOADING PHASE 2 BEST MODEL")
print("="*70)

phase2_dir = './mtl_phase2_results'
phase2_model_path = os.path.join(phase2_dir, 'phase2_best.h5')

model = load_model(phase2_model_path, compile=False)
print("✅ Phase 2 model loaded")

with open(os.path.join(phase2_dir, 'phase2_info.json'), 'r') as f:
    phase2_info = json.load(f)

num_classes_A = phase2_info['num_classes_A']
num_classes_B = phase2_info['num_classes_B']
BATCH_SIZE = phase2_info['batch_size']

# ===============================
# 2. Load Data
# ===============================
print("\n" + "="*70)
print("📊 LOADING DATA")
print("="*70)

csv_path = '/media/tohn/HDD/VISION_dataset/USAI/Data_CCA_6569.csv'
df0 = pd.read_csv(csv_path)

dataframe = df0[
    (df0['Spilt'] == 'Train') &
    (df0['Path Crop'] != 'None') &
    (df0['Path Crop'] != 'Nan')
].reset_index(drop=True)

# Load encoders
with open('./encoders/label_encoder_taskA.pkl', 'rb') as f:
    le_classA = pickle.load(f)

with open('./encoders/label_encoder_taskB.pkl', 'rb') as f:
    le_classB = pickle.load(f)

dataframe['TaskA_encoded'] = le_classA.transform(dataframe['Sub_class_New'])
dataframe['TaskB_encoded'] = le_classB.transform(dataframe['Sub_Position_Label'])

train_df, val_df = train_test_split(
    dataframe,
    test_size=0.15,
    random_state=42,
    stratify=dataframe['TaskA_encoded']
)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ===============================
# 3. Dataset
# ===============================
IMG_SIZE = 456

def load_and_preprocess_image(path, label_A, label_B, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)

    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img, (label_A, label_B)

def create_dataset(df, augment=False, shuffle=True):
    paths = df['Path Crop'].values
    labels_A = to_categorical(df['TaskA_encoded'], num_classes_A)
    labels_B = to_categorical(df['TaskB_encoded'], num_classes_B)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_A, labels_B))
    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.map(
        lambda p, a, b: load_and_preprocess_image(p, a, b, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_dataset = create_dataset(train_df, augment=True)
val_dataset = create_dataset(val_df, augment=False, shuffle=False)

# ===============================
# 4. PHASE 3: Unfreeze block6 + block7
# ===============================
print("\n" + "="*70)
print("🔥 PHASE 3: FINE-TUNE BLOCK6 + BLOCK7")
print("="*70)

total_unfrozen = 0
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = True
        for sub in layer.layers:
            sub.trainable = False

        for sub in layer.layers:
            if ('block6' in sub.name) or ('block7' in sub.name) or ('top_' in sub.name):
                sub.trainable = True
                total_unfrozen += 1

print(f"✅ Unfrozen layers: {total_unfrozen}")

# ===============================
# 5. Compile (Lower LR + Weighted Loss)
# ===============================
model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=1e-5,   # lower than phase 2
        momentum=0.9,
        nesterov=True,
        clipnorm=1.0
    ),
    loss={
        'task_subclass': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        'task_position': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    },
    loss_weights={
        'task_subclass': 3.0,   # ⭐ subclass สำคัญกว่า
        'task_position': 1.0
    },
    metrics={
        'task_subclass': ['accuracy'],
        'task_position': ['accuracy']
    }
)

print("✅ Compiled Phase 3")

# ===============================
# 6. Callbacks
# ===============================
save_dir = './mtl_phase3_results'
os.makedirs(save_dir, exist_ok=True)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'phase3_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.CSVLogger(
        os.path.join(save_dir, 'phase3_training.csv')
    )
]

# ===============================
# 7. Train
# ===============================
EPOCHS = 100

print("\n" + "="*70)
print("🏋️ TRAINING PHASE 3")
print("="*70)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ===============================
# 8. Save Final
# ===============================
model.save(os.path.join(save_dir, 'phase3_final.h5'))
print("✅ Saved phase3_final.h5")

print("\n🎉 PHASE 3 FINETUNE COMPLETE!")
