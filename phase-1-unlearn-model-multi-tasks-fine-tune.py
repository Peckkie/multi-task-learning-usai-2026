# ===============================
# MTL Training - PHASE 1 ONLY
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
from sklearn.preprocessing import LabelEncoder

# ===============================
# GPU + Memory Config
# ===============================
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gc.collect()
tf.keras.backend.clear_session()

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU: {len(gpus)} device(s)")

# Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed Precision enabled")

print(f"TensorFlow: {tf.__version__}")

# ===============================
# 1. Load Model
# ===============================
print("\n" + "="*70)
print("📥 LOADING MODEL")
print("="*70)

model = load_model('./mtl_model_new/mtl_efficientnet_b5.h5', compile=False)
print("✅ Model loaded")

# ===============================
# 2. Load Data
# ===============================
print("\n" + "="*70)
print("📊 LOADING DATA")
print("="*70)

train_csv = '/media/tohn/HDD/VISION_dataset/USAI/Data_CCA_6569.csv'
df0 = pd.read_csv(train_csv)

dataframe = df0[(df0['Path Crop'] != 'None') & (df0['Path Crop'] != 'Nan')]
dataframe = df0[df0['Spilt'] == 'Train']
dataframe = dataframe.reset_index(drop=True)

print(f"Total: {len(dataframe)}")

# Encoding
le_classA = LabelEncoder()
dataframe['TaskA_encoded'] = le_classA.fit_transform(dataframe['Sub_class_New'])
num_classes_A = len(le_classA.classes_)

le_classB = LabelEncoder()
dataframe['TaskB_encoded'] = le_classB.fit_transform(dataframe['Sub_Position_Label'])
num_classes_B = len(le_classB.classes_)

print(f"Task A: {num_classes_A} classes")
print(f"Task B: {num_classes_B} classes")

# Save encoders
os.makedirs('./encoders', exist_ok=True)
with open('./encoders/label_encoder_taskA.pkl', 'wb') as f:
    pickle.dump(le_classA, f)
with open('./encoders/label_encoder_taskB.pkl', 'wb') as f:
    pickle.dump(le_classB, f)
print("✅ Encoders saved")

# Split
train_df, val_df = train_test_split(
    dataframe, 
    test_size=0.15, 
    random_state=42,
    stratify=dataframe['TaskA_encoded']
)

print(f"Train: {len(train_df)} | Val: {len(val_df)}")

# ===============================
# 3. Create Dataset
# ===============================
print("\n" + "="*70)
print("📦 CREATING DATASETS")
print("="*70)

def load_and_preprocess_image(path, label_A, label_B, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [456, 456])
    
    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
    
    img = tf.clip_by_value(img, 0, 255)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    
    return img, (label_A, label_B)

def create_dataset(dataframe, num_classes_A, num_classes_B, 
                   batch_size=4, augment=False, shuffle=True):
    paths = dataframe['Path Crop'].values
    labels_A = dataframe['TaskA_encoded'].values
    labels_B = dataframe['TaskB_encoded'].values
    
    labels_A_onehot = to_categorical(labels_A, num_classes_A)
    labels_B_onehot = to_categorical(labels_B, num_classes_B)
    
    dataset = tf.data.Dataset.from_tensor_slices((
        paths, labels_A_onehot, labels_B_onehot
    ))
    
    if shuffle:
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    
    dataset = dataset.map(
        lambda p, lA, lB: load_and_preprocess_image(p, lA, lB, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

BATCH_SIZE = 4

train_dataset = create_dataset(
    train_df, num_classes_A, num_classes_B,
    batch_size=BATCH_SIZE, augment=True, shuffle=True
)

val_dataset = create_dataset(
    val_df, num_classes_A, num_classes_B,
    batch_size=BATCH_SIZE, augment=False, shuffle=False
)

print(f"✅ Datasets created (batch_size={BATCH_SIZE})")

# ===============================
# 4. PHASE 1: Train Heads Only
# ===============================
print("\n" + "="*70)
print("🚀 PHASE 1: TRAINING HEADS ONLY")
print("="*70)

# Freeze backbone
print("\n🔒 Freezing EfficientNet backbone...")
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = False
        print(f"   Frozen: {layer.name}")

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
total = sum([tf.size(w).numpy() for w in model.weights])
print(f"\n📊 Parameters:")
print(f"   Total:     {total:,}")
print(f"   Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0
    ),
    loss={
        'task_subclass': 'categorical_crossentropy',
        'task_position': 'categorical_crossentropy'
    },
    loss_weights={
        'task_subclass': 1.0,
        'task_position': 1.0
    },
    metrics={
        'task_subclass': ['accuracy'],
        'task_position': ['accuracy']
    }
)

print("\n✅ Compiled:")
print("   Optimizer: Adam(lr=1e-3)")
print("   Loss: Categorical Crossentropy")

# Callbacks
save_dir = './mtl_phase1_results'
os.makedirs(save_dir, exist_ok=True)

callbacks_phase1 = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'phase1_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    
    tf.keras.callbacks.CSVLogger(
        os.path.join(save_dir, 'phase1_training.csv')
    )
]

# Train
print("\n" + "="*70)
print("🏋️  TRAINING PHASE 1...")
print("="*70)
print(f"   Epochs: 20")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Training samples: {len(train_df)}")
print(f"   Validation samples: {len(val_df)}")
print("="*70 + "\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=callbacks_phase1,
    verbose=1
)

print("\n✅ Phase 1 training complete!")

# ===============================
# 5. Save Phase 1 Results
# ===============================
print("\n" + "="*70)
print("💾 SAVING PHASE 1 RESULTS")
print("="*70)

# Save final model (not just best)
model.save(os.path.join(save_dir, 'phase1_final.h5'))
print(f"✅ Saved: phase1_final.h5")

model.save_weights(os.path.join(save_dir, 'phase1_weights.h5'))
print(f"✅ Saved: phase1_weights.h5")

# Save architecture
with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
    f.write(model.to_json())
print(f"✅ Saved: model_architecture.json")

# Save training info
phase1_results = {
    'final_epoch': len(history.history['loss']),
    'train_loss': float(history.history['loss'][-1]),
    'val_loss': float(history.history['val_loss'][-1]),
    'train_acc_A': float(history.history['task_subclass_accuracy'][-1]),
    'val_acc_A': float(history.history['val_task_subclass_accuracy'][-1]),
    'train_acc_B': float(history.history['task_position_accuracy'][-1]),
    'val_acc_B': float(history.history['val_task_position_accuracy'][-1]),
    'best_val_loss': float(min(history.history['val_loss'])),
    'best_val_acc_A': float(max(history.history['val_task_subclass_accuracy'])),
    'best_val_acc_B': float(max(history.history['val_task_position_accuracy']))
}

training_info = {
    'phase': 1,
    'num_classes_A': num_classes_A,
    'num_classes_B': num_classes_B,
    'batch_size': BATCH_SIZE,
    'mixed_precision': True,
    'backbone_frozen': True,
    'results': phase1_results,
    'label_classes_A': le_classA.classes_.tolist(),
    'label_classes_B': le_classB.classes_.tolist()
}

with open(os.path.join(save_dir, 'phase1_info.json'), 'w') as f:
    json.dump(training_info, f, indent=2)
print(f"✅ Saved: phase1_info.json")

# ===============================
# 6. Plot Phase 1 History
# ===============================
print("\n📊 Plotting Phase 1 history...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

epochs = range(1, len(history.history['loss']) + 1)

# Total Loss
axes[0, 0].plot(epochs, history.history['loss'], 'b-', label='Train', linewidth=2)
axes[0, 0].plot(epochs, history.history['val_loss'], 'r-', label='Val', linewidth=2)
axes[0, 0].set_title('Phase 1: Total Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Task A Accuracy
axes[0, 1].plot(epochs, history.history['task_subclass_accuracy'], 'b-', label='Train', linewidth=2)
axes[0, 1].plot(epochs, history.history['val_task_subclass_accuracy'], 'r-', label='Val', linewidth=2)
axes[0, 1].set_title('Phase 1: Task A Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Task B Accuracy
axes[1, 0].plot(epochs, history.history['task_position_accuracy'], 'b-', label='Train', linewidth=2)
axes[1, 0].plot(epochs, history.history['val_task_position_accuracy'], 'r-', label='Val', linewidth=2)
axes[1, 0].set_title('Phase 1: Task B Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Both Tasks
axes[1, 1].plot(epochs, history.history['task_subclass_accuracy'], 'b-', label='Task A Train', linewidth=2)
axes[1, 1].plot(epochs, history.history['val_task_subclass_accuracy'], 'r-', label='Task A Val', linewidth=2)
axes[1, 1].plot(epochs, history.history['task_position_accuracy'], 'b--', label='Task B Train', linewidth=2)
axes[1, 1].plot(epochs, history.history['val_task_position_accuracy'], 'r--', label='Task B Val', linewidth=2)
axes[1, 1].set_title('Phase 1: Both Tasks', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'phase1_history.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: phase1_history.png")
plt.show()

# ===============================
# 7. Phase 1 Summary
# ===============================
print("\n" + "="*70)
print("✅ PHASE 1 COMPLETE!")
print("="*70)

print(f"\n📊 Final Results:")
print(f"   Epochs trained: {len(history.history['loss'])}")
print(f"   ")
print(f"   Task A (Subclass):")
print(f"      Train Acc: {phase1_results['train_acc_A']:.4f} ({phase1_results['train_acc_A']*100:.2f}%)")
print(f"      Val Acc:   {phase1_results['val_acc_A']:.4f} ({phase1_results['val_acc_A']*100:.2f}%)")
print(f"      Best Val:  {phase1_results['best_val_acc_A']:.4f}")
print(f"   ")
print(f"   Task B (Position):")
print(f"      Train Acc: {phase1_results['train_acc_B']:.4f} ({phase1_results['train_acc_B']*100:.2f}%)")
print(f"      Val Acc:   {phase1_results['val_acc_B']:.4f} ({phase1_results['val_acc_B']*100:.2f}%)")
print(f"      Best Val:  {phase1_results['best_val_acc_B']:.4f}")
print(f"   ")
print(f"   Validation Loss:")
print(f"      Final: {phase1_results['val_loss']:.4f}")
print(f"      Best:  {phase1_results['best_val_loss']:.4f}")

print(f"\n📁 All files saved in: {save_dir}/")
print(f"   - phase1_best.h5 (best checkpoint)")
print(f"   - phase1_final.h5 (final model)")
print(f"   - phase1_weights.h5")
print(f"   - phase1_info.json")
print(f"   - phase1_training.csv")
print(f"   - phase1_history.png")

print("\n🎯 Next step: Run Phase 2 training script")
print("   It will load phase1_best.h5 and fine-tune the backbone")

print("\n✨ Phase 1 Done! ✨")