# ===============================
# MTL Training - PHASE 2: Fine-tuning (200 Epochs)
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
# 1. Load Phase 1 Model
# ===============================
print("\n" + "="*70)
print("📥 LOADING PHASE 1 MODEL")
print("="*70)

phase1_dir = './mtl_phase1_results'
phase1_model_path = os.path.join(phase1_dir, 'phase1_best.h5')

if not os.path.exists(phase1_model_path):
    raise FileNotFoundError(f"❌ Phase 1 model not found: {phase1_model_path}")

print(f"Loading: {phase1_model_path}")
model = load_model(phase1_model_path, compile=False)
print("✅ Phase 1 model loaded")

# Load Phase 1 info
with open(os.path.join(phase1_dir, 'phase1_info.json'), 'r') as f:
    phase1_info = json.load(f)

print(f"\n📊 Phase 1 Results (Starting point):")
print(f"   Task A Val Acc: {phase1_info['results']['val_acc_A']:.4f} ({phase1_info['results']['val_acc_A']*100:.2f}%)")
print(f"   Task B Val Acc: {phase1_info['results']['val_acc_B']:.4f} ({phase1_info['results']['val_acc_B']*100:.2f}%)")
print(f"   Val Loss:       {phase1_info['results']['best_val_loss']:.4f}")

num_classes_A = phase1_info['num_classes_A']
num_classes_B = phase1_info['num_classes_B']
BATCH_SIZE = phase1_info['batch_size']

print(f"\n📋 Config:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Task A classes: {num_classes_A}")
print(f"   Task B classes: {num_classes_B}")

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

# Load encoders
with open('./encoders/label_encoder_taskA.pkl', 'rb') as f:
    le_classA = pickle.load(f)
with open('./encoders/label_encoder_taskB.pkl', 'rb') as f:
    le_classB = pickle.load(f)

print(f"✅ Encoders loaded")

dataframe['TaskA_encoded'] = le_classA.transform(dataframe['Sub_class_New'])
dataframe['TaskB_encoded'] = le_classB.transform(dataframe['Sub_Position_Label'])

# Split (same as Phase 1)
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
# 4. PHASE 2: Fine-tune Backbone
# ===============================
print("\n" + "="*70)
print("🔥 PHASE 2: FINE-TUNING BACKBONE (BLOCK7) - 200 EPOCHS")
print("="*70)

# Unfreeze block7 only
print("\n🔓 Unfreezing block7...")
total_unfrozen = 0
for layer in model.layers:
    if 'efficientnet' in layer.name.lower():
        layer.trainable = True
        # Freeze all first
        for sublayer in layer.layers:
            sublayer.trainable = False
        
        # Unfreeze block7 and top
        for sublayer in layer.layers:
            if 'block7' in sublayer.name or 'top_' in sublayer.name:
                sublayer.trainable = True
                total_unfrozen += 1
        
        print(f"   Unfrozen {total_unfrozen} layers in {layer.name}")

trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
total = sum([tf.size(w).numpy() for w in model.weights])
print(f"\n📊 Parameters:")
print(f"   Total:     {total:,}")
print(f"   Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

# Compile with low LR
model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=5e-5,
        momentum=0.9,
        nesterov=True,
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
print("   Optimizer: SGD")
print("   Learning Rate: 5e-5")
print("   Momentum: 0.9")
print("   Nesterov: True")

# Callbacks (NO EARLY STOPPING)
save_dir = './mtl_phase2_results'
os.makedirs(save_dir, exist_ok=True)

callbacks_phase2 = [
    # Save best model
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'phase2_best.h5'),
        monitor='val_loss',
        save_best_only=True,
        verbose=1,
        save_weights_only=False
    ),
    
    # Save every 10 epochs
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(save_dir, 'phase2_epoch_{epoch:03d}.h5'),
        save_freq='epoch',
        period=10,
        verbose=0,
        save_weights_only=False
    ),
    
    # Reduce LR when plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # More patience for 200 epochs
        min_lr=1e-8,
        verbose=1
    ),
    
    # Log to CSV
    tf.keras.callbacks.CSVLogger(
        os.path.join(save_dir, 'phase2_training.csv'),
        separator=',',
        append=False
    )
]

# Train - 200 EPOCHS
print("\n" + "="*70)
print("🏋️  TRAINING PHASE 2 - 200 EPOCHS (NO EARLY STOPPING)")
print("="*70)
print(f"   Epochs: 200 (full training)")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: 5e-5 (will decay with ReduceLROnPlateau)")
print(f"   Training samples: {len(train_df)}")
print(f"   Validation samples: {len(val_df)}")
print(f"   ")
print(f"   ⏱️  Estimated time:")
print(f"   ~70 seconds per epoch")
print(f"   Total: ~3.9 hours (14,000 seconds)")
print("="*70 + "\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=callbacks_phase2,
    verbose=1
)

print("\n✅ Phase 2 training complete (200 epochs)!")

# ===============================
# 5. Save Phase 2 Results
# ===============================
print("\n" + "="*70)
print("💾 SAVING PHASE 2 RESULTS")
print("="*70)

# Save final model (epoch 200)
model.save(os.path.join(save_dir, 'phase2_final.h5'))
print(f"✅ Saved: phase2_final.h5")

model.save_weights(os.path.join(save_dir, 'phase2_weights_final.h5'))
print(f"✅ Saved: phase2_weights_final.h5")

with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
    f.write(model.to_json())
print(f"✅ Saved: model_architecture.json")

# Save training info
phase2_results = {
    'total_epochs': 200,
    'train_loss': float(history.history['loss'][-1]),
    'val_loss': float(history.history['val_loss'][-1]),
    'train_acc_A': float(history.history['task_subclass_accuracy'][-1]),
    'val_acc_A': float(history.history['val_task_subclass_accuracy'][-1]),
    'train_acc_B': float(history.history['task_position_accuracy'][-1]),
    'val_acc_B': float(history.history['val_task_position_accuracy'][-1]),
    'best_val_loss': float(min(history.history['val_loss'])),
    'best_val_acc_A': float(max(history.history['val_task_subclass_accuracy'])),
    'best_val_acc_B': float(max(history.history['val_task_position_accuracy'])),
    'best_epoch_loss': int(np.argmin(history.history['val_loss']) + 1),
    'best_epoch_acc_A': int(np.argmax(history.history['val_task_subclass_accuracy']) + 1),
    'best_epoch_acc_B': int(np.argmax(history.history['val_task_position_accuracy']) + 1)
}

training_info = {
    'phase': 2,
    'phase1_model': phase1_model_path,
    'num_classes_A': num_classes_A,
    'num_classes_B': num_classes_B,
    'batch_size': BATCH_SIZE,
    'mixed_precision': True,
    'backbone_unfrozen': 'block7 + top',
    'learning_rate_initial': 5e-5,
    'optimizer': 'SGD',
    'epochs': 200,
    'early_stopping': False,
    'phase1_results': phase1_info['results'],
    'phase2_results': phase2_results,
    'label_classes_A': le_classA.classes_.tolist(),
    'label_classes_B': le_classB.classes_.tolist()
}

with open(os.path.join(save_dir, 'phase2_info.json'), 'w') as f:
    json.dump(training_info, f, indent=2)
print(f"✅ Saved: phase2_info.json")

# ===============================
# 6. Plot Training History
# ===============================
print("\n📊 Plotting Phase 2 history (200 epochs)...")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

epochs = range(1, len(history.history['loss']) + 1)

# Total Loss
axes[0, 0].plot(epochs, history.history['loss'], 'b-', label='Train', linewidth=1, alpha=0.7)
axes[0, 0].plot(epochs, history.history['val_loss'], 'r-', label='Val', linewidth=2)
axes[0, 0].axvline(x=phase2_results['best_epoch_loss'], color='green', linestyle='--', 
                    alpha=0.5, label=f'Best (epoch {phase2_results["best_epoch_loss"]})')
axes[0, 0].set_title('Phase 2: Total Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Task A Accuracy
axes[0, 1].plot(epochs, history.history['task_subclass_accuracy'], 'b-', label='Train', linewidth=1, alpha=0.7)
axes[0, 1].plot(epochs, history.history['val_task_subclass_accuracy'], 'r-', label='Val', linewidth=2)
axes[0, 1].axhline(y=phase1_info['results']['val_acc_A'], color='orange', linestyle='--', 
                    label=f'Phase 1: {phase1_info["results"]["val_acc_A"]:.3f}', alpha=0.7)
axes[0, 1].axvline(x=phase2_results['best_epoch_acc_A'], color='green', linestyle='--', 
                    alpha=0.5, label=f'Best (epoch {phase2_results["best_epoch_acc_A"]})')
axes[0, 1].set_title('Phase 2: Task A (Subclass) Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Task B Accuracy
axes[1, 0].plot(epochs, history.history['task_position_accuracy'], 'b-', label='Train', linewidth=1, alpha=0.7)
axes[1, 0].plot(epochs, history.history['val_task_position_accuracy'], 'r-', label='Val', linewidth=2)
axes[1, 0].axhline(y=phase1_info['results']['val_acc_B'], color='orange', linestyle='--', 
                    label=f'Phase 1: {phase1_info["results"]["val_acc_B"]:.3f}', alpha=0.7)
axes[1, 0].axvline(x=phase2_results['best_epoch_acc_B'], color='green', linestyle='--', 
                    alpha=0.5, label=f'Best (epoch {phase2_results["best_epoch_acc_B"]})')
axes[1, 0].set_title('Phase 2: Task B (Position) Accuracy', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Both Tasks Comparison
axes[1, 1].plot(epochs, history.history['task_subclass_accuracy'], 'b-', label='Task A Train', linewidth=1, alpha=0.6)
axes[1, 1].plot(epochs, history.history['val_task_subclass_accuracy'], 'r-', label='Task A Val', linewidth=2)
axes[1, 1].plot(epochs, history.history['task_position_accuracy'], 'b--', label='Task B Train', linewidth=1, alpha=0.6)
axes[1, 1].plot(epochs, history.history['val_task_position_accuracy'], 'r--', label='Task B Val', linewidth=2)
axes[1, 1].set_title('Phase 2: Both Tasks', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'phase2_history_full.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: phase2_history_full.png")
plt.show()

# ===============================
# 7. Final Summary
# ===============================
print("\n" + "="*70)
print("🎉 PHASE 2 TRAINING COMPLETE - 200 EPOCHS!")
print("="*70)

print(f"\n📊 Phase 1 → Phase 2 Comparison:")

print(f"\n   Task A (Subclass):")
print(f"      Phase 1:          {phase1_info['results']['val_acc_A']:.4f} ({phase1_info['results']['val_acc_A']*100:.2f}%)")
print(f"      Phase 2 (final):  {phase2_results['val_acc_A']:.4f} ({phase2_results['val_acc_A']*100:.2f}%)")
print(f"      Phase 2 (best):   {phase2_results['best_val_acc_A']:.4f} ({phase2_results['best_val_acc_A']*100:.2f}%) @ epoch {phase2_results['best_epoch_acc_A']}")
print(f"      Improvement:      {(phase2_results['best_val_acc_A'] - phase1_info['results']['val_acc_A'])*100:+.2f}%")

print(f"\n   Task B (Position):")
print(f"      Phase 1:          {phase1_info['results']['val_acc_B']:.4f} ({phase1_info['results']['val_acc_B']*100:.2f}%)")
print(f"      Phase 2 (final):  {phase2_results['val_acc_B']:.4f} ({phase2_results['val_acc_B']*100:.2f}%)")
print(f"      Phase 2 (best):   {phase2_results['best_val_acc_B']:.4f} ({phase2_results['best_val_acc_B']*100:.2f}%) @ epoch {phase2_results['best_epoch_acc_B']}")
print(f"      Improvement:      {(phase2_results['best_val_acc_B'] - phase1_info['results']['val_acc_B'])*100:+.2f}%")

print(f"\n   Validation Loss:")
print(f"      Phase 1:          {phase1_info['results']['best_val_loss']:.4f}")
print(f"      Phase 2 (final):  {phase2_results['val_loss']:.4f}")
print(f"      Phase 2 (best):   {phase2_results['best_val_loss']:.4f} @ epoch {phase2_results['best_epoch_loss']}")

print(f"\n📁 All files saved in: {save_dir}/")
print(f"   - phase2_best.h5 (best checkpoint) ⭐")
print(f"   - phase2_final.h5 (epoch 200)")
print(f"   - phase2_epoch_*.h5 (every 10 epochs)")
print(f"   - phase2_weights_final.h5")
print(f"   - phase2_info.json")
print(f"   - phase2_training.csv")
print(f"   - phase2_history_full.png")

print("\n💡 Recommendation:")
if phase2_results['best_epoch_loss'] < 180:
    print(f"   ✅ Best model was at epoch {phase2_results['best_epoch_loss']}")
    print(f"   Use: phase2_best.h5")
else:
    print(f"   ⚠️  Model was still improving at epoch 200")
    print(f"   Consider training more epochs OR use phase2_best.h5")

print("\n🎯 Next step: Evaluate on test set!")
print("   Recommended: ./mtl_phase2_results/phase2_best.h5")

print("\n✨ 200 Epochs Complete! ✨")