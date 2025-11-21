# Key Training Parameters for STT Fine-Tuning

## Overview

This guide covers the essential training parameters (hyperparameters) used when fine-tuning speech-to-text models, particularly focusing on Whisper and similar transformer-based architectures. Understanding these parameters is crucial for achieving optimal model performance.

---

## Core Training Parameters

### 1. Epochs

**Definition**: One epoch represents a complete pass through the entire training dataset.

**Typical Range**: 3-20 epochs for fine-tuning

**How It Works**:
```
Total Training Steps = (Dataset Size / Batch Size) × Number of Epochs
```

**Considerations**:

- **Too Few Epochs**: Model underfits, doesn't learn patterns
  - Symptoms: High training loss, poor performance
  - Solution: Increase epochs

- **Too Many Epochs**: Model overfits, memorizes training data
  - Symptoms: Training loss decreases but validation loss increases
  - Solution: Use early stopping or reduce epochs

**Best Practices**:
- Start with 5-10 epochs for initial experiments
- Use early stopping to prevent overtraining
- Monitor validation metrics to determine optimal number
- Smaller datasets need fewer epochs (3-5)
- Larger datasets can benefit from more epochs (10-20)

**Example Configuration**:
```python
training_args = Seq2SeqTrainingArguments(
    num_train_epochs=10,  # Complete passes through data
)
```

---

### 2. Batch Size

**Definition**: Number of training examples processed simultaneously in one forward/backward pass.

**Types**:
- **per_device_train_batch_size**: Batch size per GPU/CPU
- **per_device_eval_batch_size**: Batch size for validation
- **gradient_accumulation_steps**: Simulates larger batch sizes

**Typical Range**: 4-32 per device (depends on GPU memory)

**Effective Batch Size Calculation**:
```
Effective Batch Size = per_device_batch_size × num_devices × gradient_accumulation_steps
```

**Trade-offs**:

| Batch Size | Advantages | Disadvantages |
|-----------|-----------|---------------|
| Small (4-8) | Less memory usage<br>More gradient updates<br>Better generalization | Slower training<br>Noisier gradients<br>Less stable |
| Large (16-32+) | Faster training<br>Stable gradients<br>Better GPU utilization | High memory requirements<br>May overfit<br>Needs more data |

**Best Practices**:
- Start with largest batch size that fits in GPU memory
- Use gradient accumulation to simulate larger batches
- Typical setup: `batch_size=16, gradient_accumulation_steps=2` (effective batch size = 32)
- Reduce batch size if encountering OOM (Out of Memory) errors

**Example**:
```python
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,  # Can be larger (no gradients stored)
    gradient_accumulation_steps=4,  # Effective batch = 8 × 4 = 32
)
```

---

### 3. Learning Rate

**Definition**: Controls how much model weights are updated during training. The most critical hyperparameter.

**Typical Range**: 1e-5 to 1e-4 for fine-tuning

**Components**:

#### Base Learning Rate
```python
learning_rate = 5e-5  # Common starting point for fine-tuning
```

#### Learning Rate Schedule
Controls how learning rate changes during training:

**Common Schedules**:

1. **Linear Decay**
   ```python
   lr_scheduler_type = "linear"
   # LR decreases linearly from initial value to 0
   ```

2. **Cosine Annealing**
   ```python
   lr_scheduler_type = "cosine"
   # LR follows cosine curve, smooth decay
   ```

3. **Constant**
   ```python
   lr_scheduler_type = "constant"
   # LR stays fixed throughout training
   ```

4. **Constant with Warmup**
   ```python
   lr_scheduler_type = "constant_with_warmup"
   warmup_steps = 500
   # LR increases linearly for warmup, then stays constant
   ```

#### Warmup Steps
**Definition**: Number of steps where learning rate gradually increases from 0 to target value.

**Purpose**: Prevents unstable training at the beginning

**Typical Range**: 500-2000 steps (or 5-10% of total steps)

```python
warmup_steps = 500  # Absolute number
# OR
warmup_ratio = 0.1  # 10% of total training steps
```

**Visualization**:
```
Learning Rate Schedule (Linear with Warmup)

LR  ^
    |     /‾‾‾‾‾‾‾‾‾‾\
    |    /              \
    |   /                 \
    |  /                    \
    | /                       \
    |/_________________________\___> Steps
      Warmup    Training       End
```

**Best Practices**:
- **For fine-tuning**: Start with 1e-5 to 5e-5
- **For training from scratch**: Start with 1e-4 to 5e-4
- Use warmup to stabilize initial training
- Monitor loss curves to adjust if needed
- If loss explodes: reduce learning rate
- If loss plateaus early: increase learning rate

**Example**:
```python
training_args = Seq2SeqTrainingArguments(
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,
)
```

---

### 4. Weight Decay

**Definition**: L2 regularization that penalizes large weights to prevent overfitting.

**Typical Range**: 0.0 to 0.1

**How It Works**: Adds penalty term to loss function
```
Loss_total = Loss_task + weight_decay × Σ(weights²)
```

**Guidelines**:
- **No weight decay (0.0)**: No regularization
- **Light (0.01)**: Minimal regularization, common default
- **Moderate (0.05)**: Good for smaller datasets
- **Heavy (0.1)**: Strong regularization for overfitting prevention

**Best Practices**:
- Start with 0.01 (common default)
- Increase if overfitting occurs
- Decrease if underfitting
- Monitor validation metrics

```python
weight_decay = 0.01  # L2 regularization strength
```

---

### 5. Gradient Clipping

**Definition**: Limits the maximum gradient value to prevent exploding gradients.

**Parameter**: `max_grad_norm`

**Typical Value**: 1.0

**How It Works**:
```python
if gradient_norm > max_grad_norm:
    gradient = gradient × (max_grad_norm / gradient_norm)
```

**Purpose**:
- Prevents training instability
- Stops gradient explosions
- Particularly important for RNNs and long sequences

**Best Practices**:
- Default value of 1.0 works well for most cases
- Increase to 5.0 if you need more gradient freedom
- Decrease to 0.5 for very stable training

```python
max_grad_norm = 1.0  # Clip gradients above this norm
```

---

### 6. Dropout

**Definition**: Randomly drops (sets to zero) a percentage of neurons during training to prevent overfitting.

**Typical Range**: 0.0 to 0.3

**Types**:
- **Attention Dropout**: Applied to attention weights
- **Activation Dropout**: Applied to hidden states
- **Overall Dropout**: General dropout rate

**Guidelines**:
- **No dropout (0.0)**: No regularization
- **Light (0.1)**: Standard for well-sized datasets
- **Moderate (0.2)**: Good for smaller datasets
- **Heavy (0.3)**: Aggressive overfitting prevention

**Note**: Dropout is only active during training, disabled during evaluation.

```python
# In model configuration
dropout = 0.1
attention_dropout = 0.1
```

---

## Evaluation and Monitoring Parameters

### 7. Evaluation Strategy

**Definition**: How often to evaluate model on validation set.

**Options**:

```python
# Evaluate every N steps
evaluation_strategy = "steps"
eval_steps = 500  # Evaluate every 500 training steps

# OR evaluate at end of each epoch
evaluation_strategy = "epoch"
```

**Best Practices**:
- For small datasets: `evaluation_strategy="epoch"`
- For large datasets: `evaluation_strategy="steps"` with `eval_steps=500-1000`
- More frequent evaluation = better monitoring but slower training

---

### 8. Save Strategy

**Definition**: How often to save model checkpoints.

```python
save_strategy = "steps"  # or "epoch"
save_steps = 500  # Save every 500 steps
save_total_limit = 3  # Keep only best 3 checkpoints
```

**Best Practices**:
- Match save strategy to evaluation strategy
- Use `save_total_limit` to prevent disk space issues
- Enable `load_best_model_at_end=True` for optimal final model

---

### 9. Logging

**Definition**: How often to log training metrics.

```python
logging_steps = 100  # Log every 100 steps
report_to = ["tensorboard"]  # or "wandb", "none"
```

---

## Advanced Parameters

### 10. Optimizer

**Definition**: Algorithm used to update model weights.

**Common Options**:

```python
# AdamW (default, recommended)
optim = "adamw_torch"

# 8-bit AdamW (memory efficient)
optim = "adamw_8bit"

# Adafactor (memory efficient alternative)
optim = "adafactor"
```

**Best Practice**: Use AdamW for most cases

---

### 11. Mixed Precision Training

**Definition**: Uses lower precision (FP16/BF16) to speed up training and reduce memory.

```python
fp16 = True  # For older GPUs (Nvidia Volta, Turing)
bf16 = True  # For newer GPUs (Nvidia Ampere, Ada) - more stable
```

**Benefits**:
- 2x faster training
- 50% less memory usage
- Minimal accuracy impact

---

### 12. Generation Parameters (for Seq2Seq)

**For STT models during evaluation**:

```python
# Maximum length of generated transcription
generation_max_length = 225

# Number of beams for beam search
generation_num_beams = 1  # Greedy decoding (fastest)
# OR
generation_num_beams = 5  # Better quality, slower
```

---

## Complete Example Configuration

```python
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    # Output
    output_dir="./whisper-finetuned",

    # Training duration
    num_train_epochs=10,

    # Batch sizes
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,  # Effective batch size = 32

    # Learning rate
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=500,

    # Regularization
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Evaluation
    evaluation_strategy="steps",
    eval_steps=500,

    # Saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,

    # Logging
    logging_steps=100,
    report_to=["tensorboard"],

    # Performance
    fp16=True,  # or bf16=True for newer GPUs

    # Generation (for evaluation)
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=1,

    # Optimization
    optim="adamw_torch",

    # Misc
    push_to_hub=False,
    metric_for_best_model="wer",  # Word Error Rate
    greater_is_better=False,  # Lower WER is better
)
```

---

## Parameter Tuning Guidelines

### Starting Point (Conservative)
```python
num_train_epochs=5
per_device_train_batch_size=8
learning_rate=1e-5
warmup_steps=500
weight_decay=0.01
```

### For Small Datasets (< 20 hours)
```python
num_train_epochs=3-5
per_device_train_batch_size=4-8
learning_rate=1e-5
weight_decay=0.05  # Higher regularization
dropout=0.2
```

### For Large Datasets (> 100 hours)
```python
num_train_epochs=10-20
per_device_train_batch_size=16-32
learning_rate=5e-5
weight_decay=0.01
warmup_steps=1000
```

### If Overfitting
```python
# Reduce epochs
num_train_epochs -= 2

# Increase regularization
weight_decay += 0.02
dropout += 0.1

# Use early stopping
early_stopping_patience=3
```

### If Underfitting
```python
# Increase training
num_train_epochs += 5

# Increase learning rate
learning_rate *= 2

# Reduce regularization
weight_decay /= 2
```

---

## Monitoring Guidelines

Track these metrics during training:

1. **Training Loss**: Should steadily decrease
2. **Validation Loss**: Should decrease and track training loss
3. **WER (Word Error Rate)**: Should steadily decrease
4. **Learning Rate**: Check schedule is working as expected
5. **Gradient Norm**: Should be stable, not exploding

**Red Flags**:
- Validation loss increases while training loss decreases → Overfitting
- Both losses plateau early → Underfitting or learning rate too low
- Loss becomes NaN → Gradient explosion (reduce LR or clip gradients)
- No improvement after several epochs → Hyperparameter adjustment needed

---

## Summary Table

| Parameter | Typical Range | Purpose | Adjustment Strategy |
|-----------|---------------|---------|---------------------|
| Epochs | 3-20 | Training duration | Monitor validation loss |
| Batch Size | 4-32 | Memory/speed trade-off | Maximize within GPU limits |
| Learning Rate | 1e-5 to 1e-4 | Update speed | Reduce if unstable |
| Weight Decay | 0.0-0.1 | Regularization | Increase if overfitting |
| Warmup Steps | 500-2000 | Training stability | 5-10% of total steps |
| Gradient Clipping | 1.0 | Prevent explosions | Keep at 1.0 usually |
| Dropout | 0.0-0.3 | Regularization | Increase if overfitting |

---

## Conclusion

Successful fine-tuning requires careful balancing of these parameters. Start with conservative defaults, monitor validation metrics closely, and adjust based on training behavior. Remember that every dataset is different, so experimentation and iteration are key to achieving optimal results.
