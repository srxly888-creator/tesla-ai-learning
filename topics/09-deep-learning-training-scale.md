# Deep Learning Training at Scale - Tesla's Approach

## Introduction

Training neural networks at the scale required for autonomous driving presents unique challenges. This document covers the techniques, infrastructure, and best practices for training deep learning models at massive scale, inspired by Tesla's approach.

## Data Pipeline

### Data Collection

```python
class FleetDataManager:
    """Manage data from Tesla fleet"""
    
    def __init__(self):
        self.fleet_size = 4_000_000  # 4M+ vehicles
        self.daily_data = 100  # petabytes per day
        
    def collect_data(self):
        """Collect high-value data from fleet"""
        # Intelligent data selection
        selection_criteria = {
            'intervention_events': True,      # Human took over
            'uncertain_predictions': True,    # High entropy
            'rare_scenarios': True,           # Unusual situations
            'edge_cases': True,               # Boundary conditions
            'geographic_diversity': True      # Different locations
        }
        
        for vehicle in self.active_fleet():
            # Evaluate data value
            data_value = self.compute_data_value(vehicle.current_scenario)
            
            if data_value > threshold:
                # Request data upload
                self.request_upload(
                    vehicle.id,
                    data_type='video_and_telemetry',
                    duration=10  # seconds
                )
    
    def compute_data_value(self, scenario):
        """Compute value of collecting this data"""
        value = 0
        
        # Novelty
        if self.is_novel(scenario):
            value += 10
        
        # Uncertainty
        if scenario.model_uncertainty > 0.5:
            value += 5
        
        # Safety relevance
        if scenario.involves_vulnerable_users:
            value += 8
        
        # Rare conditions
        if scenario.is_rare:
            value += 7
        
        return value
```

### Data Processing Pipeline

```python
class DataPipeline:
    """Process raw data into training-ready format"""
    
    def __init__(self):
        self.stages = [
            'ingest',
            'validate',
            'transform',
            'augment',
            'label',
            'store'
        ]
        
    def process_batch(self, raw_data_batch):
        """Process batch of raw data"""
        processed = []
        
        for raw_data in raw_data_batch:
            # 1. Validate
            if not self.validate(raw_data):
                continue
            
            # 2. Transform
            transformed = self.transform(raw_data)
            
            # 3. Augment
            augmented = self.augment(transformed)
            
            # 4. Auto-label
            labels = self.auto_label(augmented)
            
            # 5. Quality check
            if self.quality_check(augmented, labels):
                processed.append({
                    'data': augmented,
                    'labels': labels,
                    'metadata': raw_data.metadata
                })
        
        return processed
    
    def auto_label(self, data):
        """Automatically generate labels using existing models"""
        # Run ensemble of models
        predictions = []
        for model in self.labeling_models:
            pred = model.inference(data)
            predictions.append(pred)
        
        # Consensus
        labels = self.consensus_voting(predictions)
        
        # Confidence filter
        if labels.confidence < 0.9:
            # Send for human review
            self.queue_for_review(data, labels)
        
        return labels
    
    def augment(self, data):
        """Apply data augmentation"""
        augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_hue,
            self.random_noise,
            self.random_blur,
            self.random_occlusion,
            self.weather_simulation,
            self.domain_randomization
        ]
        
        for aug in augmentations:
            if random.random() < aug.probability:
                data = aug.apply(data)
        
        return data
```

## Distributed Training

### Data Parallel Training

```python
import torch.distributed as dist
import torch.multiprocessing as mp

class DistributedTrainer:
    """Distributed training across multiple GPUs/nodes"""
    
    def __init__(self, model, world_size):
        self.model = model
        self.world_size = world_size
        
    def train(self, train_fn, port):
        """Launch distributed training"""
        mp.spawn(
            self.train_worker,
            args=(self.world_size, train_fn, port),
            nprocs=self.world_size,
            join=True
        )
    
    def train_worker(self, rank, world_size, train_fn, port):
        """Training worker for each process"""
        # Setup distributed
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://localhost:{port}',
            world_size=world_size,
            rank=rank
        )
        
        # Wrap model
        model = self.model.to(rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank]
        )
        
        # Create distributed sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=8
        )
        
        # Training loop
        for epoch in range(self.num_epochs):
            train_sampler.set_epoch(epoch)
            
            for batch in train_loader:
                loss = train_fn(model, batch, rank)
                
                if rank == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
```

### Model Parallel Training

```python
class ModelParallelTrainer:
    """Split large model across multiple GPUs"""
    
    def __init__(self, model, device_map):
        """
        Args:
            model: neural network
            device_map: {layer_name: device_id}
        """
        self.model = model
        self.device_map = device_map
        
        # Place each layer on specified device
        for name, module in model.named_modules():
            if name in device_map:
                module.to(device_map[name])
    
    def forward(self, x):
        """Forward pass across devices"""
        for name, module in self.model.named_modules():
            if name in self.device_map:
                device = self.device_map[name]
                x = x.to(device)
                x = module(x)
        
        return x
    
    def train_step(self, batch):
        """Training step with model parallelism"""
        # Forward
        output = self.forward(batch['input'])
        
        # Move output to last device for loss
        output = output.to(self.last_device)
        target = batch['target'].to(self.last_device)
        
        # Loss
        loss = self.criterion(output, target)
        
        # Backward
        loss.backward()
        
        # Gradient synchronization happens automatically
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

### Pipeline Parallelism

```python
class PipelineParallel:
    """Pipeline parallelism for large models"""
    
    def __init__(self, model_stages, devices):
        """
        Args:
            model_stages: list of model chunks
            devices: list of device ids
        """
        self.stages = [stage.to(device) for stage, device in zip(model_stages, devices)]
        self.devices = devices
        
    def forward_backward(self, micro_batches):
        """Pipeline forward and backward passes"""
        num_stages = len(self.stages)
        num_micro_batches = len(micro_batches)
        
        # Schedule
        activations = {}
        gradients = {}
        
        for step in range(num_stages + num_micro_batches - 1):
            # Forward passes
            for mb_idx in range(min(step + 1, num_micro_batches)):
                stage_idx = step - mb_idx
                
                if 0 <= stage_idx < num_stages:
                    # Get input
                    if stage_idx == 0:
                        inp = micro_batches[mb_idx]
                    else:
                        inp = activations[(stage_idx-1, mb_idx)]
                    
                    # Move to device
                    inp = inp.to(self.devices[stage_idx])
                    
                    # Forward
                    out = self.stages[stage_idx](inp)
                    activations[(stage_idx, mb_idx)] = out
            
            # Backward passes
            for mb_idx in range(max(0, step - num_stages + 1), min(step + 1, num_micro_batches)):
                stage_idx = num_stages - 1 - (step - mb_idx)
                
                if 0 <= stage_idx < num_stages:
                    # Get activation
                    act = activations[(stage_idx, mb_idx)]
                    
                    # Get gradient
                    if stage_idx == num_stages - 1:
                        grad = self.compute_loss_gradient(act, micro_batches[mb_idx])
                    else:
                        grad = gradients[(stage_idx+1, mb_idx)]
                    
                    # Backward
                    act.backward(grad)
                    
                    # Store input gradient
                    if stage_idx > 0:
                        gradients[(stage_idx, mb_idx)] = act.grad
```

## Mixed Precision Training

### Automatic Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

class AMPTrainer:
    """Automatic Mixed Precision training"""
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.scaler = GradScaler()
        
    def train_step(self, batch):
        """Training step with AMP"""
        # Move to GPU
        inputs = batch['input'].cuda()
        targets = batch['target'].cuda()
        
        # Forward with autocast
        with autocast():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        # Backward with scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step with unscaling
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        return loss.item()
```

### Custom Mixed Precision

```python
class CustomMixedPrecision:
    """Custom mixed precision implementation"""
    
    def __init__(self, model):
        self.model = model
        
        # Convert certain layers to half precision
        self.convert_to_fp16()
        
        # Keep batch norm in fp32
        self.keep_bn_fp32()
        
    def convert_to_fp16(self):
        """Convert model to FP16"""
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.half()
                
    def keep_bn_fp32(self):
        """Keep batch norm in FP32 for stability"""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.float()
    
    def forward(self, x):
        """Forward with mixed precision"""
        # Input in fp16
        x = x.half()
        
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                # Batch norm in fp32
                x = x.float()
                x = module(x)
                x = x.half()
            else:
                x = module(x)
        
        return x
```

## Gradient Accumulation

```python
class GradientAccumulator:
    """Accumulate gradients for large effective batch size"""
    
    def __init__(self, model, optimizer, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def train_step(self, batch):
        """Training step with gradient accumulation"""
        # Forward
        outputs = self.model(batch['input'])
        loss = self.criterion(outputs, batch['target'])
        
        # Normalize loss
        loss = loss / self.accumulation_steps
        
        # Backward
        loss.backward()
        
        self.step_count += 1
        
        # Update only after accumulation
        if self.step_count % self.accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps
```

## Learning Rate Scheduling

### Warmup + Cosine Decay

```python
class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine decay"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        """Update learning rate"""
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * \
                 (1 + np.cos(np.pi * progress))
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

### One-Cycle Policy

```python
class OneCycleScheduler:
    """One-cycle learning rate policy"""
    
    def __init__(self, optimizer, max_lr, total_steps):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.current_step = 0
        
        # Calculate phases
        self.phase1_end = int(total_steps * 0.45)
        self.phase2_end = int(total_steps * 0.9)
        
    def step(self):
        """Update learning rate and momentum"""
        self.current_step += 1
        
        pct = self.current_step / self.total_steps
        
        if self.current_step <= self.phase1_end:
            # Phase 1: Increase LR
            lr = self.max_lr * (1 + pct / 0.45 * (1 - 0.1)) / 1.1
            momentum = 0.95 - 0.15 * pct / 0.45
        elif self.current_step <= self.phase2_end:
            # Phase 2: Decrease LR
            lr = self.max_lr * (1 - (pct - 0.45) / 0.45)
            momentum = 0.8 + 0.15 * (pct - 0.45) / 0.45
        else:
            # Phase 3: Final decay
            lr = self.max_lr * 0.01 * (1 - (pct - 0.9) / 0.1)
            momentum = 0.95
        
        # Update
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['betas'] = (momentum, param_group['betas'][1])
```

## Model Optimization

### Pruning

```python
class ModelPruner:
    """Prune model for efficiency"""
    
    def __init__(self, model, pruning_ratio=0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        
    def prune(self):
        """Prune model weights"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Compute importance scores
                importance = self.compute_importance(module.weight)
                
                # Determine threshold
                threshold = np.percentile(
                    importance.cpu().numpy(),
                    self.pruning_ratio * 100
                )
                
                # Create mask
                mask = importance > threshold
                
                # Apply mask
                module.weight.data *= mask.float()
                
                # Store mask for fine-tuning
                module.weight_mask = mask
    
    def compute_importance(self, weight):
        """Compute weight importance"""
        # L1 magnitude
        return torch.abs(weight)
    
    def iterative_pruning(self, train_fn, num_iterations=10):
        """Iterative pruning with retraining"""
        for i in range(num_iterations):
            # Prune a fraction
            self.prune_step(ratio=self.pruning_ratio / num_iterations)
            
            # Retrain
            train_fn(self.model, epochs=5)
    
    def fine_tune(self, train_loader, epochs=10):
        """Fine-tune after pruning"""
        # Freeze masks
        self.freeze_masks()
        
        # Train
        for epoch in range(epochs):
            for batch in train_loader:
                self.train_step(batch)
```

### Quantization

```python
class ModelQuantizer:
    """Quantize model for inference"""
    
    def __init__(self, model):
        self.model = model
        
    def quantize_dynamic(self):
        """Dynamic quantization"""
        quantized = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        return quantized
    
    def quantize_static(self, calibration_loader):
        """Static quantization with calibration"""
        # Prepare
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for batch in calibration_loader:
                self.model(batch)
        
        # Convert
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model
    
    def quantize_aware_training(self, train_loader, epochs=10):
        """Quantization aware training"""
        # Prepare
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        # Train
        for epoch in range(epochs):
            for batch in train_loader:
                self.train_step(batch)
        
        # Convert
        torch.quantization.convert(self.model, inplace=True)
        
        return self.model
```

### Knowledge Distillation

```python
class KnowledgeDistillation:
    """Distill knowledge from teacher to student"""
    
    def __init__(self, teacher, student, temperature=3.0, alpha=0.5):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha  # Balance between hard and soft labels
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
    def distillation_loss(self, student_logits, teacher_logits, labels):
        """Combined distillation loss"""
        # Soft targets (KL divergence)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Hard targets (cross entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
    
    def train_step(self, batch):
        """Training step with distillation"""
        # Teacher predictions
        with torch.no_grad():
            teacher_logits = self.teacher(batch['input'])
        
        # Student predictions
        student_logits = self.student(batch['input'])
        
        # Loss
        loss = self.distillation_loss(
            student_logits,
            teacher_logits,
            batch['label']
        )
        
        # Backward
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
```

## Monitoring and Debugging

### Training Monitor

```python
class TrainingMonitor:
    """Monitor training progress"""
    
    def __init__(self):
        self.metrics = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
            'grad_norm': [],
            'activation_stats': []
        }
        
    def log_step(self, model, loss, lr):
        """Log training step"""
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(lr)
        
        # Gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.metrics['grad_norm'].append(total_norm)
        
        # Check for issues
        self.check_health(loss, total_norm)
    
    def check_health(self, loss, grad_norm):
        """Check for training issues"""
        # NaN check
        if np.isnan(loss):
            raise ValueError("NaN detected in loss!")
        
        # Exploding gradients
        if grad_norm > 100:
            print(f"Warning: Large gradient norm {grad_norm:.2f}")
        
        # Vanishing gradients
        if grad_norm < 1e-7:
            print(f"Warning: Vanishing gradients {grad_norm:.2e}")
        
        # Loss plateau
        if len(self.metrics['loss']) > 100:
            recent = self.metrics['loss'][-100:]
            if np.std(recent) < 1e-5:
                print("Warning: Loss plateau detected")
```

### Profiling

```python
class TrainingProfiler:
    """Profile training performance"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        
    def profile_step(self, model, batch):
        """Profile a training step"""
        torch.cuda.synchronize()
        
        # Data loading
        start = time.time()
        inputs = batch['input'].cuda()
        targets = batch['target'].cuda()
        torch.cuda.synchronize()
        self.timings['data_transfer'].append(time.time() - start)
        
        # Forward
        start = time.time()
        outputs = model(inputs)
        torch.cuda.synchronize()
        self.timings['forward'].append(time.time() - start)
        
        # Loss
        start = time.time()
        loss = F.cross_entropy(outputs, targets)
        torch.cuda.synchronize()
        self.timings['loss'].append(time.time() - start)
        
        # Backward
        start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        self.timings['backward'].append(time.time() - start)
        
        # Optimizer
        start = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        self.timings['optimizer'].append(time.time() - start)
    
    def report(self):
        """Generate profiling report"""
        print("\nTraining Profile:")
        print("-" * 40)
        
        for name, times in self.timings.items():
            avg = np.mean(times)
            std = np.std(times)
            print(f"{name:20s}: {avg*1000:6.2f} ms ± {std*1000:6.2f} ms")
        
        total = sum(np.mean(t) for t in self.timings.values())
        print("-" * 40)
        print(f"{'Total':20s}: {total*1000:6.2f} ms")
```

## Best Practices

### 1. Reproducibility

```python
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 2. Checkpointing

```python
class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, model, optimizer, save_dir):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.best_loss = float('inf')
        
    def save(self, epoch, loss, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        
        # Save latest
        torch.save(checkpoint, f'{self.save_dir}/latest.pt')
        
        # Save best
        if is_best or loss < self.best_loss:
            self.best_loss = loss
            torch.save(checkpoint, f'{self.save_dir}/best.pt')
        
        # Save periodic
        if epoch % 10 == 0:
            torch.save(checkpoint, f'{self.save_dir}/epoch_{epoch}.pt')
    
    def load(self, path):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
```

### 3. Gradient Clipping

```python
def clip_gradients(model, max_norm=1.0, norm_type=2):
    """Clip gradients by norm"""
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )
    return total_norm
```

## Conclusion

Training deep learning models at scale requires careful attention to data pipelines, distributed training strategies, optimization techniques, and monitoring. Tesla's approach combines massive data collection, sophisticated training infrastructure (Dojo), and continuous iteration to improve autonomous driving models.

## References

- "Accurate, Large Minibatch SGD" (Facebook, 2017)
- "Mixed Precision Training" (NVIDIA, 2017)
- "Quantization and Training of Neural Networks" (Google, 2018)
- Tesla AI Day presentations

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~3500 words  
**Size**: ~20KB
