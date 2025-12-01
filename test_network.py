"""
Test script để kiểm tra Trainer class của bạn với dữ liệu giả
"""
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# Import code CỦA BẠN
from model import CNN
from utils import RainfallLoss
from lr_schedule import MyScheduler
from train import Trainer  # Import Trainer class của bạn


class DummyRainfallDataset(Dataset):
    """Dataset giả để test"""

    def __init__(self, num_samples=500, sat_channels=13, met_channels=8, img_size=64):
        self.num_samples = num_samples
        torch.manual_seed(42)

        print(f"Creating dummy dataset with {num_samples} samples...")
        self.sat_data = torch.randn(num_samples, sat_channels, img_size, img_size)
        self.met_data = torch.randn(num_samples, met_channels, img_size, img_size)
        self.targets = torch.rand(num_samples, 1, img_size, img_size) * 50  # 0-50mm

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.sat_data[idx], self.met_data[idx], self.targets[idx]


def main():
    print("=" * 70)
    print("TESTING YOUR TRAINER WITH DUMMY DATA")
    print("=" * 70)

    # ========== CONFIGURATION ==========
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 32
    VAL_RATIO = 0.2
    EPOCHS = 3
    GRADIENT_ACCUMULATION_STEPS = 4

    # Training params
    learning_rate = 1e-3
    weight_decay = 1e-5
    log_step = 5  # Log mỗi 5 updates
    val_step = 10  # Validate mỗi 10 updates
    model_save_path = 'test_best_model.pth'

    # Scheduler params
    SCHEDULER_TYPE = 'cosine'
    WARMUP_RATIO = 0.1
    GAMMA = 0.95

    print(f"\nDevice: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {EPOCHS}")

    # ========== CREATE DUMMY DATA ==========
    print("\n" + "=" * 70)
    print("STEP 1: Creating Dataset")
    print("=" * 70)

    dataset = DummyRainfallDataset(num_samples=500)

    # Split train/val
    val_size = int(VAL_RATIO * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False)

    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # ========== CREATE MODEL ==========
    print("\n" + "=" * 70)
    print("STEP 2: Creating Model")
    print("=" * 70)

    model = CNN(in_channels_sat=13, in_channels_met=8, base_filters=32)
    print("✓ Model created")

    # ========== CREATE TRAINING COMPONENTS ==========
    print("\n" + "=" * 70)
    print("STEP 3: Setting up Training Components")
    print("=" * 70)

    criterion = RainfallLoss(alpha=0.7)
    print("✓ Loss function created")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("✓ Optimizer created")

    # Tính total updates cho scheduler
    total_batches = len(train_loader)
    total_updates = (total_batches + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS * EPOCHS

    scheduler = MyScheduler(
        optimizer,
        total_steps=total_updates,
        scheduler_type=SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        gamma=GAMMA
    )
    print(f"✓ Scheduler created (type: {SCHEDULER_TYPE})")
    print(f"  Total updates: {total_updates}")
    print(f"  Warmup steps: {int(total_updates * WARMUP_RATIO)}")

    # ========== CREATE YOUR TRAINER ==========
    print("\n" + "=" * 70)
    print("STEP 4: Creating YOUR Trainer")
    print("=" * 70)

    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        log_step=log_step,
        val_step=val_step,
        model_save_path=model_save_path
    )
    print("✓ Trainer initialized")

    # ========== START TRAINING ==========
    print("\n" + "=" * 70)
    print("STEP 5: Start Training")
    print("=" * 70)

    try:
        trained_model, history = trainer.train(
            num_epochs=EPOCHS,
            accumulate_steps=GRADIENT_ACCUMULATION_STEPS,
            save_by='rmse'  # Save best model theo RMSE
        )

        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        # ========== PRINT RESULTS ==========
        print("\nTraining History:")
        print("-" * 70)
        for epoch in range(len(history['train_loss'])):
            print(f"Epoch {epoch + 1}:")
            print(f"  Train Loss: {history['train_loss'][epoch]:.6f}")
            if history['val_loss']:
                print(f"  Val Loss:   {history['val_loss'][epoch]:.6f}")
                print(f"  RMSE:       {history['val_rmse'][epoch]:.4f}")
                print(f"  MAE:        {history['val_mae'][epoch]:.4f}")
                print(f"  Corr:       {history['val_corr'][epoch]:.4f}")

        print("\n" + "=" * 70)
        print(f"Best model saved to: {model_save_path}")
        print("=" * 70)

        return True

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TRAINING FAILED!")
        print("=" * 70)
        print(f"\nError Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 Your Trainer works perfectly with dummy data!")
        print("Now you can use it with real data.")
    else:
        print("\n⚠️  Please fix the errors above before using real data.")