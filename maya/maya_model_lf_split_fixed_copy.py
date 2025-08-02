import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────────
# 1) PARAMETERS
# ────────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_root = '/central/groups/CS156b/2025/CodeMonkeys/input_images'

target_col = 'Lung Opacity' if os.getenv('TARGET_COL') is None else os.getenv('TARGET_COL')
frontal_status = True   # set to False if you want Lateral, but here we assume frontal
end_df = None           # or int(os.getenv('END_DF'))
num_epochs = 30
hidden_size = 512
lr = 1e-4
weight_decay = 1e-4

train_save_dir = os.path.join(image_root, 'train_saved_tensors')
os.makedirs(train_save_dir, exist_ok=True)

model_save_dir = os.path.join(os.getcwd(), 'maya_models', 'lo_front_3way')
os.makedirs(model_save_dir, exist_ok=True)
print("Saving to:", model_save_dir)

# Cutoffs for mapping sigmoid outputs → {Neg/Unc/Pos}
neg_cutoff = 0.4
pos_cutoff = 0.6

# ────────────────────────────────────────────────────────────────────────────────
# 2) DATASET & MODELS
# ────────────────────────────────────────────────────────────────────────────────
class CSVDataset3Way(Dataset):
    def __init__(self, dataframe, image_root_dir, target_column, transform=None,
                 save_dir=None, use_saved_images=False):
        super().__init__()
        self.data = dataframe.reset_index(drop=True)
        self.image_root = image_root_dir
        self.target_col = target_column
        self.transform = transform
        self.save_dir = save_dir
        self.use_saved = use_saved_images

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_index = row['Unnamed: 0']
        saved_tensor_path = os.path.join(self.save_dir, f"{img_index}.pt")

        # 1) Load (or save) a 3‐channel tensor
        if self.use_saved:
            if os.path.exists(saved_tensor_path):
                img_tensor = torch.load(saved_tensor_path)
            else:
                raise FileNotFoundError(f"Saved tensor not found: {saved_tensor_path}")
        else:
            # Open grayscale → replicate to 3 channels
            img_path = os.path.join(self.image_root, row['Path'])
            img = Image.open(img_path).convert("L")  # (1,H,W)

            to_tensor = transforms.ToTensor()
            one_ch = to_tensor(img)           # (1,H,W)
            three_ch = one_ch.repeat(3, 1, 1)  # (3,H,W)

            if self.transform:
                img_tensor = self.transform(three_ch)
            else:
                img_tensor = three_ch

            if self.save_dir:
                torch.save(img_tensor, saved_tensor_path)

        # 2) Map label {0.0→Neg,0.5→Unc,1.0→Pos} → integer class {0,1,2}
        float_label = float(row[self.target_col])
        if float_label == 0.0:
            class_idx = 0
        elif float_label == 0.5:
            class_idx = 1
        else:  # == 1.0
            class_idx = 2

        return img_tensor, class_idx

class ThreeWayDenseNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        # Initially freeze everything; we'll unfreeze denseblock4 later
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False

        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(hidden_size, 3)
        )

    def forward(self, x):
        return self.backbone(x)

# ────────────────────────────────────────────────────────────────────────────────
# 3) LOAD & FILTER DATAFRAME
# ────────────────────────────────────────────────────────────────────────────────
def get_filtered_df(col, num=None, frontal_status=True):
    full_df = pd.read_csv('train2023.csv')

    # 1) Keep only Frontal or Lateral
    full_df = full_df[ full_df['Frontal/Lateral'] == ('Frontal' if frontal_status else 'Lateral') ]

    # 2) Drop rows where target is NaN and remap −1→0, 0→0.5, +1→1
    filtered = full_df.dropna(subset=[col]).copy()
    filtered[col] = (filtered[col] + 1) / 2  # now in {0.0,0.5,1.0}

    # 3) Slice to first <num> if requested
    if num is not None:
        filtered = filtered.iloc[:num]

    return filtered

filtered_df = get_filtered_df(target_col, num=end_df, frontal_status=frontal_status)
print("Label distribution (raw counts):\n", filtered_df[target_col].value_counts())

from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(
    filtered_df,
    test_size=0.15,
    random_state=42,
    stratify=filtered_df[target_col]
)
print("After split – train counts:", train_df[target_col].value_counts())
print("               val counts:",   val_df[target_col].value_counts())

# ────────────────────────────────────────────────────────────────────────────────
# 4) TRANSFORMS & DATALOADERS
# ────────────────────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.05),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_ds = CSVDataset3Way(
    dataframe=train_df,
    image_root_dir=image_root,
    target_column=target_col,
    transform=train_transform,
    save_dir=train_save_dir,
    use_saved_images=True
)
val_ds = CSVDataset3Way(
    dataframe=val_df,
    image_root_dir=image_root,
    target_column=target_col,
    transform=val_transform,
    save_dir=train_save_dir,
    use_saved_images=True
)

# Build an initial 1:1:1 sampler for epoch 0
train_labels = train_df[target_col].map({0.0:0, 0.5:1, 1.0:2}).values
raw_weights_equal = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float32)
sample_weights = raw_weights_equal[train_labels]

sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(sample_weights),
    replacement=True
)
train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=4, pin_memory=True)

# ────────────────────────────────────────────────────────────────────────────────
# 5) DEFINE MODEL, LOSS, OPTIMIZER
# ────────────────────────────────────────────────────────────────────────────────
model = ThreeWayDenseNet(hidden_size=hidden_size).to(device)

# We will reassign class_weights inside the loop; initialize with placeholders
class_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# ────────────────────────────────────────────────────────────────────────────────
# 6) TRAINING LOOP (3‐way classification) WITH “REBALANCING SCHEDULE”
# ────────────────────────────────────────────────────────────────────────────────
best_val_loss = float('inf')
patience = 8
pat_counter = 0

log_file_path = os.path.join(model_save_dir, "training_log.txt")
conf_path     = os.path.join(model_save_dir, "conf_matrices")
os.makedirs(conf_path, exist_ok=True)

with open(log_file_path, 'w') as f:
    f.write(f"Num train samples: {len(train_ds)}, Num val samples: {len(val_ds)}\n")
    f.write(f"Initial sampler = 1:1:1\n\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    # ─────── TRAIN ───────
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)             # (B,3)
        loss = criterion(logits, labels)   # weighted CE
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits.detach(), dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_ds)
    epoch_preds = np.concatenate(all_preds)
    epoch_labels = np.concatenate(all_labels)
    epoch_acc = (epoch_preds == epoch_labels).mean()

    train_report = classification_report(
        epoch_labels, epoch_preds,
        labels=[0,1,2],
        target_names=["Neg(0)", "Unc(1)", "Pos(2)"],
        output_dict=True
    )

    with open(log_file_path, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs} [Train]  "
                f"Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}\n")
        f.write("    F1 Neg: %.4f  F1 Unc: %.4f  F1 Pos: %.4f\n" % (
            train_report["Neg(0)"]["f1-score"],
            train_report["Unc(1)"]["f1-score"],
            train_report["Pos(2)"]["f1-score"],
        ))

    scheduler.step()

    # ── REBALANCING SCHEDULE ──
    # Epoch 0 → 1: unfreeze DenseBlock4, use [3,1,3] and 3:1 sampler
    if epoch == 0:
        # 1) Unfreeze DenseBlock4 & classifier
        for name, param in model.backbone.named_parameters():
            if name.startswith("features.denseblock4") or name.startswith("classifier"):
                param.requires_grad = True

        # 2) Increase LR to 1e-4 so DenseBlock4 sees strong gradients
        for g in optimizer.param_groups:
            g['lr'] = 1e-4

        # 3) Set class_weights = [3.0,1.0,3.0]
        class_weights = torch.tensor([3.0, 1.0, 3.0], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        # 4) Build a 3:1 Neg+Pos : Unc sampler
        oversample_weights = np.array([3.0, 1.0, 3.0], dtype=np.float32)
        sample_weights = oversample_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler,
                                  num_workers=4, pin_memory=True)

        with open(log_file_path, 'a') as f:
            f.write("\n-- Rebalanced for Epoch 1: class_weights=[3,1,3], sampler=3:1 Neg+Pos→Unc --\n\n")

    # Epoch 1 → 2: use [2,1,2], LR=5e-5
    if epoch == 1:
        # 1) Drop LR to 5e-5
        for g in optimizer.param_groups:
            g['lr'] = 5e-5

        # 2) Keep class_weights = [3,1,3] or soften slightly
        class_weights = torch.tensor([3.0, 1.0, 3.0], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        # 3) Build a 2:1 Neg+Pos : Unc sampler
        oversample_weights = np.array([2.0, 1.0, 2.0], dtype=np.float32)
        sample_weights = oversample_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler,
                                  num_workers=4, pin_memory=True)

        with open(log_file_path, 'a') as f:
            f.write("\n-- Rebalanced for Epoch 2: class_weights=[3,1,3], sampler=2:1 Neg+Pos→Unc --\n\n")

    # Epoch 2 onward: soften to [2,1,2], sampler=1.5:1, LR=2e-5
    if epoch == 2:
        # 1) Drop LR to 2e-5
        for g in optimizer.param_groups:
            g['lr'] = 2e-5

        # 2) Soften class_weights = [2,1,2]
        class_weights = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

        # 3) Build a 1.5:1 Neg+Pos : Unc sampler
        oversample_weights = np.array([1.5, 1.0, 1.5], dtype=np.float32)
        sample_weights = oversample_weights[train_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler,
                                  num_workers=4, pin_memory=True)

        with open(log_file_path, 'a') as f:
            f.write("\n-- Rebalanced for Epoch 3+: class_weights=[2,1,2], sampler=1.5:1 --\n\n")

    # ─────── VALIDATION ───────
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model(images)
            loss = criterion(logits, labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            val_preds.append(preds.cpu().numpy())
            val_labels.append(labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_ds)
    val_preds_np = np.concatenate(val_preds)
    val_labels_np = np.concatenate(val_labels)
    val_acc = (val_preds_np == val_labels_np).mean()

    val_report = classification_report(
        val_labels_np, val_preds_np,
        labels=[0,1,2],
        target_names=["Neg(0)", "Unc(1)", "Pos(2)"],
        output_dict=True
    )

    with open(log_file_path, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs} [Val]    "
                f"Loss: {avg_val_loss:.4f}  Acc: {val_acc:.4f}\n")
        f.write("    F1 Neg: %.4f  F1 Unc: %.4f  F1 Pos: %.4f\n\n" % (
            val_report["Neg(0)"]["f1-score"],
            val_report["Unc(1)"]["f1-score"],
            val_report["Pos(2)"]["f1-score"],
        ))

    # Confusion matrix
    cm = confusion_matrix(val_labels_np, val_preds_np, labels=[0,1,2])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Neg(0)", "Unc(1)", "Pos(2)"],
                yticklabels=["Neg(0)", "Unc(1)", "Pos(2)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Validation Confusion Matrix (Epoch {epoch+1})")
    plt.tight_layout()
    cm_path = os.path.join(conf_path, f"epoch_{epoch+1}.png")
    plt.savefig(cm_path)
    plt.close()

    # Early stopping on val loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        pat_counter = 0
        torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
    else:
        pat_counter += 1

    with open(log_file_path, 'a') as f:
        f.write(f"    Patience count: {pat_counter}/{patience}\n\n")

    if pat_counter >= patience:
        with open(log_file_path, 'a') as f:
            f.write("⛔ Early stopping triggered.\n")
        break

    # Save an epoch checkpoint
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch+1}.pth"))

print("Training complete. Best val loss: %.4f" % best_val_loss)
