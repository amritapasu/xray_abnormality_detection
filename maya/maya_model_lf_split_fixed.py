import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────────────────────────────────────────────────────────────────────────────
# 1) PARAMETERS
# ────────────────────────────────────────────────────────────────────────────────
image_root        = '/central/groups/CS156b/2025/CodeMonkeys/input_images'
target_col        = 'Pneumonia'
frontal_status    = True
end_df            = 10000
num_epochs        = 30
hidden_size       = 512
lr                = 1e-4
weight_decay      = 1e-4
train_save_dir    = os.path.join(image_root, 'train_saved_tensors')
model_save_dir    = os.path.join(os.getcwd(), 'maya_models', 'pn_front_mse_matrix')
os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)
print("Saving to:", model_save_dir)

uncertain_weight_factor = 0.2  # for computing class_weights
neg_cutoff = 0.4
pos_cutoff = 0.6

# ────────────────────────────────────────────────────────────────────────────────
# 2) DATASET: returns (image_tensor, float_label_tensor)
# ────────────────────────────────────────────────────────────────────────────────
class CSVDatasetMSE(Dataset):
    def __init__(self, dataframe, image_root_dir, target_column,
                 transform=None, save_dir=None, use_saved_images=False):
        super().__init__()
        self.data       = dataframe.reset_index(drop=True)
        self.image_root = image_root_dir
        self.target_col = target_column
        self.transform  = transform
        self.save_dir   = save_dir
        self.use_saved  = use_saved_images

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_index = row['Unnamed: 0']
        saved_tensor_path = os.path.join(self.save_dir, f"{img_index}.pt")

        if self.use_saved:
            if os.path.exists(saved_tensor_path):
                img_tensor = torch.load(saved_tensor_path)
            else:
                raise FileNotFoundError(f"Saved tensor not found: {saved_tensor_path}")
        else:
            img_path = os.path.join(self.image_root, row['Path'])
            img = Image.open(img_path).convert("L")
            to_tensor = transforms.ToTensor()
            one_ch = to_tensor(img)                # shape = (1, H, W)
            three_ch = one_ch.repeat(3, 1, 1)       # shape = (3, H, W)

            if self.transform:
                img_tensor = self.transform(three_ch)
            else:
                img_tensor = three_ch

            if self.save_dir:
                torch.save(img_tensor, saved_tensor_path)

        float_label = float(row[self.target_col])  # in {0.0, 0.5, 1.0}
        label_tensor = torch.tensor([float_label], dtype=torch.float32)
        return img_tensor, label_tensor

# ────────────────────────────────────────────────────────────────────────────────
# 3) LOAD & FILTER THE DATAFRAME
# ────────────────────────────────────────────────────────────────────────────────
def get_filtered_df(col, num=None, frontal_status=True):
    full_df = pd.read_csv('train2023.csv')
    full_df = full_df[full_df['Frontal/Lateral'] == ('Frontal' if frontal_status else 'Lateral')]
    filtered = full_df.dropna(subset=[col]).copy()
    filtered[col] = (filtered[col] + 1) / 2   # map {–1→0, 0→0.5, 1→1}
    if num is not None:
        filtered = filtered.iloc[:num]
    return filtered

filtered_df = get_filtered_df(target_col, num=end_df, frontal_status=frontal_status)
print("Label distribution (raw counts):")
print(filtered_df[target_col].value_counts())

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
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_ds = CSVDatasetMSE(
    dataframe=train_df,
    image_root_dir=image_root,
    target_column=target_col,
    transform=train_transform,
    save_dir=train_save_dir,
    use_saved_images=True
)
val_ds = CSVDatasetMSE(
    dataframe=val_df,
    image_root_dir=image_root,
    target_column=target_col,
    transform=val_transform,
    save_dir=train_save_dir,
    use_saved_images=True
)

# WeightedRandomSampler to balance classes
train_labels_float = train_df[target_col].values
train_labels_idx   = np.array([0 if x==0.0 else (1 if x==0.5 else 2) for x in train_labels_float])
class_counts = np.bincount(train_labels_idx, minlength=3)
inv_freq = 1.0 / (class_counts + 1e-6)
sample_weights = inv_freq[train_labels_idx]
sampler = WeightedRandomSampler(
    weights=torch.from_numpy(sample_weights).float(),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    sampler=sampler,
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_ds,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ────────────────────────────────────────────────────────────────────────────────
# 5) MODEL: single-output DenseNet121 + Sigmoid
# ────────────────────────────────────────────────────────────────────────────────
class PneumoniaDenseNet(nn.Module):
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # Freeze everything except DenseBlock4 and classifier head
        for name, param in self.backbone.named_parameters():
            if name.startswith("features.denseblock4") or name.startswith("classifier"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()    # ensures output ∈ [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

model = PneumoniaDenseNet(hidden_size=hidden_size).to(device)

# ────────────────────────────────────────────────────────────────────────────────
# 6) DEFINE masked_MSE_loss
# ────────────────────────────────────────────────────────────────────────────────
counts = filtered_df[target_col].value_counts()
total  = len(filtered_df)
w0  = total / (counts.get(0.0, 0) + 1)
w05 = total / (counts.get(0.5, 0) + 1) * uncertain_weight_factor
w1  = total / (counts.get(1.0, 0) + 1)
max_ratio = 5
max_w = max(w0, w05, w1)
min_w = max_w / max_ratio
class_weights = {
    target_col: {
        0.0: max(min_w, w0),
        0.5: max(min_w, w05),
        1.0: max(min_w, w1)
    }
}

criterion = nn.MSELoss(reduction='none')

def masked_MSE_loss(output: torch.Tensor, target: torch.Tensor, class_weights: dict) -> torch.Tensor:
    """
    output: (B,1) ∈ [0,1]
    target: (B,1) ∈ {0.0, 0.5, 1.0}
    """
    mask = ~torch.isnan(target)   # shape (B,1)
    loss = criterion(output, target)  # (B,1)

    class_vals = target[:,0]  # (B,)
    weight_vec = torch.tensor(
        [class_weights[target_col].get(x.item(), 1.0) for x in class_vals],
        dtype=torch.float32,
        device=output.device
    )
    loss = loss * mask                # zero out any NaN rows (none here)
    loss[:,0] *= weight_vec           # apply per-sample weight

    return loss.sum() / mask.sum()

# ────────────────────────────────────────────────────────────────────────────────
# 7) OPTIMIZER
# ────────────────────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=weight_decay
)

# ────────────────────────────────────────────────────────────────────────────────
# 8) TRAINING LOOP (with confusion matrices & F1)
# ────────────────────────────────────────────────────────────────────────────────
best_val_loss    = float('inf')
early_stop_pat   = 5
patience_counter = 0

log_file_path = os.path.join(model_save_dir, "training_log_mse.txt")
conf_path     = os.path.join(model_save_dir, "conf_matrices_mse")
os.makedirs(conf_path, exist_ok=True)

with open(log_file_path, 'w') as f:
    f.write(f"Num train samples: {len(train_ds)}, Num val samples: {len(val_ds)}\n")
    f.write(f"Class weights: {class_weights[target_col]}\n\n")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # ── Training ─────────────────────────────────────────────────────────────
    for images, labels in train_loader:
        images = images.to(device)   # (B,3,224,224)
        labels = labels.to(device)   # (B,1)

        optimizer.zero_grad()
        preds = model(images)        # (B,1) in [0,1]
        loss = masked_MSE_loss(preds, labels, class_weights)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_ds)

    # ── Validation ───────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)        # (B,1)
            loss = masked_MSE_loss(preds, labels, class_weights)
            val_loss += loss.item() * images.size(0)

            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_ds)

    # Concatenate predictions and labels
    flat_preds  = np.concatenate(all_preds)
    flat_labels = np.concatenate(all_labels)

    # Convert float predictions to discrete classes {0,1,2} using cutoffs
    pred_classes = np.where(
        flat_preds < neg_cutoff, 0,
        np.where(flat_preds < pos_cutoff, 1, 2)
    )
    # Convert float targets to discrete classes
    label_classes = np.array([0 if x==0.0 else (1 if x==0.5 else 2) for x in flat_labels])

    # Compute confusion matrix and classification report
    cm = confusion_matrix(label_classes, pred_classes, labels=[0,1,2])
    report = classification_report(
        label_classes,
        pred_classes,
        labels=[0,1,2],
        target_names=["Neg(0)", "Unc(1)", "Pos(2)"],
        zero_division=0,
        output_dict=False
    )

    # Log epoch results
    with open(log_file_path, 'a') as f:
        f.write(f"Epoch {epoch+1}/{num_epochs}  TrainLoss: {epoch_loss:.4f}  ValLoss: {avg_val_loss:.4f}\n")
        f.write(report + "\n")

    # Save confusion matrix figure
    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Neg(0)", "Unc(1)", "Pos(2)"],
        yticklabels=["Neg(0)", "Unc(1)", "Pos(2)"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Validation Confusion Matrix (Epoch {epoch+1})")
    plt.tight_layout()
    cm_path = os.path.join(conf_path, f"epoch_{epoch+1}_cm.png")
    plt.savefig(cm_path)
    plt.close()

    # ── Early stopping ─────────────────────────────────────────────────────────
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(model_save_dir, "best_mse_model.pth"))
    else:
        patience_counter += 1

    if patience_counter >= early_stop_pat:
        with open(log_file_path, 'a') as f:
            f.write("⛔ Early stopping triggered.\n")
        break

    # Optionally save per-epoch checkpoint
    torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch+1}_mse.pth"))

print("MSE training complete. Best val MSE: %.4f" % best_val_loss)
