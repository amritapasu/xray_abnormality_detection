import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product

#--------------------------------------------PARAMETERS------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_root = '/central/groups/CS156b/2025/CodeMonkeys/input_images'
target_col = 'Lung Opacity'
image_root_dir = "input_images/train"
model_save_dir = 'maya_models/grid_search'
os.makedirs(model_save_dir, exist_ok=True)

end_df = None
uncertain_weight_factor = 0.25
neg_cutoff = 0.25
pos_cutoff = 0.75

num_epochs = 5  # Shorter for grid search
freeze_until = 4
batch_size = 16

#--------------------------------------------DATASET------------------------------------------------------#

class CSVDataset(Dataset):
    def __init__(self, dataframe, image_root_dir, target_columns=None, transform=None, save_dir=None, use_saved_images=False):
        self.data = dataframe
        self.image_root_dir = image_root_dir
        self.target_columns = target_columns
        self.transform = transform
        self.save_dir = save_dir
        self.use_saved_images = use_saved_images
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_index = row['Unnamed: 0']
        saved_image_path = os.path.join(self.save_dir, f"{image_index}.pt")
        if self.use_saved_images and os.path.exists(saved_image_path):
            image_tensor = torch.load(saved_image_path)
        else:
            original_image_path = os.path.join(self.image_root_dir, row['Path'])
            image = Image.open(original_image_path).convert("L")
            image_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)
            if self.save_dir:
                torch.save(image_tensor, saved_image_path)
        labels = pd.to_numeric(row[self.target_columns], errors='coerce').fillna(0).astype(float).values
        return image_tensor, torch.tensor(labels, dtype=torch.float32)

#--------------------------------------------MODELS------------------------------------------------------#

class MultiLabelResNet50(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x): return self.base_model(x)

class MultiLabelDenseNet121(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )
    def forward(self, x): return self.base_model(x)

def freeze_base_layers(model, until_layer=6):
    child_counter = 0
    for child in model.base_model.children():
        if child_counter < until_layer:
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1
    return model

#--------------------------------------------PREPROCESSING------------------------------------------------------#

def get_filtered_df(col, num=None):
    df = pd.read_csv('train2023.csv')
    if num:
        df = df.iloc[:num]
    df = df.dropna(subset=[col]).copy()
    df[col] = (df[col] + 1) / 2
    return df

filtered_df = get_filtered_df(target_col, num=end_df)
target_columns = [target_col]

train_df, val_df = train_test_split(filtered_df, test_size=0.15, random_state=42)
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = CSVDataset(train_df, image_root, target_columns, transform, save_dir=os.path.join(image_root, 'train'), use_saved_images=True)
val_dataset = CSVDataset(val_df, image_root, target_columns, transform, save_dir=os.path.join(image_root, 'train'), use_saved_images=True)

lo_labels = train_df[target_col].values
label_map = {0.0: 0, 0.5: 1, 1.0: 2}
mapped_labels = np.array([label_map[float(lbl)] for lbl in lo_labels])
class_counts = np.bincount(mapped_labels)
weights = 1. / (class_counts + 1e-6)
sample_weights = torch.tensor(weights[mapped_labels], dtype=torch.float)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class_weights = {target_col: {0: 1.0, 0.5: uncertain_weight_factor, 1: 1.0}}

criterion = nn.MSELoss(reduction='none')

def masked_MSE_loss(output, target, class_weights):
    mask = ~torch.isnan(target)
    loss = criterion(output, target)
    for class_idx, col in enumerate(target_columns):
        class_values = target[:, class_idx]
        weight = torch.tensor([class_weights[col].get(x.item(), 1) for x in class_values], dtype=torch.float32, device=output.device)
        loss = loss * mask
        loss[:, class_idx] *= weight
    return loss.sum() / mask.sum()

#--------------------------------------------TRAINING------------------------------------------------------#

def train_one_model(model_type, hidden_size, lr, weight_decay):
    if model_type == "resnet50":
        model = MultiLabelResNet50(num_classes=1, hidden_size=hidden_size).to(device)
        model = freeze_base_layers(model, until_layer=freeze_until)
    else:
        model = MultiLabelDenseNet121(num_classes=1, hidden_size=hidden_size).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = masked_MSE_loss(outputs, labels, class_weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.where(outputs < neg_cutoff, 0.0, torch.where(outputs < pos_cutoff, 0.5, 1.0))
            correct += (preds == labels).sum().item()
            total += labels.numel()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = masked_MSE_loss(outputs, labels, class_weights)
                val_loss += loss.item()
                preds = torch.where(outputs < neg_cutoff, 0.0, torch.where(outputs < pos_cutoff, 0.5, 1.0))
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        print(f"{model_type} H{hidden_size} LR{lr} WD{weight_decay} | Epoch {epoch+1}: ValLoss={avg_val_loss:.4f}, ValAcc={val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(model_save_dir, f"{model_type}_h{hidden_size}_lr{lr}_wd{weight_decay}.pth"))
        else:
            patience_counter += 1
        if patience_counter >= 2: break

    return best_val_loss, val_accuracy

#--------------------------------------------GRID SEARCH------------------------------------------------------#

def grid_search():
    model_types = ['resnet50', 'densenet121']
    hidden_sizes = [256, 384, 512]
    lrs = [1e-4, 5e-4]
    wds = [1e-5, 1e-4]
    frozen_layers = [3, 4, 5]
    dropouts = [0.2, 0.4, 0.6]

    best_loss = float('inf')
    best_config = None
    results = []

    for m, h, lr, wd, fl, ds in product(model_types, hidden_sizes, lrs, wds):
        val_loss, val_acc = train_one_model(m, h, lr, wd)
        results.append((m, h, lr, wd, val_loss, val_acc))
        if val_loss < best_loss:
            best_loss = val_loss
            best_config = (m, h, lr, wd)

    print(f"\n✅ Best Config: {best_config} with ValLoss {best_loss:.4f}")
    pd.DataFrame(results, columns=["Model", "Hidden", "LR", "WD", "ValLoss", "ValAcc"]).to_csv(
        os.path.join(model_save_dir, "grid_results.csv"), index=False
    )

#--------------------------------------------RUN------------------------------------------------------#

if __name__ == "__main__":
    grid_search()
