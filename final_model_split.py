import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.optim as optim
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#--------------------------------------------PARAMETERS------------------------------------------------------#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_root = '/central/groups/CS156b/2025/CodeMonkeys/input_images'

target_col = 'Lung Opacity' if os.getenv('TARGET_COL')   is None else os.getenv('TARGET_COL')
image_root_dir = "input_images/train" if os.getenv('IMAGE_ROOT_DIR')   is None else os.getenv('IMAGE_ROOT_DIR')
model_save_dir = 'amb_models/lo_full' if os.getenv('SAVE_DIR')   is None else os.getenv('SAVE_DIR')

frontal_status = True if os.getenv('FRONTAL_STATUS')   is None else bool(os.getenv('FRONTAL_STATUS'))
end_df = None if os.getenv('END_DF')   is None else int(os.getenv('END_DF'))

uncertain_weight_factor = 0.25 if os.getenv('UNCERTAIN_WF')   is None else float(os.getenv('UNCERTAIN_WF'))
neg_cutoff = 0.25 if os.getenv('NEG_CUTOFF')   is None else float(os.getenv('NEG_CUTOFF'))
pos_cutoff = 0.75 if os.getenv('POS_CUTOFF')   is None else float(os.getenv('POS_CUTOFF'))

num_epochs = 30 if os.getenv('NUM_EPOCHS')   is None else int(os.getenv('NUM_EPOCHS'))
hidden_size = 512 if os.getenv('HIDDEN_SIZE')   is None else int(os.getenv('HIDDEN_SIZE'))
lr = 1e-4 if os.getenv('LR')   is None else float(os.getenv('LR'))
weight_decay = 1e-4 if os.getenv('WEIGHT_DECAY')   is None else float(os.getenv('WEIGHT_DECAY'))
freeze_until = 4 if os.getenv('FREEZE_UNTIL')   is None else int(os.getenv('FREEZE_UNTIL'))

train_save_dir = os.path.join(image_root, 'train_lateral')

#--------------------------------------------FUNCTIONS------------------------------------------------------#

class CSVDataset(Dataset):
    def __init__(self, dataframe, image_root_dir, target_columns=None, transform=None,
                 save_dir=None, use_saved_images=False):
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
        
        # Use index for the saved tensor filename
        image_index = row['Unnamed: 0']
        saved_image_path = os.path.join(self.save_dir, f"{image_index}.pt")

        if self.use_saved_images:
            if os.path.exists(saved_image_path):
                image_tensor = torch.load(saved_image_path)
            else:
                raise FileNotFoundError(f"Saved tensor not found: {saved_image_path}")
        else:
            original_image_path = os.path.join(self.image_root_dir, row['Path'])
            image = Image.open(original_image_path).convert("L")
            image_tensor = self.transform(image) if self.transform else transforms.ToTensor()(image)

            if self.save_dir:
                torch.save(image_tensor, saved_image_path)

        if self.target_columns:
            labels = pd.to_numeric(row[self.target_columns], errors='coerce').fillna(0).astype(float).values
            labels = torch.tensor(labels, dtype=torch.float32)
            return image_tensor, labels

        return image_tensor

class MultiLabelResNet50(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(MultiLabelResNet50, self).__init__()
        
        # Load pre-trained ResNet50
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the fully connected layer for multi-label classification
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, hidden_size),  # 512, New intermediate layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout to prevent overfitting
            nn.Linear(hidden_size, num_classes),  # Output layer
            nn.Sigmoid()  # Sigmoid for multi-label classification (soften the data)
            #nn.Tanh()  #This is between -1 and 1

           # nn.Linear(self.base_model.fc.in_features, num_classes),
           # nn.Sigmoid()  # Sigmoid activation for multi-label classification
        )

    def forward(self, x):
        return self.base_model(x)

class MultiLabelDenseNet121(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super(MultiLabelDenseNet121, self).__init__()

        # Load pre-trained DenseNet-121
        self.base_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        # Replace the classifier with a custom head
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier.in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

def get_filtered_df(col, num=None, frontal_status=True):
    full_train_df = pd.read_csv('train2023.csv')

    if num is not None:
        full_train_df = full_train_df.iloc[:num]

    # Filter for Frontal if specified
    #if frontal_status:
        #full_train_df = full_train_df[full_train_df['Frontal/Lateral'] == 'Frontal']
    #else:
    full_train_df = full_train_df[full_train_df['Frontal/Lateral'] == 'Lateral']

    # Drop rows with missing label in the given column
    filtered_train_df = full_train_df.dropna(subset=[col]).copy()

    # Normalize label values from {-1, 0, 1} to {0, 0.5, 1}
    filtered_train_df[col] = (filtered_train_df[col] + 1) / 2

    return filtered_train_df

def freeze_base_layers(model, until_layer=6):
    """
    Freeze layers of ResNet-50 up to a certain stage (e.g., until_layer=6 means keep layers 0-5 frozen).
    """
    child_counter = 0
    for child in model.base_model.children():
        if child_counter < until_layer:
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1
    return model

def masked_MSE_loss(output, target, class_weights):
    # Create a mask for non-NaN target values
    mask = ~torch.isnan(target)
    
    # Apply the MSE loss
    loss = criterion(output, target)
    
    # Loop through each class and apply the class weights
    for class_idx, col in enumerate(target_columns):
        # Get the class values for the current class
        class_values = target[:, class_idx]
        
        # Apply the class weights to each class value
        weight = torch.tensor([class_weights[col].get(x.item(), 1) for x in class_values], dtype=torch.float32, device=output.device)
        
        # Apply the weight to the loss (broadcast the weight to match the loss shape)
        loss = loss * mask  # Apply mask to exclude NaN targets
        loss[:, class_idx] *= weight  # Apply weight per class
    
    # Return mean loss for valid entries
    return loss.sum() / mask.sum()


#--------------------------------------------PREPROCESSING------------------------------------------------------#

filtered_train_df = get_filtered_df(target_col, num=end_df, frontal_status=frontal_status)
label_counts = filtered_train_df[target_col].value_counts()

# Define your target columns once
target_columns = [target_col]

# Step 1: Split the dataframe
train_df, val_df = train_test_split(filtered_train_df, test_size=0.15, random_state=42)

# Step 2: Create training dataset
train_dataset = CSVDataset(
    dataframe=train_df, 
    image_root_dir=image_root, 
    target_columns=target_columns, 
    save_dir=train_save_dir, 
    use_saved_images=True
)

# Step 3: Create validation dataset
val_dataset = CSVDataset(
    dataframe=val_df, 
    image_root_dir=image_root, 
    target_columns=target_columns, 
    save_dir=train_save_dir, 
    use_saved_images=True
)

# Step 4: Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

lo_labels = train_df[target_col].values
label_map = {0.0: 0, 0.5: 1, 1.0: 2}
mapped_labels = np.array([label_map[float(lbl)] for lbl in lo_labels])

class_counts = np.bincount(mapped_labels)
weights = 1. / (class_counts + 1e-6)
sample_weights = torch.tensor(weights[mapped_labels], dtype=torch.float)

sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

class_weights = {}

# Loop over each target column
for col in target_columns:
    # Count the occurrences of each class in the column
    counts = filtered_train_df[col].value_counts()
    total = len(filtered_train_df[col])
    
    # Calculate class weights using inverse frequency (you can also experiment with other strategies)
    weights = {
        0: total / (counts.get(0, 0) + 1),  # Add 1 to avoid division by zero
        0.5: total / (counts.get(0.5, 0) + 1) * uncertain_weight_factor,
        1: total / (counts.get(1, 0) + 1)
    }
    
    # Store weights for each class
    class_weights[col] = weights

class_weights[target_col] = {0: 1.0, 0.5: uncertain_weight_factor, 1: 1.0}  ##GET RID OF THIS LINE IF DONT HAVE SAMPLER

criterion = nn.MSELoss(reduction='none')

#-------------------------------------------TRAINING LOOP-------------------------------------------------------#

# Prepare directory and log file
os.makedirs(model_save_dir, exist_ok=True)
log_file_path = os.path.join(model_save_dir, "training_log.txt")
conf_path = os.path.join(model_save_dir, "conf_matrices/")
os.makedirs(conf_path, exist_ok=True)

with open(log_file_path, 'w') as f:
    f.write(f"Filtered train DataFrame length: {len(filtered_train_df)}\n")

# Hyperparameters and model setup
num_classes = 1  # Predicting 'Pleural Effusion'
model = MultiLabelResNet50(num_classes=num_classes, hidden_size=hidden_size).to(device)
model = freeze_base_layers(model, until_layer=freeze_until)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = masked_MSE_loss(outputs, labels, class_weights)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted_class = torch.where(
            outputs < neg_cutoff, torch.tensor(0.0).to(device),
            torch.where(
                outputs < pos_cutoff, torch.tensor(0.5).to(device),
                torch.tensor(1.0).to(device)
            )
        )

        correct += (predicted_class == labels).sum().item()
        total += labels.numel()

    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    with open(log_file_path, "a") as f:
        f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}\n")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = masked_MSE_loss(outputs, labels, class_weights)
            val_loss += loss.item()

            predicted_class = torch.where(
                outputs < neg_cutoff, torch.tensor(0.0).to(device),
                torch.where(
                    outputs < pos_cutoff, torch.tensor(0.5).to(device),
                    torch.tensor(1.0).to(device)
                )
            )

            all_preds.append(predicted_class.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            val_correct += (predicted_class == labels).sum().item()
            val_total += labels.numel()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    with open(log_file_path, "a") as f:
        f.write(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")

    # Convert float predictions to int for confusion matrix
    float_to_int = {0.0: 0, 0.5: 1, 1.0: 2}
    all_preds_np = np.array([float_to_int[val] for val in np.concatenate(all_preds).flatten()])
    all_labels_np = np.array([float_to_int[val] for val in np.concatenate(all_labels).flatten()])

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels_np, all_preds_np, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0.0 (Neg)", "0.5 (Unc)", "1.0 (Pos)"],
                yticklabels=["0.0 (Neg)", "0.5 (Unc)", "1.0 (Pos)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Validation Confusion Matrix (Epoch {epoch+1})")

    cm_path = os.path.join(conf_path, f"epoch_{epoch+1}.png")
    plt.savefig(cm_path)
    plt.close()

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(model_save_dir, "best_model.pth"))
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        with open(log_file_path, "a") as f:
            f.write("⛔ Early stopping triggered.\n")
        break

    torch.save(model.state_dict(), os.path.join(model_save_dir, f"epoch_{epoch+1}.pth"))