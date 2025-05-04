import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Dataset Class
class MultiModalMRIDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list  # List of dicts: {'T1': path, 'T2': path, 'FLAIR': path}
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        modalities = self.file_list[idx]
        image_T1 = nib.load(modalities['T1']).get_fdata()
        image_T2 = nib.load(modalities['T2']).get_fdata()
        image_FLAIR = nib.load(modalities['FLAIR']).get_fdata()

        image = np.stack([image_T1, image_T2, image_FLAIR], axis=0)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Model Definition
class Multimodal3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Multimodal3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 16 * 16 * 16, 128)  # Adjust shape if needed
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Training Function
def train_model(model, dataset, epochs=10, batch_size=2, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(dataloader):
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(dataloader):.4f}")
    return model

# Evaluation Function
def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            preds.append(predicted.cpu().item())
            trues.append(labels.item())

    print(classification_report(trues, preds))

# Main Script
if __name__ == '__main__':
    # Dummy file list and labels (replace with actual .nii.gz paths and labels)
    file_list = [
        {'T1': 'data/patient1_T1.nii.gz', 'T2': 'data/patient1_T2.nii.gz', 'FLAIR': 'data/patient1_FLAIR.nii.gz'},
        {'T1': 'data/patient2_T1.nii.gz', 'T2': 'data/patient2_T2.nii.gz', 'FLAIR': 'data/patient2_FLAIR.nii.gz'}
    ]
    labels = [0, 1]  # Example labels

    dataset = MultiModalMRIDataset(file_list, labels)
    model = Multimodal3DCNN(num_classes=2).cuda()

    trained_model = train_model(model, dataset, epochs=10)
    evaluate_model(trained_model, dataset)

model = Multimodal3DCNN(num_classes=2).to(device)