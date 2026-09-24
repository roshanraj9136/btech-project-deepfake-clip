import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import timm
import time
import ctypes

class HF_CIFAR10_Dataset(Dataset):
    def __init__(self, hf_data, transform=None):
        self.hf_data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.hf_data)

    def __getitem__(self, idx):
        item = self.hf_data[idx]
        img = item['img']
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label

class SideBlock(nn.Module):
    def __init__(self, clip_dim=768, side_dim=768):
        super().__init__()
        self.down_proj = nn.Linear(clip_dim, side_dim)
        self.attn = nn.MultiheadAttention(embed_dim=side_dim, num_heads=12, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(side_dim, side_dim * 2),
            nn.GELU(),
            nn.Linear(side_dim * 2, side_dim)
        )
        self.norm1 = nn.LayerNorm(side_dim)
        self.norm2 = nn.LayerNorm(side_dim)

    def forward(self, side_tokens, clip_tokens):
        x = side_tokens + self.down_proj(clip_tokens)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class CLIPWithSideNetworkCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        print("Loading Pre-trained CLIP ViT-B/16 Vision Transformer...", flush=True)
        self.clip = timm.create_model('vit_base_patch16_clip_224.openai', pretrained=True)
        
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.side_blocks = nn.ModuleList([
            SideBlock(clip_dim=768, side_dim=768) for _ in range(4)
        ])
        
        self.classifier = nn.Linear(768 + 768, num_classes)

    def forward(self, x):
        if x.shape[-1] != 224:
            x = TF.resize(x, [224, 224], interpolation=transforms.InterpolationMode.BICUBIC)
            x = TF.normalize(x, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        B = x.shape[0]
        x_clip = self.clip.patch_embed(x)
        x_clip = self.clip._pos_embed(x_clip)
        x_clip = self.clip.patch_drop(x_clip)
        x_clip = self.clip.norm_pre(x_clip)
        
        side_tokens = torch.zeros(B, x_clip.shape[1], 768, device=x.device)
        tap_indices = [2, 5, 8, 11]
        
        side_idx = 0
        for i, block in enumerate(self.clip.blocks):
            x_clip = block(x_clip)
            if i in tap_indices:
                side_tokens = self.side_blocks[side_idx](side_tokens, x_clip)
                side_idx += 1
                
        x_clip = self.clip.norm(x_clip)
        clip_cls = self.clip.forward_head(x_clip, pre_logits=True)
        side_cls = side_tokens[:, 0]
        
        feat = torch.cat([clip_cls, side_cls], dim=-1)
        logits = self.classifier(feat)
        return logits

def evaluate(model, test_loader, criterion, device, classes):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                lbl = labels[i].item()
                class_correct[lbl] += c[i].item()
                class_total[lbl] += 1
                
    acc = 100.0 * correct / total
    avg_loss = test_loss / total
    return avg_loss, acc, class_correct, class_total

def main():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    except Exception:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=================================================================", flush=True)
    print(" DFD Side-Network on CIFAR-10 (side_dim = 768)", flush=True)
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print("=================================================================", flush=True)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    print("Fast-loading CIFAR-10 dataset via HuggingFace CDN...", flush=True)
    raw_dataset = load_dataset("uoft-cs/cifar10")
    
    train_set = HF_CIFAR10_Dataset(raw_dataset['train'], transform=transform_train)
    test_set = HF_CIFAR10_Dataset(raw_dataset['test'], transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Loaded {len(train_set):,} training samples and {len(test_set):,} test samples.", flush=True)

    model = CLIPWithSideNetworkCIFAR(num_classes=10).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("\nModel Architecture Parameter Summary (768 Dimensions):", flush=True)
    print(f" -> Trainable Parameters (Side-Network + Head): {trainable_params:,}", flush=True)
    print(f" -> Frozen Parameters (CLIP ViT-B/16 Backbone):  {frozen_params:,}", flush=True)
    print(f" -> Parameter Efficiency: {trainable_params / (trainable_params + frozen_params) * 100:.2f}% of parameters are trainable!\n", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    epochs = 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    print(f"Starting Training for {epochs} Epoch on {device}...\n", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = 100.0 * correct / total
                batch_loss = running_loss / total
                print(f" Epoch [{epoch}/{epochs}] Batch [{batch_idx+1:3d}/{len(train_loader)}] | Loss: {batch_loss:.4f} | Train Acc: {batch_acc:.2f}%", flush=True)

        scheduler.step()
        epoch_time = time.time() - epoch_start
        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        
        print("\nEvaluating on Test Set (10,000 images)...", flush=True)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device, classes)
        print(f"==> Epoch {epoch} Complete in {epoch_time:.1f}s | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n", flush=True)

    total_time = time.time() - start_time
    print("=================================================================", flush=True)
    print(f" TRAINING COMPLETE in {total_time/60:.2f} minutes!", flush=True)
    print(" Final Evaluation on CIFAR-10 Test Set (10,000 images):", flush=True)
    final_loss, final_acc, class_correct, class_total = evaluate(model, test_loader, criterion, device, classes)
    print(f" Final Test Accuracy: {final_acc:.2f}%", flush=True)
    print(f" Final Test Loss:     {final_loss:.4f}", flush=True)
    print("-----------------------------------------------------------------", flush=True)
    for i in range(10):
        cls_acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  - {classes[i]:<12}: {cls_acc:6.2f}% ({class_correct[i]}/{class_total[i]})", flush=True)
    print("=================================================================", flush=True)

    torch.save(model.state_dict(), "dfd_side_network_cifar10_768.pth")
    print("Model weights successfully saved to 'dfd_side_network_cifar10_768.pth'", flush=True)

    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    except Exception:
        pass

if __name__ == '__main__':
    main()
