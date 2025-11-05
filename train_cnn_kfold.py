import os, cv2
import numpy as np
import matplotlib.pyplot as plt

# Para garantir que o K-Fold seja reproduzível
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

# =================================================================
# 0. CONFIGURAÇÃO DE AMBIENTE E PARÂMETROS
# =================================================================

# Ajuste este caminho para o local da sua pasta 'dataset'
dataset_path = "dataset" 
NUM_EPOCHS = 10     # Número de épocas por fold
BATCH_SIZE = 4      # Tamanho do lote (ajuste conforme a sua GPU/memória)
K_FOLDS = 5         # Número de folds para validação cruzada
LEARNING_RATE = 0.001
# Se preferir usar hold-out (treino/val) em vez de K-Fold, ajuste estas opções:
USE_KFOLD = True    # True para K-Fold, False para hold-out
VALIDATION_SPLIT = 0.2  # 0.2 -> 20/80, 0.3 -> 30/70 (usado quando USE_KFOLD=False)

# Tenta usar a GPU (CUDA) se estiver disponível, senão usa a CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# =================================================================
# 1. PRÉ-PROCESSAMENTO: Definição das Transformações
# =================================================================

# Define as transformações a serem aplicadas em cada imagem
transform = transforms.Compose([
    transforms.ToTensor(), # Converte imagem para tensor (H, W, C -> C, H, W) e normaliza (0-255 -> 0.0-1.0)
    transforms.Resize((64, 64)), # Redimensiona todas as imagens para 64x64
    # Normalização com base em estatísticas médias (melhora a convergência)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# =================================================================
# 2. DATASET: Classe para Carregar Imagens Personalizadas
# =================================================================

class CustomImageDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform
        
        # Gera a lista de classes (subpastas) e atribui labels
        self.classes = sorted(os.listdir(base_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images, self.labels = [], [] 
        
        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            folder = os.path.join(base_dir, cls_name)
            for img_name in os.listdir(folder):
                self.images.append(os.path.join(folder, img_name))
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        # Converte de BGR (OpenCV) para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
            
        label = self.labels[idx]
        
        # Retorna o tensor da imagem e o label (índice da classe)
        return img, label

# =================================================================
# 3. MODELO: Arquitetura da SimpleCNN
# =================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Bloco 1: CONV -> ReLU -> POOL
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        )
        
        # Bloco 2: CONV -> ReLU -> POOL
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        )
        
        # Bloco 3: CONV -> ReLU -> POOL
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        )
        
        # Camada Totalmente Conectada (FC)
        # O tamanho de entrada é (canais * altura * largura) -> 64 * 8 * 8
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes) # Saída final com o número de classes
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Achatamento (Flatten) para a camada densa
        out = out.reshape(out.size(0), -1) 
        out = self.fc(out)
        return out

# =================================================================
# 4. K-FOLD CROSS-VALIDATION
# =================================================================

def train_and_evaluate():
    # Inicializa o Dataset completo
    full_dataset = CustomImageDataset(dataset_path, transform)
    
    if not full_dataset.images:
        print(f"ERRO: Nenhuma imagem encontrada no caminho: {dataset_path}")
        print("Verifique se a pasta 'dataset' existe e contém subpastas de classes.")
        return

    NUM_CLASSES = len(full_dataset.classes)
    print(f"Classes detectadas: {full_dataset.classes} ({NUM_CLASSES} classes)")
    
    # Obter os índices de todas as imagens
    all_indices = np.arange(len(full_dataset))

    # Preparar splits: K-Fold ou Hold-out
    fold_splits = []
    if USE_KFOLD:
        # Configuração do K-Fold
        kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
        for train_idx, val_idx in kfold.split(all_indices):
            fold_splits.append((train_idx, val_idx))
    else:
        # Hold-out: stratificado para preservar proporção de classes
        train_idx, val_idx = train_test_split(all_indices, test_size=VALIDATION_SPLIT,
                                              stratify=full_dataset.labels, random_state=42)
        fold_splits.append((train_idx, val_idx))

    fold_results = []
    all_true_labels = []
    all_predictions = []
    # Guarda as perdas (loss) por época para cada fold (para plotagem)
    losses_by_fold = []

    # Loop principal sobre os splits (K folds ou único hold-out)
    for fold, (train_indices, val_indices) in enumerate(fold_splits):
        print(f"\n--- FOLD {fold+1}/{K_FOLDS} ---")
        
        # 4.1. Criar Subsets e DataLoaders
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 4.2. Inicializar Modelo, Loss e Otimizador
        model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 4.3. Loop de Treinamento (agora registrando perda por época para plotagem)
        epoch_losses = []
        for epoch in range(NUM_EPOCHS):
            model.train() # Coloca o modelo em modo de treinamento
            running_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}')

        # Armazena as perdas deste fold
        losses_by_fold.append(epoch_losses)
            
        # 4.4. Avaliação do Fold
        model.eval() # Coloca o modelo em modo de avaliação
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Obtém a classe com maior probabilidade
                _, predicted = torch.max(outputs.data, 1) 
                
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(predicted.cpu().tolist())
        
        # Calcular e armazenar resultados do fold
        fold_accuracy = accuracy_score(y_true, y_pred)
        fold_results.append(fold_accuracy)
        print(f"Acurácia do Fold {fold+1}: {fold_accuracy:.4f}")
        
        all_true_labels.extend(y_true)
        all_predictions.extend(y_pred)

    # 4.5. Resultados Finais (Após todos os Folds)
    final_accuracy = np.mean(fold_results)
    final_cm = confusion_matrix(all_true_labels, all_predictions)

    print("\n--- RESULTADOS FINAIS ---")
    print(f"Acurácia Média K-Fold ({K_FOLDS} folds): {final_accuracy:.4f}")
    print("Matriz de Confusão Final (Combinada):")
    print(final_cm)
    
    # Imprimir a matriz de confusão com nomes de classes
    print("\nMatriz de Confusão Mapeada:")
    print(f"Classes: {full_dataset.classes}")
    
    # Você pode usar a matriz de confusão para analisar onde o modelo erra.
    # Exemplo: Se o índice [0, 1] for alto, o modelo confundiu "apartamento" (0) com "sobrado" (1).

    # =================================================================
    # 5. PLOTAGEM: Gera e salva gráficos de perda por fold, acurácia por fold e matriz de confusão
    # =================================================================
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)

    # 5.1. Curvas de perda por fold
    if losses_by_fold:
        plt.figure(figsize=(8, 6))
        for i, losses in enumerate(losses_by_fold):
            plt.plot(range(1, len(losses) + 1), losses, marker='o', label=f'Fold {i+1}')
        plt.title('Training Loss por Época (por Fold)')
        plt.xlabel('Época')
        plt.ylabel('Loss (média do batch)')
        plt.legend()
        plt.grid(True)
        loss_plot_path = os.path.join(results_dir, 'loss_curves.png')
        plt.savefig(loss_plot_path, bbox_inches='tight')
        print(f"Curvas de perda salvas em: {loss_plot_path}")
        plt.close()

    # 5.2. Acurácia por fold (barras)
    if fold_results:
        plt.figure(figsize=(6, 4))
        plt.bar([f'Fold {i+1}' for i in range(len(fold_results))], fold_results, color='skyblue')
        plt.ylim(0, 1)
        plt.ylabel('Acurácia')
        plt.title('Acurácia por Fold')
        for i, v in enumerate(fold_results):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        acc_plot_path = os.path.join(results_dir, 'fold_accuracies.png')
        plt.savefig(acc_plot_path, bbox_inches='tight')
        print(f"Acurácias por fold salvas em: {acc_plot_path}")
        plt.close()

    # 5.3. Matriz de confusão (heatmap)
    try:
        classes = full_dataset.classes
        cm = np.array(final_cm)
        plt.figure(figsize=(6, 6))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusão (Combinada)')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)

        # Anotações nas células
        thresh = cm.max() / 2. if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('Classe Verdadeira')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        cm_plot_path = os.path.join(results_dir, 'confusion_matrix.png')
        plt.savefig(cm_plot_path, bbox_inches='tight')
        print(f"Matriz de confusão salva em: {cm_plot_path}")
        plt.close()
    except Exception as e:
        print(f"Não foi possível plotar a matriz de confusão: {e}")

if __name__ == "__main__":
    train_and_evaluate()