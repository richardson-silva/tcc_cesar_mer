import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import time
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# Importa o modelo
from cesar_net import CESAR_Net

# Dataset Customizado para o CASME2
class CASME2Dataset(Dataset):
    def __init__(self, annotations_df, data_dir, subject_list, emotion_map, emotion_col, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.emotion_map = emotion_map
        self.emotion_col = emotion_col
        self.samples_df = annotations_df[annotations_df['Subject'].isin(subject_list)].reset_index(drop=True)
        self.num_rois = 7
    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        row = self.samples_df.iloc[idx]
        subject_num = int(row['Subject'])
        subject_str = f"sub{subject_num:02d}"
        filename = row['Filename']
        emotion_label = row[self.emotion_col]
        emotion_idx = self.emotion_map[emotion_label]

        sample_path = os.path.join(self.data_dir, subject_str, filename)
        path_u = os.path.join(sample_path, 'flow_u.png')
        path_v = os.path.join(sample_path, 'flow_v.png')
        path_strain = os.path.join(sample_path, 'strain.png')
        path_rois = os.path.join(sample_path, 'rois.npy')       
  
        img_u = Image.open(path_u).convert("L")
        img_v = Image.open(path_v).convert("L")
        img_strain = Image.open(path_strain).convert("L")
        
        try:
            rois = np.load(path_rois)
            if rois.shape[0] != self.num_rois:
                raise FileNotFoundError 
        except (FileNotFoundError, IOError):
            print(f"Aviso: rois.npy não encontrado ou inválido em {sample_path}. Usando ROIs de fallback (imagem inteira).")
            rois = np.array([[0.0, 0.0, 1.0, 1.0]] * self.num_rois, dtype=np.float32)
  
        if self.transform:
            img_u = self.transform(img_u)
            img_v = self.transform(img_v)
            img_strain = self.transform(img_strain)

        return img_u, img_v, img_strain, torch.from_numpy(rois).float(), emotion_idx

def save_checkpoint(state, filename="checkpoint.pth"):
    print(f"=> Salvando checkpoint em {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, optimizer):
    # <<< MODIFICADO (lógica do 'grape') >>>
    if os.path.exists(checkpoint_path):
        print(f"=> Carregando checkpoint de {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Carrega os valores do checkpoint
        start_epoch = checkpoint['epoch']
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        best_gap = checkpoint.get('best_gap', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        
        return start_epoch, best_accuracy, best_gap, best_epoch
    else:
        print(f"=> Nenhum checkpoint encontrado em {checkpoint_path}")
        return 0, 0.0, float('inf'), 0

def get_emotion_map_3_classes():
    emotion_list = ['Negative', 'Positive', 'Surprise']
    return {emotion: i for i, emotion in enumerate(emotion_list)}

def collate_rois(batch):
    imgs_u, imgs_v, imgs_strain, rois_list, targets = zip(*batch)
    imgs_u_batch = torch.stack(imgs_u, 0)
    imgs_v_batch = torch.stack(imgs_v, 0)
    imgs_strain_batch = torch.stack(imgs_strain, 0)
    targets_batch = torch.tensor(targets, dtype=torch.long)
    
    batched_rois = []
    for i, rois in enumerate(rois_list):
        batch_idx_col = torch.full((rois.shape[0], 1), i, dtype=rois.dtype)
        batched_rois.append(torch.cat([batch_idx_col, rois], dim=1))
    
    rois_batch = torch.cat(batched_rois, dim=0)
    return imgs_u_batch, imgs_v_batch, imgs_strain_batch, rois_batch, targets_batch

def generate_diagnostic_image(annotations_df, processed_data_dir, output_dir):
    print("Gerando imagem de diagnóstico do pipeline de dados...")
    try:
        first_sample = annotations_df.iloc[0]
        subject_num = int(first_sample['Subject'])
        subject_str = f"sub{subject_num:02d}"
        filename = first_sample['Filename']
        onset_frame = int(first_sample['OnsetFrame'])
        apex_frame = int(first_sample['ApexFrame'])

        RAW_DATA_DIR = '../CASME2_RAW_selected' 
        raw_sample_path = os.path.join(RAW_DATA_DIR, subject_str, filename)
        processed_sample_path = os.path.join(processed_data_dir, subject_str, filename)

        apex_frame_files = glob(os.path.join(raw_sample_path, f"img{apex_frame}.*"))
        if not apex_frame_files:
            print(f"AVISO: Não foi possível encontrar o frame do ápice original em '{raw_sample_path}'. Pulando diagnóstico.")
            return

        flow_u_path = os.path.join(processed_sample_path, 'flow_u.png')
        rois_path = os.path.join(processed_sample_path, 'rois.npy')

        target_size = (224, 224) 
        img_apex_raw = cv2.imread(apex_frame_files[0])
        img_apex_raw = cv2.cvtColor(img_apex_raw, cv2.COLOR_BGR2RGB)
        img_apex_resized = cv2.resize(img_apex_raw, target_size)

        img_flow = cv2.imread(flow_u_path, cv2.IMREAD_GRAYSCALE)
        rois_norm = np.load(rois_path)

        def draw_rois(image, rois_normalized):
            img_with_rois = image.copy()
            if len(img_with_rois.shape) == 2:
                img_with_rois = cv2.cvtColor(img_with_rois, cv2.COLOR_GRAY2RGB)
            H, W = img_with_rois.shape[:2]
            for roi in rois_normalized:
                x1 = int(roi[0] * W)
                y1 = int(roi[1] * H)
                x2 = int(roi[2] * W)
                y2 = int(roi[3] * H)
                if img_with_rois.shape[2] == 3:
                     cv2.rectangle(img_with_rois, (x1, y1), (x2, y2), (255, 0, 0), 1) 
            return img_with_rois

        img_apex_with_landmarks = draw_rois(img_apex_resized, rois_norm)
        img_flow_with_landmarks = draw_rois(img_flow, rois_norm)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_apex_resized)
        axes[0].set_title('1. Frame Ápice (Original)')
        axes[0].axis('off')
        
        axes[1].imshow(img_flow, cmap='gray')
        axes[1].set_title('2. Fluxo Óptico (Farneback - U)')
        axes[1].axis('off')
        
        axes[2].imshow(img_apex_with_landmarks)
        axes[2].set_title('3. Ápice c/ Landmarks (ROIs)')
        axes[2].axis('off')
        
        axes[3].imshow(img_flow_with_landmarks)
        axes[3].set_title('4. Fluxo c/ Landmarks (ROIs)')
        axes[3].axis('off')
        
        fig.suptitle(f"Diagnóstico da Amostra: {subject_str}/{filename}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(output_dir, 'diagnostic_pipeline_image.png')
        plt.savefig(save_path)
        plt.close()
        
        print(f"Imagem de diagnóstico salva com sucesso em: {save_path}")

    except Exception as e:
        print(f"ERRO ao gerar imagem de diagnóstico: {e}")
        print("Certifique-se que o diretório '../CASME2_RAW_selected' está acessível")
        print("e que os dados da primeira amostra foram pré-processados (flow_u.png e rois.npy).")


def save_epoch_history_plot(history, file_path, subject_id, total_epochs, best_epoch, best_acc):
    """
    Salva um gráfico da evolução da acurácia de treino e validação
    ao longo das épocas para um único fold (sujeito).
    
    <<< MODIFICADO (lógica do 'grape') >>>
    history: dict com 'epochs', 'train_acc', 'val_acc'
    best_epoch: O número da época (ex: 102) que teve o melhor modelo
    best_acc: A acurácia de validação (ex: 0.6667) daquela época
    """
    try:
        # Converte acurácias de [0, 1] para [0, 100]
        train_accs_percent = [acc * 100 for acc in history['train_acc']]
        val_accs_percent = [acc * 100 for acc in history['val_acc']]
        epochs = history['epochs']
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(epochs, train_accs_percent, 'bo-', label='Acurácia de Treino', markersize=4)
        plt.plot(epochs, val_accs_percent, 'ro-', label='Acurácia de Validação', markersize=4)
        
        if best_epoch > 0:
            best_val_acc_percent = best_acc * 100
            plt.axvline(x=best_epoch, color='grey', linestyle='--', linewidth=1, 
                        label=f'Melhor Val: {best_val_acc_percent:.2f}% na Época {best_epoch}')
        
        plt.title(f'Evolução do Treinamento - Sujeito {subject_id}')
        plt.xlabel('Época')
        plt.ylabel('Acurácia (%)')
        
        # Configura os eixos
        plt.xlim(0, total_epochs + 1)
        plt.ylim(0, 105) 
        plt.xticks(np.arange(0, total_epochs + 1, max(1, total_epochs // 10))) 
        plt.yticks(np.arange(0, 101, 10))
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        plt.savefig(file_path)
        plt.close() 
        
    except Exception as e:
        print(f"AVISO: Não foi possível gerar o gráfico de épocas para o sujeito {subject_id}. Erro: {e}")

def check_for_best_model(current_train_acc, current_val_acc, best_val_acc, best_gap):
    """
    Verifica se o modelo atual é melhor que o anterior.
    
    Critérios:
    1. Se a acurácia de validação (current_val_acc) for maior, é o melhor.
    2. Se a acurácia de validação for igual, é o melhor se o "gap" (diferença
       absoluta entre treino e validação) for menor.
       
    Retorna: (is_best, new_best_val_acc, new_best_gap)
    """
    # Calcula a diferença absoluta (o "gap")
    current_gap = abs(current_train_acc - current_val_acc)
    
    if current_val_acc > best_val_acc:
        # Caso 1: Acurácia de validação melhorou
        print(f"  > Nova melhor acurácia de validação: {current_val_acc:.4f} (superou {best_val_acc:.4f})")
        return True, current_val_acc, current_gap
        
    if current_val_acc == best_val_acc:
        # Caso 2: Acurácia de validação é a mesma, verificar o 'gap'
        if current_gap < best_gap:
            print(f"  > Acurácia de validação igual ({current_val_acc:.4f}), mas 'gap' melhorou: {current_gap:.4f} (era {best_gap:.4f})")
            return True, current_val_acc, current_gap
    
    # Caso 3: Sem melhoria
    return False, best_val_acc, best_gap

# --- Funções de Treino e Validação ---
# focado em 'bestmodel save e plot'.
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0
    all_preds, all_labels = [], []
    for batch_idx, (data_u, data_v, data_strain, rois, targets) in enumerate(loop):
        data_u, data_v, data_strain = data_u.to(device), data_v.to(device), data_strain.to(device)
        rois = rois.to(device)
        targets = targets.to(device)
        scores = model(data_u, data_v, data_strain, rois)
        loss = loss_fn(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        # Sem clipping de gradiente (mantido do 'apple' original)
        optimizer.step()
        running_loss += loss.item()
        _, predictions = torch.max(scores, 1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())
        loop.set_postfix(loss=loss.item())
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate_model(loader, model, loss_fn, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data_u, data_v, data_strain, rois, targets in loader:
            data_u, data_v, data_strain = data_u.to(device), data_v.to(device), data_strain.to(device)
            rois = rois.to(device)
            targets = targets.to(device)
            scores = model(data_u, data_v, data_strain, rois)
            loss = loss_fn(scores, targets)
            running_loss += loss.item()
            _, predictions = torch.max(scores, 1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(get_emotion_map_3_classes()))))
    return val_loss, val_acc, cm

# --- Função Principal ---
def main(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        annotations_df = pd.read_excel(args.annotations_file)
    except FileNotFoundError:
        print(f"Erro: Arquivo de anotações não encontrado em {args.annotations_file}")
        return

    generate_diagnostic_image(annotations_df, args.data_dir, args.output_dir)

    emotion_map = get_emotion_map_3_classes()
    
    # Mapeamento de emoções sem alterações
    def map_emotion(emotion):
        if emotion in ['sadness', 'disgust', 'contempt', 'fear', 'anger']:
            return 'Negative'
        elif emotion == 'happiness':
            return 'Positive'
        elif emotion == 'surprise':
            return 'Surprise'
        else:
            return None 

    new_emotion_col = 'Generic Emotion'
    annotations_df[new_emotion_col] = annotations_df['Estimated Emotion'].apply(map_emotion)
    
    print("DataFrame original de anotações:", len(annotations_df))
    annotations_df.dropna(subset=[new_emotion_col], inplace=True)
    print(f"Total de amostras após mapeamento RÍGIDO para 3 classes: {len(annotations_df)}")
    print("Distribuição das classes a serem utilizadas:")
    print(annotations_df[new_emotion_col].value_counts())

    # Transforms sem alterações
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    all_subjects = sorted(annotations_df['Subject'].unique())
    fold_accuracies = []

    start_fold_idx = 0
    # Lógica de 'resume' sem alterações
    if args.resume:
        print("Tentando continuar o treinamento...")
        completed_subjects = set()
        for f in os.listdir(args.output_dir):
            if f.startswith("best_model_loso_sub") and f.endswith(".pth"):
                try:
                    sub_num = int(f.replace("best_model_loso_sub", "").replace(".pth", ""))
                    completed_subjects.add(sub_num)
                except ValueError:
                    continue
        
        for i, subject in enumerate(all_subjects):
            if subject not in completed_subjects:
                start_fold_idx = i
                print(f"Treinamento continuará a partir do Fold {start_fold_idx + 1} (Sujeito {subject}).")
                break
        else: 
            print("Todos os folds já foram completados. Nenhum treinamento a ser continuado.")
            return

    for fold_idx in range(start_fold_idx, len(all_subjects)):
        test_subject = all_subjects[fold_idx]
        start_time_fold = time.time()
        print("\n" + "="*50)
        print(f"Iniciando Fold {fold_idx + 1}/{len(all_subjects)}: Deixando o Sujeito {test_subject} de fora.")
        print("="*50)

        train_subjects = [s for s in all_subjects if s != test_subject]
        
        train_dataset = CASME2Dataset(annotations_df, args.data_dir, train_subjects, emotion_map, new_emotion_col, train_transform)
        val_dataset = CASME2Dataset(annotations_df, args.data_dir, [test_subject], emotion_map, new_emotion_col, val_transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_rois)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_rois)

        model = CESAR_Net(
            num_classes=len(emotion_map),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            dropout_rate=args.dropout_cnn,
            dropout_classifier=args.dropout_fc
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

        train_labels = [emotion_map[e] for e in train_dataset.samples_df[new_emotion_col]]
        class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
        
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

        start_epoch = 0
        
        best_val_acc = 0.0
        best_gap = float('inf')
        best_epoch = 0
        
        resume_path = os.path.join(args.output_dir, f"resume_checkpoint_sub{test_subject}.pth")
        plot_save_path = resume_path.replace('.pth', '.png')
  
        if args.resume and os.path.exists(resume_path):
            start_epoch, best_val_acc, best_gap, best_epoch = load_checkpoint(resume_path, model, optimizer)
            start_epoch += 1
            print(f"  > Continuando com best_val_acc={best_val_acc:.4f} e best_gap={best_gap:.4f}")


        epoch_numbers = []
        epoch_train_accs = []
        epoch_val_accs = []
  
        for epoch in range(start_epoch, args.epochs):
            print(f"\n--- Fold {fold_idx + 1}, Época {epoch + 1}/{args.epochs} ---")
            train_loss, train_acc = train_one_epoch(train_loader, model, optimizer, loss_fn, DEVICE)
            scheduler.step()
   
            val_loss, val_acc, fold_cm_epoch = validate_model(val_loader, model, loss_fn, DEVICE)
            print(f"Treino Loss: {train_loss:.4f}, Treino Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            current_epoch_num = epoch + 1
            epoch_numbers.append(current_epoch_num)
            epoch_train_accs.append(train_acc)
            epoch_val_accs.append(val_acc)
            
            history = {
                'epochs': epoch_numbers,
                'train_acc': epoch_train_accs,
                'val_acc': epoch_val_accs
            }
            # Passa best_epoch e best_val_acc para a função de plotar
            save_epoch_history_plot(history, plot_save_path, int(test_subject), args.epochs, best_epoch, best_val_acc)
   
            is_best, new_best_val_acc, new_best_gap = check_for_best_model(
                train_acc, val_acc, best_val_acc, best_gap
            )
            
            # Atualiza os valores de rastreamento
            best_val_acc = new_best_val_acc
            best_gap = new_best_gap
            
            if is_best:
                best_epoch = current_epoch_num
                best_val_acc = val_acc # Garante que estamos salvando a acurácia correta
                best_filename = os.path.join(args.output_dir, f"best_model_loso_sub{int(test_subject)}.pth")
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_accuracy': best_val_acc,
                    'best_gap': best_gap,
                    'best_epoch': best_epoch,
                    'best_cm': fold_cm_epoch
                }, filename=best_filename)
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_accuracy': best_val_acc,
                'best_gap': best_gap,
                'best_epoch': best_epoch
            }, filename=resume_path)

        print(f"\n--- Avaliação Final do Fold {fold_idx + 1} no Sujeito {test_subject} ---")
        best_model_path = os.path.join(args.output_dir, f"best_model_loso_sub{int(test_subject)}.pth")
        
        final_model = CESAR_Net(
            num_classes=len(emotion_map),
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        ).to(DEVICE)
        final_optimizer = optim.AdamW(final_model.parameters(), lr=args.lr)
        
        load_checkpoint(best_model_path, final_model, final_optimizer)
        
        fold_loss, fold_acc, fold_cm = validate_model(val_loader, final_model, loss_fn, DEVICE)
        fold_accuracies.append(fold_acc)
        
        end_time_fold = time.time()
        print(f"Acurácia para o Sujeito {test_subject}: {fold_acc:.4f}")
        print("Matriz de Confusão:\n", fold_cm)
        print(f"Tempo do fold: {(end_time_fold - start_time_fold)/60:.2f} minutos")

    mean_accuracy = np.mean(fold_accuracies) if fold_accuracies else 0.0
    std_accuracy = np.std(fold_accuracies) if fold_accuracies else 0.0

    print("\n" + "="*50)
    print("Treinamento LOSO Concluído!")
    print(f"Acurácia Média Final: {mean_accuracy:.4f}")
    print(f"Desvio Padrão da Acurácia: {std_accuracy:.4f}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento LOSO da CESAR-Net no CASME2 para 3 classes (versão rigorosa)")
    parser.add_argument('--data_dir', type=str, default='./processed_data', help='Diretório com os dados pré-processados.')
    parser.add_argument('--annotations_file', type=str, default='../CASME2-coding-20140508.xlsx', help='Caminho para o arquivo Excel de anotações.')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Diretório para salvar checkpoints e resultados.')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas de treinamento por fold.')
    parser.add_argument('--batch_size', type=int, default=16, help='Tamanho do lote.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Taxa de aprendizado.')
    parser.add_argument('--d_model', type=int, default=96, help='Dimensão do modelo no Transformer.')
    parser.add_argument('--n_heads', type=int, default=2, help='Número de cabeças de atenção no Transformer.')
    parser.add_argument('--n_layers', type=int, default=1, help='Número de camadas no Transformer Encoder.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Força da regularização L2 (weight decay).')
    parser.add_argument('--dropout_cnn', type=float, default=0.2, help='Taxa de dropout para as camadas da CNN.')
    parser.add_argument('--dropout_fc', type=float, default=0.5, help='Taxa de dropout para o classificador final.')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Fator de suavização para a loss function.')
    parser.add_argument('--resume', action='store_true', help='Use esta flag para continuar o treinamento LOSO do último fold não finalizado.')
    
    args = parser.parse_args()
    main(args)