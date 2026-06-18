import os
import argparse
import torch
import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate_metrics_from_cm(cm, num_classes):
    """Calcula UAR e UF1-score a partir de uma matriz de confusão agregada."""
    if np.sum(cm) == 0:
        return 0.0, 0.0

    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        
    uar = np.mean(recalls)
    uf1 = np.mean(f1_scores)
    
    return uar, uf1

def plot_accuracy_evolution(subjects, accuracies, save_path):
    try:
        sorted_data = sorted(zip(subjects, accuracies))
        subjects_sorted = [s for s, a in sorted_data]
        accuracies_sorted = [a * 100 for s, a in sorted_data] 
        
        subject_labels = [f"sub{int(s):02d}" for s in subjects_sorted]
        x_ticks = np.arange(len(subject_labels))
        
        plt.figure(figsize=(15, 7))
        plt.plot(x_ticks, accuracies_sorted, marker='o', linestyle='-', color='b', zorder=5)
        
        plt.title('Acurácia de Validação por Fold (LOSO)')
        plt.xlabel('Sujeito (Deixado de Fora)')
        plt.ylabel('Acurácia (%)')
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 101, 5)) 
        
        plt.xticks(x_ticks, subject_labels, rotation=90, fontsize=8) 
        
        mean_acc = np.mean(accuracies_sorted)
        plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Média: {mean_acc:.2f}%')
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6, axis='y')
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()

    except Exception as e:
        print(f"\nAVISO: Não foi possível gerar o gráfico de acurácia. Erro: {e}")

def get_all_subjects_3_class(annotations_file_path):
    try:
        df = pd.read_excel(annotations_file_path)
        
        def map_emotion(emotion):
            if emotion in ['sadness', 'disgust', 'contempt', 'fear', 'anger']: return 'Negative'
            elif emotion == 'happiness': return 'Positive'
            elif emotion == 'surprise': return 'Surprise'
            else: return None
        
        df['Generic Emotion'] = df['Estimated Emotion'].apply(map_emotion)
        df.dropna(subset=['Generic Emotion'], inplace=True)
        
        return sorted(df['Subject'].unique())
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo de anotações não encontrado em '{annotations_file_path}'")
        return None
    except Exception as e:
        print(f"ERRO ao ler o arquivo de anotações: {e}")
        return None

def monitor_training(checkpoints_dir, all_subjects, num_classes):
    print(f"Verificando o diretório: '{checkpoints_dir}'...")
    
    total_cm = np.zeros((num_classes, num_classes), dtype=int)
    completed_subjects = []
    
    plot_subjects = []
    plot_accuracies = []

    for filename in os.listdir(checkpoints_dir):
        if filename.startswith("best_model_loso_sub") and filename.endswith(".pth"):
            try:
                subject_num_str = filename.replace("best_model_loso_sub", "").replace(".pth", "")
                subject_num = int(subject_num_str)
                
                checkpoint_path = os.path.join(checkpoints_dir, filename)
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=torch.device('cpu'))
                
                best_cm = checkpoint.get('best_cm')
                best_acc = checkpoint.get('best_accuracy')

                if best_cm is not None and best_cm.shape == (num_classes, num_classes):
                    if subject_num not in completed_subjects:
                        total_cm += best_cm
                        completed_subjects.append(subject_num)
                        
                        if best_acc is not None:
                            plot_subjects.append(subject_num)
                            plot_accuracies.append(best_acc)
                else:
                    if subject_num not in completed_subjects:
                        print(f"AVISO: Checkpoint do sujeito {subject_num} não contém 'best_cm' válida.")

            except (ValueError, IndexError):
                continue
            except Exception as e:
                print(f"ERRO ao carregar o checkpoint '{filename}': {e}")
    
    completed_subjects.sort()
    
    if not completed_subjects:
        print("\nNenhum checkpoint com resultado válido ('best_cm') encontrado ainda. Aguardando...")
        return
        
    pending_subjects = [s for s in all_subjects if s not in completed_subjects]

    print("\n--- RELATÓRIO DE STATUS DO TREINAMENTO LOSO ---")
    print(f"Folds com resultado: {len(completed_subjects)} de {len(all_subjects)}")
    print(f"Sujeitos Pendentes ({len(pending_subjects)}): {pending_subjects}")
    
    # --- CÁLCULO DAS MÉTRICAS ---
    
    # 1. Média das Acurácias (Igual ao train.py)
    mean_acc = np.mean(plot_accuracies) if plot_accuracies else 0.0
    std_acc = np.std(plot_accuracies) if plot_accuracies else 0.0
    
    # 2. Acurácia Global / Micro (Cálculo em cima da Matriz Total)
    micro_acc = np.trace(total_cm) / np.sum(total_cm) if np.sum(total_cm) > 0 else 0.0
    
    # 3. UAR e UF1 (Cálculo em cima da Matriz Total)
    uar, uf1 = calculate_metrics_from_cm(total_cm, num_classes)
    
    print("\n--- MATRIZ DE CONFUSÃO AGREGADA ---")
    print(total_cm)
    
    print("\n--- MÉTRICAS DE ACURÁCIA ---")
    print(f"Acurácia Média Final (Igual ao train.py): {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Acurácia Global (Micro via Matriz Total): {micro_acc:.4f}")
    
    print("\n--- MÉTRICAS AVANÇADAS (Baseado na CM agregada) ---")
    print(f"Unweighted Average Recall (UAR): {uar:.4f}")
    print(f"Unweighted F1-Score (UF1):       {uf1:.4f}")
    
    if plot_subjects:
        plot_save_path = os.path.join(checkpoints_dir, 'loso_accuracy_evolution.png')
        plot_accuracy_evolution(plot_subjects, plot_accuracies, plot_save_path)

def main():
    parser = argparse.ArgumentParser(description="Monitora um treinamento LOSO calculando UAR e UF1.")
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--annotations_file', type=str, default='./CASME2-coding-20140508.xlsx')
    parser.add_argument('--interval', type=int, default=60)

    args = parser.parse_args()

    all_subjects = get_all_subjects_3_class(args.annotations_file)
    if all_subjects is None:
        return
    
    num_classes = 3

    if args.interval > 0:
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("==========================================================")
                print("   MONITOR DE TREINAMENTO LOSO (MÉTRICAS CORRIGIDAS)      ")
                print(f"   Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   ")
                print("==========================================================")
                monitor_training(args.checkpoints_dir, all_subjects, num_classes)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoramento interrompido.")
    else:
        monitor_training(args.checkpoints_dir, all_subjects, num_classes)

if __name__ == "__main__":
    main()