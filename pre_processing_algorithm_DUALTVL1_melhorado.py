import os
import cv2
import numpy as np
import pandas as pd
from glob import glob

def calculate_optical_strain(flow_u, flow_v):
    """
    Calcula a deformação óptica (optical strain) a partir dos componentes
    horizontal (u) e vertical (v) do fluxo óptico.
    A fórmula é ε = 0.5 * (∇u + (∇u)T)
    """
    grad_u_x = cv2.Sobel(flow_u, cv2.CV_64F, 1, 0, ksize=3)
    grad_u_y = cv2.Sobel(flow_u, cv2.CV_64F, 0, 1, ksize=3)
    grad_v_x = cv2.Sobel(flow_v, cv2.CV_64F, 1, 0, ksize=3)
    grad_v_y = cv2.Sobel(flow_v, cv2.CV_64F, 0, 1, ksize=3)

    strain_magnitude = np.sqrt(grad_u_x**2 + grad_v_y**2 + 0.5 * (grad_u_y + grad_v_x)**2)
    return strain_magnitude

def preprocess_casme2_tvl1(
    base_dir, 
    coding_sheet_path, 
    output_dir, 
    target_size=(224, 224)
):
    """
    Processa o dataset CASME2 para extrair fluxo óptico e deformação óptica
    usando TV-L1 com lógica robusta de seleção de frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_excel(coding_sheet_path)
    
    # Instancia o algoritmo de fluxo óptico TV-L1 (Requer opencv-contrib-python)
    tvl1 = cv2.optflow.createOptFlow_DualTVL1()

    print("Iniciando o pré-processamento com TV-L1...")
    
    for index, row in df.iterrows():
        subject = f"sub{int(row['Subject']):02d}"
        filename = row['Filename']
        onset_frame_num = int(row['OnsetFrame'])
        
        apex_frame_raw = row['ApexFrame']
        try:
            apex_frame_num = int(apex_frame_raw)
        except ValueError:
            print(f"AVISO: 'ApexFrame' inválido para {subject}/{filename}. Usando o frame do MEIO como Apex.")
            apex_frame_num = -1 
            
        emotion = row['Estimated Emotion']
        sample_path = os.path.join(base_dir, subject, filename)
        
        if not os.path.isdir(sample_path):
            print(f"Aviso: Diretório não encontrado, pulando: {sample_path}")
            continue

        # --- INÍCIO DA LÓGICA DE SELEÇÃO DE FRAMES ---
        all_frames = sorted(glob(os.path.join(sample_path, '*.jpg')))
        
        if not all_frames:
             print(f"Aviso: Nenhum frame encontrado em {sample_path}. Pulando.")
             continue
             
        onset_img_path = all_frames[0] 
        
        if apex_frame_num == -1: 
            apex_img_index = len(all_frames) // 2
        else:
            apex_img_index = apex_frame_num - onset_frame_num

        try:
            if apex_img_index >= len(all_frames) or apex_img_index < 0:
                apex_img_index = len(all_frames) // 2
                print(f"Aviso: Índice fora do alcance em {sample_path}. Interpolando para o frame do MEIO.")
                
            apex_img_path = all_frames[apex_img_index]
        except Exception:
            apex_img_path = all_frames[len(all_frames) // 2]
        # --- FIM DA LÓGICA ---

        frame_onset = cv2.imread(onset_img_path, cv2.IMREAD_GRAYSCALE)
        frame_apex = cv2.imread(apex_img_path, cv2.IMREAD_GRAYSCALE)

        if frame_onset is None or frame_apex is None:
            print(f"Erro ao ler imagens para {sample_path}. Pulando.")
            continue
            
        frame_onset = cv2.resize(frame_onset, target_size)
        frame_apex = cv2.resize(frame_apex, target_size)

        # Calcula o fluxo óptico com TV-L1
        flow = tvl1.calc(frame_onset, frame_apex, None)
        flow_u, flow_v = flow[..., 0], flow[..., 1]

        strain = calculate_optical_strain(flow_u, flow_v)

        flow_u_norm = cv2.normalize(flow_u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_v_norm = cv2.normalize(flow_v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        strain_norm = cv2.normalize(strain, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        output_sample_path = os.path.join(output_dir, subject, filename)
        if not os.path.exists(output_sample_path):
            os.makedirs(output_sample_path)

        cv2.imwrite(os.path.join(output_sample_path, 'flow_u.png'), flow_u_norm)
        cv2.imwrite(os.path.join(output_sample_path, 'flow_v.png'), flow_v_norm)
        cv2.imwrite(os.path.join(output_sample_path, 'strain.png'), strain_norm)

        print(f"Processado (TV-L1): {subject}/{filename}")

    print("Pré-processamento concluído.")

# --- Execução ---
CASME2_SELECTED_RAW_PATH = './CASME2_RAW_selected'
CODING_SHEET_PATH = './CASME2-coding-20140508.xlsx'
OUTPUT_DATA_PATH = './processed_data_DUALTVL1_melhorado'

preprocess_casme2_tvl1(CASME2_SELECTED_RAW_PATH, CODING_SHEET_PATH, OUTPUT_DATA_PATH)