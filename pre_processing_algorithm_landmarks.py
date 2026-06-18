import os
import cv2
import numpy as np
import pandas as pd
from glob import glob
import mediapipe as mp
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

# Índices dos marcos faciais do MediaPipe para as ROIs
ROI_INDICES = {
    # 1. Olho Esquerdo (Cobre AU5, AU6, AU7)
    'left_eye': [33, 160, 158, 133, 153, 144],
    # 2. Olho Direito (Cobre AU5, AU6, AU7)
    'right_eye': [362, 385, 387, 263, 373, 380],
    # 3. Sobrancelha Esquerda (Cobre AU1, AU2, AU4)
    'left_eyebrow': [70, 63, 105, 66, 107],
    # 4. Sobrancelha Direita (Cobre AU1, AU2, AU4)
    'right_eyebrow': [336, 296, 334, 293, 300],
    # 5. Boca (Generalista - Cobre AU12, AU15, AU26, etc.)
    'mouth': [61, 291, 0, 17, 405, 191], 
    
    # 6.DESATIVADO
    # Ponte do Nariz (Especialista - Cobre AU9 - 'Nose Wrinkler')
    # 'nose_wrinkle_bridge': [6, 168, 197, 195, 5, 4], 
    
    # 7. Nasolabial Esquerdo (Cobre AU10)
    # Foca na 'asa' do nariz e bochecha adjacente
    'left_nasolabial': [111, 134, 241, 131, 165], 
    # 8. Nasolabial Direito (Cobre AU10)
    'right_nasolabial': [340, 363, 461, 360, 391],

    # 8. Testa/Glabela (Cobre AU1, AU4)
    # 'forehead_glabella': [10, 151, 107, 336, 9, 8]
}

def get_rois_from_landmarks(landmarks, padding_factor=0.2):
    """
    Converte landmarks em bboxes normalizadas [x_min, y_min, x_max, y_max],
    adicionando um padding percentual e garantindo que fique entre [0, 1].
    """
    rois = []
    
    if not landmarks:
        return np.array([], dtype=np.float32)

    for roi_name, indices in ROI_INDICES.items():
        points_x = [landmarks[i].x for i in indices]
        points_y = [landmarks[i].y for i in indices]
        
        x_min = np.min(points_x)
        y_min = np.min(points_y)
        x_max = np.max(points_x)
        y_max = np.max(points_y)
        
        # --- Lógica de Padding ---
        width = x_max - x_min
        height = y_max - y_min
        
        # Define o quanto adicionar em cada lado
        x_padding = (width * padding_factor) / 2
        y_padding = (height * padding_factor) / 2
        
        # Aplica o padding
        x_min_padded = x_min - x_padding
        y_min_padded = y_min - y_padding
        x_max_padded = x_max + x_padding
        y_max_padded = y_max + y_padding
        
        # Garante (Clamping) que as coordenadas fiquem entre [0.0, 1.0]
        x_min_padded = np.clip(x_min_padded, 0.0, 1.0)
        y_min_padded = np.clip(y_min_padded, 0.0, 1.0)
        x_max_padded = np.clip(x_max_padded, 0.0, 1.0)
        y_max_padded = np.clip(y_max_padded, 0.0, 1.0)
        
        rois.append([x_min_padded, y_min_padded, x_max_padded, y_max_padded])
    return np.array(rois, dtype=np.float32)

# --- FUNÇÃO PRINCIPAL ADAPTADA ---
def preprocess_casme2_landmarks(
    base_dir, 
    coding_sheet_path, 
    output_dir, 
    target_size=(224, 224),
    padding=0.2,
    generate_pipeline_images=False,
    pipeline_images_output_dir=None
):
    """
    Processa o dataset CASME2 para extrair landmarks faciais (ROIs)
    do frame de ÁPICE, agora com 8 ROIs e padding.
    Gera imagens de pipeline para depuração se 'generate_pipeline_images' for True.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if generate_pipeline_images:
        if pipeline_images_output_dir is None:
            raise ValueError("pipeline_images_output_dir deve ser especificado quando generate_pipeline_images é True.")
        os.makedirs(pipeline_images_output_dir, exist_ok=True)
        print(f"Imagens do pipeline serão salvas em: {pipeline_images_output_dir}")

    df = pd.read_excel(coding_sheet_path)

    print(f"Iniciando pré-processamento de LANDMARKS com {len(ROI_INDICES)} ROIs e padding de {padding*100}%...")
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        min_detection_confidence=0.5
    )

    try:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processando ROIs"):
            subject = f"sub{int(row['Subject']):02d}"
            filename = row['Filename']
            onset_frame_num = int(row['OnsetFrame'])
            
            apex_frame_raw = row['ApexFrame']
            try:
                apex_frame_num = int(apex_frame_raw)
            except ValueError:
                apex_frame_num = -1 
            
            sample_path = os.path.join(base_dir, subject, filename)
            
            if not os.path.isdir(sample_path):
                continue

            all_frames = sorted(glob(os.path.join(sample_path, '*.jpg')))
            
            if not all_frames:
                 continue
            
            apex_img_index = apex_frame_num - onset_frame_num
            
            # Abordagem do Frame do Meio
            if apex_frame_num == -1:
                apex_img_index = len(all_frames) // 2
                print(f"Aviso: Apex inválido em {sample_path}. Interpolando para o frame do MEIO (Índice {apex_img_index}).")
            else:
                apex_img_index = apex_frame_num - onset_frame_num

            try:
                # se o apex calculado for maior que a qtd de frames, pega o do meio também
                if apex_img_index >= len(all_frames) or apex_img_index < 0:
                    apex_img_index = len(all_frames) // 2
                    print(f"Aviso: Índice calculado fora do alcance em {sample_path}. Interpolando para o frame do MEIO.")
                    
                raw_apex_img_path = all_frames[apex_img_index]
            except Exception:
                raw_apex_img_path = all_frames[len(all_frames) // 2]

            frame_apex_bgr = cv2.imread(raw_apex_img_path, cv2.IMREAD_COLOR)

            if frame_apex_bgr is None:
                # print(f"Erro ao ler imagem (ápex) para {sample_path}. Pulando.")
                continue
                
            frame_apex_bgr = cv2.resize(frame_apex_bgr, target_size)
            frame_apex_rgb = cv2.cvtColor(frame_apex_bgr, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_apex_rgb)

            if not results.multi_face_landmarks:
                # print(f"Aviso: Nenhuma face detectada em {subject}/{filename}. Pulando.")
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            
            rois_norm = get_rois_from_landmarks(landmarks, padding_factor=padding) 
            
            # Garante que temos n ROIs
            if rois_norm.shape[0] != len(ROI_INDICES):
                # print(f"Aviso: Erro ao gerar ROIs para {subject}/{filename}. Pulando.")
                continue
                
            output_sample_path = os.path.join(output_dir, subject, filename)
            if not os.path.exists(output_sample_path):
                os.makedirs(output_sample_path)

            output_rois_path = os.path.join(output_sample_path, 'rois.npy')
            np.save(output_rois_path, rois_norm)

            # Gerar imagem do pipeline para esta amostra ---
            if generate_pipeline_images:
                flow_u_path = os.path.join(output_sample_path, 'flow_u.png')
                if not os.path.exists(flow_u_path):
                    # Se o flow_u.png não existir, pulamos a imagem do pipeline
                    print(f"AVISO: flow_u.png não encontrado para {subject}/{filename}. Imagem de pipeline não será gerada.")
                else:
                    pipeline_image_filename = f"{subject}_{filename}_pipeline.png"
                    pipeline_image_full_path = os.path.join(pipeline_images_output_dir, pipeline_image_filename)
                    plot_sample_pipeline(
                        raw_apex_img_path, 
                        flow_u_path, 
                        rois_norm, 
                        pipeline_image_full_path, 
                        subject, 
                        filename, 
                        target_size
                    )
    finally:
        face_mesh.close()

    print("Pré-processamento de LANDMARKS concluído.")

def plot_sample_pipeline(
    raw_apex_img_path, 
    flow_u_img_path, 
    rois_norm, 
    output_image_path, 
    subject_str, 
    filename, 
    target_size=(224, 224)
):
    """
    Gera e salva uma imagem de 1x4 do pipeline de pré-processamento para uma amostra.
    """
    try:
        # Carregar Imagem 1: Frame do Ápice Original (RGB)
        img_apex_raw = cv2.imread(raw_apex_img_path)
        img_apex_raw = cv2.cvtColor(img_apex_raw, cv2.COLOR_BGR2RGB)
        img_apex_resized = cv2.resize(img_apex_raw, target_size)

        # Carregar Imagem 2: Fluxo Óptico (DUALTVL1 - U)
        img_flow = cv2.imread(flow_u_img_path, cv2.IMREAD_GRAYSCALE)
        
        # Função auxiliar para desenhar as ROIs
        def draw_rois_on_image(image, rois_normalized, color=(255, 0, 0), thickness=1):
            img_with_rois = image.copy()
            if len(img_with_rois.shape) == 2: # Se for grayscale, converte para RGB para desenhar em cor
                img_with_rois = cv2.cvtColor(img_with_rois, cv2.COLOR_GRAY2RGB)
            
            H, W = img_with_rois.shape[:2]
            for roi in rois_normalized:
                x1 = int(roi[0] * W)
                y1 = int(roi[1] * H)
                x2 = int(roi[2] * W)
                y2 = int(roi[3] * H)
                cv2.rectangle(img_with_rois, (x1, y1), (x2, y2), color, thickness) 
            return img_with_rois

        # Imagem 3: Ápice com Landmarks
        img_apex_with_landmarks = draw_rois_on_image(img_apex_resized, rois_norm)
        
        # Imagem 4: Fluxo com Landmarks
        img_flow_with_landmarks = draw_rois_on_image(img_flow, rois_norm)

        # --- Plotar e Salvar ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_apex_resized)
        axes[0].set_title('1. Frame Ápice (Original)')
        axes[0].axis('off')
        
        axes[1].imshow(img_flow, cmap='gray')
        axes[1].set_title('2. Fluxo Óptico (DUALTVL1 - U)')
        axes[1].axis('off')
        
        axes[2].imshow(img_apex_with_landmarks)
        axes[2].set_title('3. Ápice c/ Landmarks (ROIs)')
        axes[2].axis('off')
        
        axes[3].imshow(img_flow_with_landmarks)
        axes[3].set_title('4. Fluxo c/ Landmarks (ROIs)')
        axes[3].axis('off')
        
        fig.suptitle(f"Pipeline de Pré-processamento: {subject_str}/{filename}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plt.savefig(output_image_path)
        plt.close(fig) # Fecha a figura para liberar memória
        
    except Exception as e:
        print(f"AVISO: Não foi possível gerar imagem de pipeline para {subject_str}/{filename}. Erro: {e}")


if __name__ == "__main__":
    CASME2_SELECTED_RAW_PATH = './CASME2_RAW_selected'
    CODING_SHEET_PATH = './CASME2-coding-20140508.xlsx'
    
    # O diretório 'processed_data_DUALTVL1_roi_8' é onde os ROIs e fluxo serão salvos.
    # Certifique-se de que este é o mesmo diretório que você passa para o train_loso_apple.py
    # OUTPUT_DATA_PATH = './processed_data'
    OUTPUT_DATA_PATH = './processed_data_DUALTVL1_melhorado'
    
    PADDING_PERCENT = 0.2 # 20% de padding

    # Configuração para gerar imagens de pipeline
    GENERATE_PIPELINE_IMAGES = True # Defina como True para gerar as imagens, False para desativar
    PIPELINE_IMAGES_OUTPUT_DIR = './pipeline_subjects_dual_tvl1_roi_images' # Novo diretório

    preprocess_casme2_landmarks(
        CASME2_SELECTED_RAW_PATH, 
        CODING_SHEET_PATH, 
        OUTPUT_DATA_PATH,
        padding=PADDING_PERCENT,
        generate_pipeline_images=GENERATE_PIPELINE_IMAGES,
        pipeline_images_output_dir=PIPELINE_IMAGES_OUTPUT_DIR
    )
