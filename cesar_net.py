import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import os

# Bloco Residual
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out) 
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

# Módulo de Atenção Regional
class RegionalAttention(nn.Module):
    def __init__(self, in_channels, n_patches=3, n_heads=4, d_model=128, n_layers=2):
        super(RegionalAttention, self).__init__()
        
        self.n_patches = n_patches
        self.d_model = d_model
        
        self.roi_output_size = (7, 7)
        self.patch_dim = in_channels * self.roi_output_size[0] * self.roi_output_size[1]
        
        self.patch_projection = nn.Linear(self.patch_dim, d_model)
        
        # --- TRAVA DE DETERMINISMO NO TRANSFORMER ---
        # Garantindo que o Dropout do Transformer siga o mesmo rigor
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, 
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x, rois_norm):
        B, C, H, W = x.shape
        
        rois_pixel = rois_norm.clone()
        rois_pixel[:, 1] = rois_norm[:, 1] * W  # x1
        rois_pixel[:, 2] = rois_norm[:, 2] * H  # y1
        rois_pixel[:, 3] = rois_norm[:, 3] * W  # x2
        rois_pixel[:, 4] = rois_norm[:, 4] * H  # y2

        pooled_rois = roi_align(x, rois_pixel, output_size=self.roi_output_size, spatial_scale=1.0)
        pooled_rois_flat = pooled_rois.view(-1, self.patch_dim)
        
        embeddings = self.patch_projection(pooled_rois_flat)
        embeddings_batch = embeddings.view(B, self.n_patches, self.d_model)
        
        # O processamento passa pelo Transformer
        attn_output = self.transformer_encoder(embeddings_batch)
        
        return attn_output.view(B, -1) 

# --- Arquitetura principal da CESAR-Net ---
class CESAR_Net(nn.Module):
    def __init__(self, num_classes, d_model=128, n_heads=4, n_layers=2, dropout_rate=0.2, dropout_classifier=0.5):
        super(CESAR_Net, self).__init__()
        
        # --- TRAVA DE INICIALIZAÇÃO DE PESOS ---
        # Se a flag de determinismo estiver ativa no ambiente, desliga heurísticas rápidas
        if os.environ.get('PYTHONHASHSEED') is not None:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.allow_tf32 = False # Desativa TensorFloat-32 (gera variabilidade)
            
        self.stream_u = self._make_stream(1, [32, 64], dropout_rate)
        self.stream_v = self._make_stream(1, [32, 64], dropout_rate)
        self.stream_strain = self._make_stream(1, [32, 64], dropout_rate)

        self.n_patches = 7
        self.d_model = d_model      
        self.roi_output_size = (7, 7)
  
        self.attention_u = RegionalAttention(64, n_patches=self.n_patches, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
        self.attention_v = RegionalAttention(64, n_patches=self.n_patches, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
        self.attention_strain = RegionalAttention(64, n_patches=self.n_patches, n_heads=n_heads, d_model=d_model, n_layers=n_layers)
        
        fusion_dim = 3 * (self.n_patches * d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim,  256),
            nn.ReLU(True),
            nn.Dropout(dropout_classifier),
            nn.Linear(256, num_classes)
        )

    def _make_stream(self, in_channels, channel_list, dropout_rate):
        layers = []
        layers.append(nn.Conv2d(in_channels, channel_list[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(channel_list[0]))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        in_c = channel_list[0]
        for out_c in channel_list:
            layers.append(ResidualBlock(in_c, out_c, stride=2, dropout_rate=dropout_rate)) 
            in_c = out_c
            
        return nn.Sequential(*layers)

    def forward(self, x_u, x_v, x_strain, rois):
        feat_u = self.stream_u(x_u)
        feat_v = self.stream_v(x_v)
        feat_strain = self.stream_strain(x_strain)

        attn_u = self.attention_u(feat_u, rois)
        attn_v = self.attention_v(feat_v, rois)
        attn_strain = self.attention_strain(feat_strain, rois)
        
        fused_features = torch.cat([attn_u, attn_v, attn_strain], dim=1)
        output = self.classifier(fused_features)
        
        return output