import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # For malware graph analysis
from transformers import BertModel  # For CVE text analysis

class JointThreatModel(nn.Module):
    """
    Multi-modal threat detection model combining:
    - LSTM Autoencoder for network anomaly detection
    - Graph Attention Network for malware analysis
    - Transformer for CVE vulnerability assessment
    """
    
    def __init__(self, 
                 network_feat_dim=120,
                 malware_feat_dim=2382,
                 cve_feat_dim=12,
                 latent_dim=256):
        super().__init__()
        
        # Network traffic branch (LSTM Autoencoder)
        self.network_encoder = nn.LSTM(
            input_size=network_feat_dim,
            hidden_size=latent_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.network_decoder = nn.LSTM(
            input_size=latent_dim*2,
            hidden_size=network_feat_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Malware analysis branch (Graph Attention Network)
        self.malware_gat1 = GATConv(malware_feat_dim, latent_dim, heads=3)
        self.malware_gat2 = GATConv(latent_dim*3, latent_dim)
        
        # CVE analysis branch (Transformer)
        self.cve_bert = BertModel.from_pretrained('bert-base-uncased')
        self.cve_proj = nn.Linear(768, latent_dim)  # BERT hidden size to latent
        
        # Joint threat assessment
        self.fusion = nn.Linear(latent_dim*3, 512)
        self.threat_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Binary classification: threat/normal
        )
        
        # Auxiliary outputs
        self.anomaly_scorer = nn.Linear(latent_dim*2, 1)  # For network anomalies
        self.malware_scorer = nn.Linear(latent_dim, 1)    # For malware detection
        
    def forward(self, network_data, malware_data, cve_data):
        """
        Forward pass for multi-modal threat analysis
        
        Args:
            network_data: Tensor of shape (batch, seq_len, network_feat_dim)
            malware_data: Tuple (x, edge_index) for graph data
            cve_data: Tensor of CVE text embeddings
        
        Returns:
            dict: {
                'threat_pred': main threat prediction,
                'anomaly_score': network anomaly score,
                'malware_score': malware detection score,
                'reconstruction': network traffic reconstruction
            }
        """
        # Network anomaly detection
        enc_out, (h_n, c_n) = self.network_encoder(network_data)
        network_latent = torch.cat([h_n[-2], h_n[-1]], dim=1)  # Last hidden states
        anomaly_score = torch.sigmoid(self.anomaly_scorer(network_latent))
        
        # Reconstruct for anomaly detection
        dec_out, _ = self.network_decoder(enc_out)
        
        # Malware analysis
        x, edge_index = malware_data
        x = F.leaky_relu(self.malware_gat1(x, edge_index))
        malware_latent = F.elu(self.malware_gat2(x, edge_index))
        malware_score = torch.sigmoid(self.malware_scorer(malware_latent.mean(dim=0)))
        
        # CVE analysis
        cve_embeddings = self.cve_bert(**cve_data).last_hidden_state[:, 0, :]
        cve_latent = F.gelu(self.cve_proj(cve_embeddings))
        
        # Multi-modal fusion
        combined = torch.cat([
            network_latent,
            malware_latent.mean(dim=0).unsqueeze(0).expand(network_latent.size(0), -1),
            cve_latent
        ], dim=1)
        
        fused = F.relu(self.fusion(combined))
        threat_pred = self.threat_classifier(fused)
        
        return {
            'threat_pred': F.softmax(threat_pred, dim=1),
            'anomaly_score': anomaly_score,
            'malware_score': malware_score,
            'reconstruction': dec_out
        }

class ThreatAwareLoss(nn.Module):
    """
    Custom loss function for joint threat modeling with:
    - Classification loss (threat detection)
    - Reconstruction loss (anomaly detection)
    - Malware detection loss
    """
    
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Threat classification weight
        self.beta = beta    # Anomaly detection weight
        self.gamma = gamma  # Malware detection weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict from JointThreatModel
            targets: dict {
                'threat_label': threat classification labels,
                'network_recon': original network data,
                'malware_label': binary malware labels
            }
        """
        # Threat classification loss
        cls_loss = self.ce_loss(outputs['threat_pred'], targets['threat_label'])
        
        # Network anomaly reconstruction loss
        recon_loss = self.mse_loss(outputs['reconstruction'], targets['network_recon'])
        
        # Malware detection loss
        malware_loss = self.bce_loss(outputs['malware_score'], targets['malware_label'])
        
        # Combined loss
        total_loss = (self.alpha * cls_loss + 
                     self.beta * recon_loss + 
                     self.gamma * malware_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': cls_loss,
            'reconstruction_loss': recon_loss,
            'malware_loss': malware_loss
        }

# Utility functions for model handling
def save_model(model, path):
    """Save model with metadata"""
    torch.save({
        'state_dict': model.state_dict(),
        'config': {
            'network_feat_dim': model.network_encoder.input_size,
            'malware_feat_dim': model.malware_gat1.in_channels,
            'cve_feat_dim': model.cve_proj.in_features,
            'latent_dim': model.fusion.in_features // 3
        }
    }, path)

def load_model(path, device='cuda'):
    """Load model with configuration handling"""
    checkpoint = torch.load(path, map_location=device)
    model = JointThreatModel(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device)

# Example usage:
if __name__ == "__main__":
    # Initialize model
    model = JointThreatModel()
    
    # Sample inputs
    network_input = torch.randn(32, 100, 120)  # batch, seq_len, features
    malware_x = torch.randn(100, 2382)         # 100 malware samples
    malware_edge = torch.randint(0, 100, (2, 200))  # Random edges
    cve_input = {'input_ids': torch.randint(0, 1000, (32, 128)),
                'attention_mask': torch.ones(32, 128)}
    
    # Forward pass
    outputs = model(network_input, (malware_x, malware_edge), cve_input)
    print(f"Threat prediction shape: {outputs['threat_pred'].shape}")