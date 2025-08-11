from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
import torch
from pytorch_msssim import ssim, ms_ssim

from src.modules.data.metadataframe.metadataframe import MetadataFrame

class RegularizationLossFunction(torch.nn.Module):
    def __init__(self, config, experiment_execution_paths):
        super(RegularizationLossFunction, self).__init__()
        self.config = config

        self.weights = self._get_label_weights(experiment_execution_paths)
        print(f"Label weights: {self.weights}")

        # For dynamic loss scaling
        self.register_buffer("surv_avg", torch.tensor(1.0))
        self.register_buffer("recon_avg", torch.tensor(1.0))
        self.register_buffer("kld_avg", torch.tensor(1.0))
        self.momentum = 0.99  # smoothing factor for running averages

    def forward(self, surv_output, surv_target, recon_output, recon_target, mu, logvar, lambda_rec=0.5, alpha = 1.0, beta = 0.5, anneal=1.0):

        # Survival Loss 
        surv_loss = binary_cross_entropy_with_logits(
            input=surv_output,
            target=surv_target,
            
            weight=self.weights.to(surv_output.device)
        )

        # Reconstruction Loss (SSIM + MS-SSIM)
        ssim_val = 1 - ssim(recon_output, recon_target, data_range=1.0, size_average=True)
        mssim_val = 1 - ms_ssim(recon_output, recon_target, data_range=1.0, size_average=True)
        recon_loss_no_kld = lambda_rec * ssim_val + (1 - lambda_rec) * mssim_val

        # KLD Loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / recon_output.size(0)

        # Update running averages
        self.surv_avg = self.momentum * self.surv_avg + (1 - self.momentum) * surv_loss.item()
        self.recon_avg = self.momentum * self.recon_avg + (1 - self.momentum) * recon_loss_no_kld.item()
        self.kld_avg = self.momentum * self.kld_avg + (1 - self.momentum) * kld_loss.item()

        # Normalize losses by their running averages
        surv_loss = surv_loss / (self.surv_avg + 1e-8)
        recon_loss_no_kld = recon_loss_no_kld / (self.recon_avg + 1e-8)
        kld_loss = kld_loss / (self.kld_avg + 1e-8)

        recon_loss = recon_loss_no_kld + anneal * kld_loss

        loss = alpha * surv_loss + beta * recon_loss
        return loss, surv_loss, recon_loss_no_kld, kld_loss, mu, logvar

    def _get_label_weights(self, experiment_execution_paths):
        if self.config.apply_weights:
            metadataframe = MetadataFrame(
                config=self.config.metadataframe,
                experiment_execution_paths=experiment_execution_paths
            )
            lung_nodule_metadataframe = \
                metadataframe.get_lung_metadataframe()

            label_counts = \
                lung_nodule_metadataframe['label'].value_counts().sort_index()
            pos_weight = label_counts[0] / label_counts[1]
            return torch.tensor([pos_weight]).to(self.config.device)
        
        else:
            label_weights = torch.tensor([1.0, 1.0]).to(self.config.device)
        return label_weights