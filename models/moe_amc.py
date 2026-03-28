import torch
import torch.nn as nn
import torch.nn.functional as F
from models.snr_estimator import SNREstimator
from models.expert_networks import ExpertCNN
from models.gating_network import GatingNetwork, SignalAwareGatingNetwork


class MoEAMC(nn.Module):
    def __init__(self, num_experts=3, num_classes=8, input_channels=2,
                 expert_filters=[64, 128, 256], gating_mode='soft',
                 use_signal_aware_gating=True):
        """
        Mixture of Experts Automatic Modulation Classification

        Args:
            num_experts: Number of expert networks
            num_classes: Number of modulation classes
            input_channels: Number of input channels (2 for I/Q)
            expert_filters: Filter sizes for expert CNNs
            gating_mode: 'soft' for weighted combination, 'hard' for single expert selection
            use_signal_aware_gating: Use signal-aware gating for better routing
        """
        super(MoEAMC, self).__init__()

        self.num_experts = num_experts
        self.num_classes = num_classes
        self.gating_mode = gating_mode
        self.use_signal_aware_gating = use_signal_aware_gating

        # SNR Estimator — outputs RAW LOGITS (no softmax)
        self.snr_estimator = SNREstimator(
            input_channels=input_channels,
            hidden_dims=[128, 64],
            output_dim=num_experts
        )

        # Expert Networks
        self.experts = nn.ModuleList([
            ExpertCNN(
                input_channels=input_channels,
                num_classes=num_classes,
                filters=expert_filters
            ) for _ in range(num_experts)
        ])

        # Gating Network
        if use_signal_aware_gating:
            self.gating = SignalAwareGatingNetwork(
                input_channels=input_channels,
                num_experts=num_experts,
                hidden_dims=[128, 64]
            )
        else:
            self.gating = GatingNetwork(
                num_experts=num_experts,
                hidden_dims=[128, 64]
            )

    def forward(self, x, return_expert_outputs=False):
        """
        Args:
            x: Input I/Q signal (batch_size, 2, sample_length)
            return_expert_outputs: If True, return individual expert outputs
        Returns:
            Final classification logits (batch_size, num_classes)
        """
        batch_size = x.size(0)

        # Estimate SNR bin logits (raw, no softmax)
        snr_logits = self.snr_estimator(x)

        # Get gating weights
        if self.gating_mode == 'hard':
            # Hard selection: choose expert with highest SNR probability
            selected = torch.argmax(snr_logits, dim=1)
            gating_weights = F.one_hot(selected, num_classes=self.num_experts).float()
        else:
            # Soft gating: weighted combination
            if self.use_signal_aware_gating:
                gating_weights = self.gating(x, snr_logits)
            else:
                gating_weights = self.gating(snr_logits)

        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)

        # Stack expert outputs (batch_size, num_experts, num_classes)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Weighted combination of expert outputs
        # gating_weights: (batch_size, num_experts, 1)
        # expert_outputs: (batch_size, num_experts, num_classes)
        gating_weights_expanded = gating_weights.unsqueeze(2)
        final_output = torch.sum(expert_outputs * gating_weights_expanded, dim=1)

        if return_expert_outputs:
            return final_output, expert_outputs, gating_weights, snr_logits

        return final_output

    def predict(self, x):
        """
        Get predicted class labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_load_balance_loss(self, gating_weights):
        """
        Compute load-balance loss to encourage equal usage of experts.
        Prevents expert collapse where one expert dominates.
        """
        # Average gating weight per expert across the batch
        avg_weights = gating_weights.mean(dim=0)  # (num_experts,)
        # Uniform target
        uniform = torch.ones_like(avg_weights) / self.num_experts
        # KL divergence from uniform
        loss = F.kl_div(
            (avg_weights + 1e-8).log(), uniform, reduction='batchmean'
        )
        return loss

    def get_diversity_loss(self, expert_outputs):
        """
        Encourage expert outputs to be diverse (different from each other).
        This prevents all experts from learning the same thing.
        """
        # expert_outputs: (batch, num_experts, num_classes)
        # Compute pairwise cosine similarity between experts
        expert_probs = F.softmax(expert_outputs, dim=2)  # (batch, E, C)
        
        diversity_loss = 0.0
        count = 0
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                # Cosine similarity between expert i and j outputs
                sim = F.cosine_similarity(
                    expert_probs[:, i, :], expert_probs[:, j, :], dim=1
                )
                diversity_loss += sim.mean()
                count += 1
        
        if count > 0:
            diversity_loss /= count
        
        return diversity_loss


class MoEAMCWithAnalysis(MoEAMC):
    def __init__(self, *args, **kwargs):
        super(MoEAMCWithAnalysis, self).__init__(*args, **kwargs)

    def forward_with_analysis(self, x):
        """
        Forward pass with detailed analysis
        Returns predictions, expert contributions, SNR estimates
        """
        batch_size = x.size(0)

        # Estimate SNR
        snr_logits = self.snr_estimator(x)
        snr_probs = F.softmax(snr_logits, dim=1)

        # Get gating weights
        if self.gating_mode == 'hard':
            selected = torch.argmax(snr_logits, dim=1)
            gating_weights = F.one_hot(selected, num_classes=self.num_experts).float()
        else:
            if self.use_signal_aware_gating:
                gating_weights = self.gating(x, snr_logits)
            else:
                gating_weights = self.gating(snr_logits)

        # Expert outputs
        expert_outputs = []
        expert_predictions = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
            expert_predictions.append(torch.argmax(output, dim=1))

        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_predictions = torch.stack(expert_predictions, dim=1)

        # Final prediction
        gating_weights_expanded = gating_weights.unsqueeze(2)
        final_output = torch.sum(expert_outputs * gating_weights_expanded, dim=1)
        final_prediction = torch.argmax(final_output, dim=1)

        return {
            'final_prediction': final_prediction,
            'final_logits': final_output,
            'expert_predictions': expert_predictions,
            'expert_outputs': expert_outputs,
            'gating_weights': gating_weights,
            'snr_probs': snr_probs
        }