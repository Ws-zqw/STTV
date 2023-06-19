import torch
import torch.nn.functional as F
from torch import nn

class InfoNCE(nn.Module):
    def __init__(self, initial_temperature=0.1):
        super(InfoNCE, self).__init__()
        # self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        self.temperature = initial_temperature

    def forward(self, inputs_q, inputs_k):
        loss_1, mean_pos_sim_1, mean_neg_sim_1 = self.single_forward(inputs_q, inputs_k)
        loss_2, mean_pos_sim_2, mean_neg_sim_2 = self.single_forward(inputs_k, inputs_q)
        return (loss_1+loss_2)*0.5, (mean_pos_sim_1+mean_pos_sim_2)*0.5, (mean_neg_sim_1+mean_neg_sim_2)*0.5

    def single_forward(self, input_q, input_k):
        N, C = input_q.size()

        normalized_input_q = input_q / torch.norm(input_q, dim=1, keepdim=True)
        normalized_input_k = input_k / torch.norm(input_k, dim=1, keepdim=True)
        sim_mat = torch.matmul(normalized_input_q, normalized_input_k.t())

        eyes_ = torch.eye(N).cuda()
        pos_mask = eyes_.eq(1)
        neg_mask = ~pos_mask
        pos_sim = torch.masked_select(sim_mat, pos_mask)
        neg_sim = torch.masked_select(sim_mat, neg_mask)

        # labels = torch.arange(N, dtype=torch.long).cuda()
        # loss_fn = LabelSmoothingCrossEntropyLoss(smoothing=0.02, num_classes=N)
        # loss = loss_fn(sim_mat / self.temperature, labels)
        loss = -torch.log(torch.exp(pos_sim / self.temperature) / torch.exp(sim_mat / self.temperature).sum(dim=-1))
        loss = loss.mean()

        mean_pos_sim = pos_sim.mean().item()
        mean_neg_sim = neg_sim.mean().item()

        return loss, mean_pos_sim, mean_neg_sim

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.0, num_classes=10):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing

    def forward(self, inputs, targets):
        one_hot = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1).long(), 1)
        smooth_targets = one_hot * self.confidence + (1 - one_hot) * self.smoothing / (self.num_classes - 1)
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        loss = (-smooth_targets * log_probs).sum(dim=1).mean()
        return loss
