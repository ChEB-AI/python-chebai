import torch

class BoxLoss(torch.nn.Module):
    def __init__(
        self, base_loss: torch.nn.Module = None
    ):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, input, target, **kwargs):
        b = input["boxes"]
        points = input["embedded_points"]
        target = target.float().unsqueeze(-1)
        l, lind = torch.min(b, dim=-1)
        r, rind = torch.max(b, dim=-1)

        widths = r - l

        l += 0.1*widths
        r -= 0.1 * widths
        inside = ((l < points) * (points < r)).float()
        closer_to_l_than_to_r = (torch.abs(l - points) < torch.abs(r - points)).float()
        fn_per_dim = ((1 - inside) * target)
        fp_per_dim = (inside * (1 - target))
        diff = torch.abs(fp_per_dim - fn_per_dim)
        return self.base_loss(diff * closer_to_l_than_to_r * points,  diff * closer_to_l_than_to_r * l) + self.base_loss(
            diff * (1 - closer_to_l_than_to_r) * points,  diff * (1 - closer_to_l_than_to_r) * r)

    def _calculate_implication_loss(self, l, r):
        capped_difference = torch.relu(l - r)
        return torch.mean(
            torch.sum(
                (torch.softmax(capped_difference, dim=-1) * capped_difference), dim=-1
            ),
            dim=0,
        )
