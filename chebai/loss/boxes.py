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

        width = (r - l) / 2
        r_fp = r + 0.1 * width
        r_fn = r - 0.1 * width

        l_fp = l - 0.1 * width
        l_fn = l + 0.1 * width

        inside = ((l < points) * (points < r))
        inside_fp = (l_fp < points) * (points < r_fp)
        inside_fn = (l_fn < points) * (points < r_fn)

        fn_per_dim = ~inside_fn * target
        fp_per_dim = inside_fp * (1 - target)

        false_per_dim = fn_per_dim + fp_per_dim
        number_of_false_dims = torch.sum(false_per_dim, dim=-1, keepdim=True)

        dl = torch.abs(l - points)
        dr = torch.abs(r - points)

        closer_to_l_than_r = dl < dr

        r_scale_fp = number_of_false_dims * torch.rand_like(fp_per_dim) * (fp_per_dim * ~closer_to_l_than_r)
        l_scale_fp = number_of_false_dims * torch.rand_like(fp_per_dim) * (fp_per_dim * closer_to_l_than_r)

        r_scale_fn = number_of_false_dims * torch.rand_like(fn_per_dim) * (fn_per_dim * ~closer_to_l_than_r)
        l_scale_fn = number_of_false_dims * torch.rand_like(fn_per_dim) * (fn_per_dim * closer_to_l_than_r)

        r_loss = torch.mean(torch.sum(torch.abs(r_scale_fp * (r_fp - points)), dim=-1) + torch.sum(
            torch.abs(r_scale_fn * (r_fn - points)), dim=-1))
        l_loss = torch.mean(torch.sum(torch.abs(l_scale_fp * (l_fp - points)), dim=-1) + torch.sum(
            torch.abs(l_scale_fn * (l_fn - points)), dim=-1))
        return l_loss + r_loss

    def _calculate_implication_loss(self, l, r):
        capped_difference = torch.relu(l - r)
        return torch.mean(
            torch.sum(
                (torch.softmax(capped_difference, dim=-1) * capped_difference), dim=-1
            ),
            dim=0,
        )
