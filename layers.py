import torch


def diag_offdiag_maxpool(input):
    N = input.shape[-1]

    max_diag = torch.max(torch.diagonal(input, dim1=-2, dim2=-1), dim=2)[0]  # BxS

    # with torch.no_grad():
    max_val = torch.max(max_diag)
    min_val = torch.max(-1 * input)
    val = torch.abs(torch.add(max_val, min_val))

    min_mat = torch.mul(val, torch.eye(N, device=input.device)).view(1, 1, N, N)

    max_offdiag = torch.max(torch.max(input - min_mat, dim=3)[0], dim=2)[0]  # BxS

    return torch.cat((max_diag, max_offdiag), dim=1)  # output Bx2S
