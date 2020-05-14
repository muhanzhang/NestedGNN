import torch
import pdb


class Distance(object):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes.

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, norm=True, max_value=None, cat=True, relative_pos=False, 
                 squared=False):
        self.norm = norm
        self.max = max_value
        self.cat = cat
        self.relative_pos = relative_pos
        self.squared = squared

    def __call__(self, data):
        if type(data) == dict:
            return {key: self.__call__(data_) for key, data_ in data.items()}

        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        if self.squared:
            dist = ((pos[col] - pos[row]) ** 2).sum(1).view(-1, 1)
        else:
            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        if self.relative_pos:
            relative_pos = pos[col] - pos[row]
            data.edge_attr = torch.cat([data.edge_attr, relative_pos], dim=-1)

        if "original_edge_index" in data:
            (row, col), pos, pseudo = (
                data.original_edge_index, data.original_pos, data.original_edge_attr
            )

            dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)

            if self.norm and dist.numel() > 0:
                dist = dist / (dist.max() if self.max is None else self.max)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.original_edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
            else:
                data.original_edge_attr = dist

        return data

    def __repr__(self):
        return '{}(norm={}, max_value={})'.format(self.__class__.__name__,
                                                  self.norm, self.max)
