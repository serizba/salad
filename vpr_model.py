import torch
from models import helper


class VPRModel(torch.nn.Module):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='resnet50',
        backbone_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):

        # Check input size and resize if necessary
        if x.shape[-1] % 14 != 0 or x.shape[-2] % 14 != 0:
            # Default inference size is 322x322
            x = torch.nn.functional.interpolate(x, size=(322, 322), mode='bilinear')

        x = self.backbone(x)
        x = self.aggregator(x)
        return x
