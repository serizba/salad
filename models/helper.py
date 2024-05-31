from models import aggregators
from models import backbones


def get_backbone(
        backbone_arch='dinov2',
        backbone_config={}
    ):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional)
        backbone_config (dict, optional): this must contain all the arguments needed to instantiate the backbone class. Defaults to {}.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'dinov2' in backbone_arch.lower():
        return backbones.DINOv2(model_name=backbone_arch, **backbone_config)


def get_aggregator(agg_arch='salad', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    if 'salad' in agg_arch.lower():
        assert 'num_channels' in agg_config
        assert 'num_clusters' in agg_config
        assert 'cluster_dim' in agg_config
        assert 'token_dim' in agg_config
        return aggregators.SALAD(**agg_config)
