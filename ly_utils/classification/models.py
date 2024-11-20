##########################################################################################
# Description: functions that allows to create classification models out of a specific
# backbone of one's choice, e.g., TIMM, etc.
##########################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch.model import EfficientNet, load_pretrained_weights
from timm import create_model
from timm.models import load_checkpoint


def create_cls_model(model_name, in_channels, label_index, branch_dict, **kwargs):
    """
    Create a classification model based on the given parameters. You should provide the model name,
    the number of input channels, the label index, and the branch dictionary. The function will return
    the corresponding classification model. Right now only supports EfficientNet and ConvNextV2 models.
    You can also provide the path to the pre-trained weights if you want to load them by passing it as
    "weights_path". More models, such as MultiBranchViT, will be implemented in the future.

    Args:
        model_name (str): The name of the model.
        in_channels (int): The number of input channels.
        label_index (list): The list of label indices.
        branch_dict (dict): The dictionary containing branch information.
        **kwargs: Additional keyword arguments.

    Returns:
        model: The created classification model.
    """

    model_kwargs = dict()
    model_kwargs["model_name"] = model_name
    model_kwargs["in_channels"] = in_channels
    model_kwargs["num_classes"] = len(label_index)
    model_kwargs["label_index"] = label_index
    model_kwargs["branch_dict"] = branch_dict
    weight_path = kwargs.get("weights_path", None)

    if model_name in [
        "efficientnet-b0",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
        "efficientnet-b6",
        "efficientnet-b7",
    ]:

        if weight_path:
            model_kwargs["weights_path"] = weight_path
            model = MultiBranchEfficientNet.from_pretrained(**model_kwargs)
        else:
            model = MultiBranchEfficientNet.from_name(**model_kwargs)

    elif model_name in ["convnextv2_tiny.fcmae_ft_in22k_in1k_384"]:
        model_kwargs["weights_path"] = weight_path
        model = MultiBranchConvNext(**model_kwargs)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    return model


class MultiBranchEfficientNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__(blocks_args=blocks_args, global_params=global_params)
        self.output_branch = None

    def forward(self, inputs):
        x = super().forward(inputs=inputs)

        output = dict()
        for branch_name, branch_index in self.branch_dict.items():
            output[branch_name] = x[:, branch_index]

        if self.output_branch is not None:
            return output[self.output_branch]
        else:
            return output

    def get_features(self, inputs):
        x = self.extract_features(inputs=inputs)
        x = self._avg_pooling(x)

        return x

    def set_output_branch(self, output_branch=None):
        self.output_branch = output_branch

    @classmethod
    def from_name(
        cls, model_name, label_index, branch_dict, in_channels=3, **override_params
    ):
        # call the super class constructor
        model = super(MultiBranchEfficientNet, cls).from_name(
            model_name, in_channels, **override_params
        )

        # assign the label index to the model
        model.label_index = label_index.copy()
        model.label_index_reverse = {value: key for key, value in label_index.items()}

        # assign the branch dictionary to the model
        model.branch_dict = branch_dict.copy()

        return model

    @classmethod
    def from_pretrained(
        cls,
        model_name,
        label_index,
        branch_dict,
        weights_path=None,
        advprop=False,
        in_channels=3,
        num_classes=1000,
        **override_params,
    ):
        model = cls.from_name(
            model_name,
            label_index=label_index,
            branch_dict=branch_dict,
            num_classes=num_classes,
            in_channels=in_channels,
            **override_params,
        )
        load_pretrained_weights(
            model,
            model_name,
            weights_path=weights_path,
            load_fc=(num_classes == 1000),
            advprop=advprop,
        )
        model._change_in_channels(in_channels)

        return model


class MultiBranchConvNext(nn.Module):
    def __init__(
        self,
        model_name,
        label_index,
        branch_dict,
        pooling="avg",
        in_channels=1,
        num_classes=8,
        **kwargs,
    ):
        super().__init__()

        # Load the timm classification model
        self.model = create_model(
            model_name=model_name, pretrained=False, in_chans=in_channels
        )
        weights_path = kwargs.get("weights_path", None)
        if weights_path:  # load the pre-trained weights if there is any
            _ = load_checkpoint(
                self.model, weights_path, strict=False
            )  # strict set to False!!
            print("Pretrained weights successfully loaded.")

        # fully connected layer
        self.num_in_features = self.model.num_features
        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.num_in_features),
            nn.Linear(
                in_features=self.num_in_features, out_features=num_classes, bias=False
            ),
            nn.LeakyReLU(),
        )

        # assign the label index to the model
        self.label_index = label_index.copy()
        self.label_index_reverse = {value: key for key, value in label_index.items()}

        # assign the branch dictionary to the model
        self.branch_dict = branch_dict.copy()
        self.output_branch = None

        self.pooling = pooling

    def forward(self, inputs):
        embeddings = self.model.forward_features(inputs)
        poolings = self._pool(embeddings)
        x = self.fc(poolings)

        output = dict()
        for branch_name, branch_index in self.branch_dict.items():
            output[branch_name] = x[:, branch_index]

        if self.output_branch is not None:
            return output[self.output_branch]
        else:
            return output

    def get_features(self, inputs):
        embeddings = self.model.forward_features(inputs)
        poolings = self._pool(embeddings)
        return poolings

    def _pool(self, x):
        if self.pooling == "avg":
            return F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif self.pooling == "max":
            return F.adaptive_max_pool2d(x, 1).flatten(1)
        elif self.pooling == "catavgmax":
            self.num_in_features = self.num_in_features * 2
            avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
            max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
            return torch.cat([avg_pool, max_pool], 1)
        else:
            raise NotImplementedError("Pooling type not implemented!")

    def set_output_branch(self, output_branch=None):
        self.output_branch = output_branch
