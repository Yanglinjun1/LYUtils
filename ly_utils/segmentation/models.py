##########################################################################################
# Description: functions that allows to create segmentation models of one's choice from
# different sources: MONAI, TIMM, etc.
##########################################################################################

import torch
import segmentation_models_pytorch as smp


def create_seg_model(model_name, in_channels, out_channels, **kwargs):
    """
    Create a segmentation model based on the given parameters. You should provide the model name,
    the number of input channels and output_channels. You should specific NO activation function at
    the end to make losses and metrics work. The function will return
    the corresponding segmentation model. Right now only supports smp Unet models.
    You can also provide the path to the pre-trained weights if you want to load them by passing it as
    "weights_path". More models will be implemented in the future.
    Args:
        model_name (str): Name of the segmentation model.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        **kwargs: Additional keyword arguments for model configuration.

    Returns:
        torch.nn.Module: Segmentation model.

    """
    model_kwargs = dict()
    if model_name == "SMPUNet":
        model_class = smp.Unet
        model_kwargs["in_channels"] = in_channels
        model_kwargs["classes"] = out_channels

        model_kwargs["encoder_name"] = kwargs.get("encoder_name", "efficientnet-b0")
        model_kwargs["decoder_attention_type"] = kwargs.get(
            "decoder_attention_type", "scse"
        )
        model_kwargs["activation"] = kwargs.get("activation", None)
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    model = model_class(**model_kwargs)

    weight_path = kwargs.get("weight_path", None)
    if weight_path:
        print("Trying to load pretrained model weights")
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained model weights loaded.")

    return model
