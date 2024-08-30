##########################################################################################
# Description: functions that allows to create segmentation models of one's choice from
# different sources: MONAI, TIMM, etc.
##########################################################################################

import segmentation_models_pytorch.Unet as smp_unet


def create_seg_model(model_name, in_channels, out_channels, **kwargs):

    model_class = seg_model_dict[model_name]

    model_kwargs = dict()
    if model_name == "SMPUNet":
        model_class = smp_unet
        model_kwargs["in_channels"] = in_channels
        model_kwargs["classes"] = out_channels

        model_kwargs["encoder_name"] = kwargs.get("encoder_name", "efficientnet-b0")
        model_kwargs["decoder_attention_type"] = kwargs.get(
            "decoder_attention_type", "scse"
        )
        model_kwargs["activation"] = kwargs.get("activation", None)

    model = model_class(**model_kwargs)

    weight_path = kwargs.get("weight_path", None)
    if weight_path:
        print("Trying to load pretrained model weights")
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained model weights.")

    return model
