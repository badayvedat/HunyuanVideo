from diffusers.loaders import HunyuanVideoLoraLoaderMixin as DiffusersLoraLoaderMixin
from unittest.mock import patch

# patching the original function
# https://github.com/huggingface/diffusers/blob/196aef5a6f76e1ad6ba889184860c3633d166910/src/diffusers/loaders/lora_conversion_utils.py#L128
def _convert_hunyuan_video_lora_to_diffusers(original_state_dict):
    print("called fake conversion function", original_state_dict)
    return original_state_dict

class HunyuanVideoLoraLoaderMixin(DiffusersLoraLoaderMixin):
    def load_lora_state_dict(*args, **kwargs):
        with patch(
            "diffusers.loaders.lora_pipeline._convert_hunyuan_video_lora_to_diffusers",
            _convert_hunyuan_video_lora_to_diffusers,
        ):
            return super().load_lora_state_dict(*args, **kwargs)


