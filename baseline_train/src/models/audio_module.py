import torch
from einops import rearrange
from torch import nn
from .transformer_3d_hallo import Transformer3DModel


class MultiHumanAudioBlock(nn.Module):
    """
    Standard 3D downsampling block for the U-Net architecture. This block performs downsampling
    operations in the U-Net using residual blocks and an optional motion module.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - temb_channels (int): Number of channels for the temporal embedding.
    - dropout (float): Dropout rate for the block.
    - num_layers (int): Number of layers in the block.
    - resnet_eps (float): Epsilon for residual block stability.
    - resnet_time_scale_shift (str): Time scale shift for the residual block's time embedding.
    - resnet_act_fn (str): Activation function used in the residual block.
    - resnet_groups (int): Number of groups for the convolutions in the residual block.
    - resnet_pre_norm (bool): Whether to use pre-normalization in the residual block.
    - output_scale_factor (float): Scaling factor for the block's output.
    - add_downsample (bool): Whether to add a downsampling layer.
    - downsample_padding (int): Padding for the downsampling layer.
    - use_inflated_groupnorm (bool): Whether to use inflated group normalization.
    - use_motion_module (bool): Whether to include a motion module.
    - motion_module_type (str): Type of motion module to use.
    - motion_module_kwargs (dict): Keyword arguments for the motion module.

    Forward method:
    The forward method processes the input hidden states through the residual blocks and optional
    motion modules, followed by an optional downsampling step. It supports gradient checkpointing
    during training to reduce memory usage.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        resnet_groups: int = 32,
        attn_num_head_channels=1,
        audio_attention_dim=768,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        use_audio_module=True,
        depth=0,
        stack_enable_blocks_name=None,
        stack_enable_blocks_depth=None,
        feature_shape=None,
    ):
        super().__init__()
        audio_modules = []
        self.has_cross_attention = True
        self.gradient_checkpointing = False
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            audio_modules.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=audio_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_audio_module=use_audio_module,
                    depth=depth,
                    unet_block_name="up",
                    feature_shape=feature_shape,
                )
                if use_audio_module
                else None
            )
        self.audio_modules = nn.ModuleList(audio_modules)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        
        face_mask_list=None,
        lip_mask_list=None,
        activate_score_list=None,
        
        audio_embedding=None,
        motion_scale=None,
    ):
        """
        Forward pass for the CrossAttnUpBlock3D class.

        Args:
            self (CrossAttnUpBlock3D): An instance of the CrossAttnUpBlock3D class.
            hidden_states (Tensor): The input hidden states tensor.
            res_hidden_states_tuple (Tuple[Tensor]): A tuple of residual hidden states tensors.
            temb (Tensor, optional): The token embeddings tensor. Defaults to None.
            encoder_hidden_states (Tensor, optional): The encoder hidden states tensor. Defaults to None.
            upsample_size (int, optional): The upsample size. Defaults to None.
            attention_mask (Tensor, optional): The attention mask tensor. Defaults to None.
            full_mask (Tensor, optional): The full mask tensor. Defaults to None.
            face_mask (Tensor, optional): The face mask tensor. Defaults to None.
            lip_mask (Tensor, optional): The lip mask tensor. Defaults to None.
            audio_embedding (Tensor, optional): The audio embedding tensor. Defaults to None.

        Returns:
            Tensor: The output tensor after passing through the CrossAttnUpBlock3D.
        """
        for _, (audio_module) in enumerate(self.audio_modules):
            # pop res hidden states
            # res_hidden_states = res_hidden_states_tuple[-1]
            # res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            # hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        return module(*inputs)
                    return custom_forward

                if audio_module is not None:
                    # audio_embedding = audio_embedding
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(audio_module, return_dict=False),
                        hidden_states,
                        audio_embedding,
                        attention_mask,
                        face_mask,
                        lip_mask,
                        motion_scale,
                    )[0]

            else:
                #print('audio_module', audio_module)
                if audio_module is not None:
                    #print('audio_module', audio_module)
                    hidden_states = (
                        audio_module(
                            hidden_states,
                            encoder_hidden_states=audio_embedding,
                            attention_mask=attention_mask,
                            face_mask_list=face_mask_list,
                            lip_mask_list=lip_mask_list,
                            activate_score_list=activate_score_list,
                        )
                    ).sample
        return hidden_states
    
if __name__ == '__main__':
    model = MultiHumanAudioBlock(in_channels=320, out_channels=320, feature_shape=(48, 64))
    
    audio_embeding = torch.zeros((1, 25, 12, 768))
    activate_score = [torch.zeros((1, 25, 1, 1)), torch.zeros((1, 25, 1, 1))]
    hidden_states = torch.zeros((1, 320, 25, 48, 64))
    
    face_mask = [[torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)], [torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)]]
    lip_mask = [[torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)], [torch.zeros((1, 1, 25, 48, 64)).reshape(1, 25, 48*64)]]
    hidden_states = model.forward(hidden_states, 
                                  lip_mask_list=lip_mask, 
                                  face_mask_list=face_mask,
                                  audio_embedding=audio_embeding, 
                                  activate_score_list=activate_score)
    print(hidden_states.shape)