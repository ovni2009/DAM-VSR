# Adpated from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_stable_video_diffusion.py
import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union
import copy
import numpy as np
import PIL.Image
import torch
import random
import torch.nn.functional as F
from torchvision import transforms

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from torchvision.utils import save_image
from diffusers import AutoencoderKLTemporalDecoder,  UNetSpatioTemporalConditionModel
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import logging
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
        _append_dims,
        tensor2vid,
        _resize_with_antialiasing,
        StableVideoDiffusionPipelineOutput
)
from ..schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class FrameInterpolationWithNoiseInjectionPipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        controlnet,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.ori_unet = self.unet
        self.unet = None
        self.controlnet = controlnet
       
    def _encode_image(
        self,
        image: PipelineImageInput,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.FloatTensor:
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents

    def _get_add_time_ids(
        self,
        fps: int,
        motion_bucket_id: int,
        noise_aug_strength: float,
        dtype: torch.dtype,
        batch_size: int,
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.ori_unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.ori_unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def decode_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in
            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        if isinstance(self.guidance_scale, (int, float)):
            return self.guidance_scale > 1
        return self.guidance_scale.max() > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps
    

    
    # bidrection infer + generate inbetween. *********important******
    @torch.no_grad()
    def multidiffusion_step(self, unet, controlnet_cond, latents, t, 
                    image1_embeddings, 
                    image2_embeddings, 
                    image1_latents,
                    image2_latents,
                    added_time_ids, 
                    avg_weight,
                    conditioning_scale_forward=None, 
                    conditioning_scale_backwrad=None
    ):
        # expand the latents if we are doing classifier free guidance
        latents1 = latents
        latents2 = torch.flip(latents, (1,))
        latent_model_input1 = torch.cat([latents1] * 2) if self.do_classifier_free_guidance else latents1
        latent_model_input1 = self.scheduler.scale_model_input(latent_model_input1, t)

        latent_model_input2 = torch.cat([latents2] * 2) if self.do_classifier_free_guidance else latents2
        latent_model_input2= self.scheduler.scale_model_input(latent_model_input2, t)


        # Concatenate image_latents over channels dimention
        latent_model_input1 = torch.cat([latent_model_input1, image1_latents], dim=2)
        latent_model_input2 = torch.cat([latent_model_input2, image2_latents], dim=2)

        # predict the noise residual
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input1,
            t,
            encoder_hidden_states=image1_embeddings,
            added_time_ids=added_time_ids,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        noise_pred1 = self.ori_unet(
            latent_model_input1,
            t,
            encoder_hidden_states=image1_embeddings,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=[
                sample.to(dtype=latent_model_input1.dtype) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=latent_model_input1.dtype),
            return_dict=False,
            conditioning_scale=conditioning_scale_forward
        )[0]
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latent_model_input2,
            t,
            encoder_hidden_states=image2_embeddings,
            added_time_ids=added_time_ids,
            controlnet_cond=torch.flip(controlnet_cond, (1,)),
            return_dict=False,
        )
        noise_pred2 = unet(
            latent_model_input2,
            t,
            encoder_hidden_states=image2_embeddings,
            added_time_ids=added_time_ids,
            down_block_additional_residuals=[
                sample.to(dtype=latent_model_input1.dtype) for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample.to(dtype=latent_model_input1.dtype),
            return_dict=False,
            conditioning_scale=conditioning_scale_backwrad
        )[0]
        if self.do_classifier_free_guidance:
            noise_pred_uncond1, noise_pred_cond1 = noise_pred1.chunk(2)
            noise_pred1 = noise_pred_uncond1 + self.guidance_scale * (noise_pred_cond1 - noise_pred_uncond1)

            noise_pred_uncond2, noise_pred_cond2 = noise_pred2.chunk(2)
            noise_pred2 = noise_pred_uncond2 + self.guidance_scale * (noise_pred_cond2 - noise_pred_uncond2)

        noise_pred2 = torch.flip(noise_pred2, (1,))
        noise_pred = avg_weight*noise_pred1+ (1-avg_weight)*noise_pred2
        return noise_pred


    

    def get_views(self, height, width, window_size=[128, 128], stride=[64, 64], random_jitter=False):
        # Here, we define the mappings F_i (see Eq. 7 in the MultiDiffusion paper https://arxiv.org/abs/2302.08113)
        # print(f"height:{height}, width:{width}, window_size:{window_size}, stride:{stride}")
        # if panorama's height/width < window_size, num_blocks of height/width should return 1
        num_blocks_height = int((height - window_size[0]) / stride[0] - 1e-6) + 2 if height > window_size[0] else 1
        num_blocks_width = int((width - window_size[1]) / stride[1] - 1e-6) + 2 if width > window_size[1] else 1
        total_num_blocks = int(num_blocks_height * num_blocks_width)
        # coverage_tensor = torch.zeros([height, width], dtype=int)
        views = []
        for i in range(total_num_blocks):
            h_start = int((i // num_blocks_width) * stride[0])
            h_end = h_start + window_size[0]
            w_start = int((i % num_blocks_width) * stride[1])
            w_end = w_start + window_size[1]

            if h_end > height:
                h_start = int(h_start + height - h_end)
                h_end = int(height)
            if w_end > width:
                w_start = int(w_start + width - w_end)
                w_end = int(width)
            if h_start < 0:
                h_end = int(h_end - h_start)
                h_start = 0
            if w_start < 0:
                w_end = int(w_end - w_start)
                w_start = 0

            if random_jitter:
                jitter_range_h = (window_size[0] - stride[0]) // 4
                jitter_range_w = (window_size[1] - stride[1]) // 4

                w_jitter = 0
                h_jitter = 0

                if (w_start != 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range_w, jitter_range_w)
                elif (w_start == 0) and (w_end != width):
                    w_jitter = random.randint(-jitter_range_w, 0)
                elif (w_start != 0) and (w_end == width):
                    w_jitter = random.randint(0, jitter_range_w)

                # 处理高度抖动
                if (h_start != 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range_h, jitter_range_h)
                elif (h_start == 0) and (h_end != height):
                    h_jitter = random.randint(-jitter_range_h, 0)
                elif (h_start != 0) and (h_end == height):
                    h_jitter = random.randint(0, jitter_range_h)
                h_start += (h_jitter + jitter_range_h)
                h_end += (h_jitter + jitter_range_h)
                w_start += (w_jitter + jitter_range_w)
                w_end += (w_jitter + jitter_range_w)
            
            views.append((h_start, h_end, w_start, w_end))
            # coverage_tensor[h_start:h_end, w_start: w_end] += 1
        return views, num_blocks_height, num_blocks_width

    
    
    @torch.no_grad()
    def tile_step(self, unet, sum_controlnet_cond, latents, t, sum_image1, sum_image2, added_time_ids, avg_weight, device, num_videos_per_prompt, generator, noise_aug_strength, view_batch_size=1, stride=None, current_step=None, conditioning_scale_forward=None, conditioning_scale_backwrad=None):

        current_height = latents.shape[3]*self.vae_scale_factor
        current_width = latents.shape[4]*self.vae_scale_factor
        if current_height*current_width>576*1024:
            height = 576
            width = 1024
        else:
            height = current_height
            width = current_width
        window_size = [height, width]
        # stride = [288, 512]
        
        
        views, _, _ = self.get_views(current_height//self.vae_scale_factor, current_width//self.vae_scale_factor, 
                               window_size = [window_size[0]//self.vae_scale_factor, window_size[1]//self.vae_scale_factor], 
                               stride = [stride[0]//self.vae_scale_factor, stride[1]//self.vae_scale_factor], 
                               random_jitter=True)
        views_batch = [views[i : i + view_batch_size] for i in range(0, len(views), view_batch_size)]

        jitter_range_h = (window_size[0]//self.vae_scale_factor - stride[0]//self.vae_scale_factor) // 4
        jitter_range_w = (window_size[1]//self.vae_scale_factor - stride[1]//self.vae_scale_factor) // 4
        latents_ = F.pad(latents, (jitter_range_w, jitter_range_w, jitter_range_h, jitter_range_h), 'constant', 0)
        # print(f"latents.shape={latents.shape}, latents_.shape={latents_.shape}")
        sum_controlnet_cond_ = F.pad(sum_controlnet_cond, (jitter_range_w*self.vae_scale_factor, jitter_range_w*self.vae_scale_factor, jitter_range_h*self.vae_scale_factor, jitter_range_h*self.vae_scale_factor), 'constant', 0)
        sum_image1_ = F.pad(sum_image1, (jitter_range_w*self.vae_scale_factor, jitter_range_w*self.vae_scale_factor, jitter_range_h*self.vae_scale_factor, jitter_range_h*self.vae_scale_factor), 'constant', 0)
        sum_image2_ = F.pad(sum_image2, (jitter_range_w*self.vae_scale_factor, jitter_range_w*self.vae_scale_factor, jitter_range_h*self.vae_scale_factor, jitter_range_h*self.vae_scale_factor), 'constant', 0)
        

        count_local = torch.zeros_like(latents_)
        value_local = torch.zeros_like(latents_)

        for j, batch_view in enumerate(views_batch):
            vb_size = len(batch_view)
            latents_for_view = torch.cat(
                [
                    latents_[:, :, :, h_start:h_end, w_start:w_end]
                    for h_start, h_end, w_start, w_end in batch_view
                ]
            )
            controlnet_cond = torch.cat(
                [
                    sum_controlnet_cond_[:, :, :, h_start*self.vae_scale_factor:h_end*self.vae_scale_factor, w_start*self.vae_scale_factor:w_end*self.vae_scale_factor]
                    for h_start, h_end, w_start, w_end in batch_view
                ]
            )

            image1 = torch.cat([sum_image1_[:, :, h_start*self.vae_scale_factor:h_end*self.vae_scale_factor, w_start*self.vae_scale_factor:w_end*self.vae_scale_factor] for h_start, h_end, w_start, w_end in batch_view])
            image2 = torch.cat([sum_image2_[:, :, h_start*self.vae_scale_factor:h_end*self.vae_scale_factor, w_start*self.vae_scale_factor:w_end*self.vae_scale_factor] for h_start, h_end, w_start, w_end in batch_view])
            image1 = transforms.ToPILImage()(image1[0]).convert('RGB')
            image2 = transforms.ToPILImage()(image2[0]).convert('RGB')

            

            ####################################################################################
            # 3. Encode input image
            image1_embeddings = self._encode_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
            image2_embeddings = self._encode_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)

            # 4. Encode input image using VAE
            image1 = self.image_processor.preprocess(image1, height=height, width=width).to(device)
            image2 = self.image_processor.preprocess(image2, height=height, width=width).to(device)
            noise = randn_tensor(image1.shape, generator=generator, device=image1.device, dtype=image1.dtype)
            image1 = image1 + noise_aug_strength * noise
            image2 = image2 + noise_aug_strength * noise

            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
            if needs_upcasting:
                self.vae.to(dtype=torch.float32)

            # Repeat the image latents for each frame so we can concatenate them with the noise
            # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
            image1_latent = self._encode_vae_image(image1, device, num_videos_per_prompt, self.do_classifier_free_guidance)
            image1_latent = image1_latent.to(image1_embeddings.dtype)
            image1_latents = image1_latent.unsqueeze(1).repeat(1, latents.shape[1], 1, 1, 1)

            image2_latent = self._encode_vae_image(image2, device, num_videos_per_prompt, self.do_classifier_free_guidance)
            image2_latent = image2_latent.to(image2_embeddings.dtype)
            image2_latents = image2_latent.unsqueeze(1).repeat(1, latents.shape[1], 1, 1, 1)
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            if controlnet_cond is not None:
                controlnet_cond = torch.cat([controlnet_cond] * 2) if self.do_classifier_free_guidance else controlnet_cond
            ####################################################################################
            

            # print(f"latents_for_view.shape={latents_for_view.shape}, image1_embeddings.shape={image1_embeddings.shape}, image1_latents.shape={image1_latents.shape}, controlnet_cond.shape={controlnet_cond.shape}")
            noise_pred_for_view = self.multidiffusion_step(unet, controlnet_cond, latents_for_view, t, 
                image1_embeddings, image2_embeddings, 
                image1_latents, image2_latents, added_time_ids, avg_weight, conditioning_scale_forward=conditioning_scale_forward, conditioning_scale_backwrad=conditioning_scale_backwrad
            )

            for latents_view_denoised, (h_start, h_end, w_start, w_end) in zip(
                noise_pred_for_view.chunk(vb_size), batch_view
            ):
                value_local[:, :, :, h_start:h_end, w_start:w_end] += latents_view_denoised
                count_local[:, :, :, h_start:h_end, w_start:w_end] += 1


        value_local = value_local[:, : ,:, jitter_range_h: jitter_range_h + current_height // self.vae_scale_factor, jitter_range_w: jitter_range_w + current_width // self.vae_scale_factor]
        count_local = count_local[:, : ,:, jitter_range_h: jitter_range_h + current_height // self.vae_scale_factor, jitter_range_w: jitter_range_w + current_width // self.vae_scale_factor]
        value = value_local / count_local
        return value

    @torch.no_grad()
    def __call__(
        self,
        unet,
        image1: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        image2: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        controlnet_cond = None,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        weighted_average: bool = False,
        noise_injection_steps: int = 0,
        noise_injection_ratio: float=0.0,
        return_dict: bool = True,
        stride = None,
        noise_level = 0,
        conditioning_scale_forward=1.0,
        conditioning_scale_backwrad=1.0,
        low_res_video = None,
        sdedit_noise_rate = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.ori_unet.config.sample_size * self.vae_scale_factor
        width = width or self.ori_unet.config.sample_size * self.vae_scale_factor
        
        num_frames = num_frames if num_frames is not None else self.ori_unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(image1, height, width)
        self.check_inputs(image2, height, width)

        # 2. Define call parameters
        if isinstance(image1, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image1, list):
            batch_size = len(image1)
        else:
            batch_size = image1.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            next(self.image_encoder.parameters()).dtype,
            batch_size,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.ori_unet.config.in_channels

        

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        if sdedit_noise_rate is None:
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_frames,
                num_channels_latents,
                height,
                width,
                added_time_ids.dtype,
                device,
                generator,
                latents,
            ) # torch.Size([1, 14, 4, 90, 159]) 9
            star_index = 0
        else:
            star_index = len(timesteps) - int(len(timesteps) * sdedit_noise_rate)
            low_res_video = self.image_processor.preprocess(controlnet_cond[0], height=controlnet_cond.shape[3], width=controlnet_cond.shape[4])
            low_video_latent = self.tile_encode(low_res_video.to(device), num_frames, decode_chunk_size, stride=stride).to(dtype=added_time_ids.dtype)


            noise = torch.randn_like(low_video_latent).to(dtype=low_video_latent.dtype)
            sigmas_reshaped = timesteps[star_index]
            # TODO
            # sigmas_reshaped = timesteps[len(timesteps) - int(len(timesteps) * 0.1)]
            noisy_latents  = low_video_latent + noise * sigmas_reshaped
            latents = noisy_latents  / ((sigmas_reshaped**2 + 1) ** 0.5)


        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        if weighted_average:
            self._guidance_scale = guidance_scale
            w = torch.linspace(1, 0, num_frames).unsqueeze(0).to(device, latents.dtype)
            w = w.repeat(batch_size*num_videos_per_prompt, 1)
            w = _append_dims(w, latents.ndim)
        else:
            self._guidance_scale = (guidance_scale+torch.flip(guidance_scale, (1,)))*0.5
            w = 0.5

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        self.ori_unet = self.ori_unet.to(device)
        noise_injection_step_threshold = int(num_inference_steps*noise_injection_ratio)




        

        if controlnet_cond is not None:
            controlnet_cond = self.image_processor.preprocess(controlnet_cond[0], height=height, width=width)
            controlnet_cond = controlnet_cond.unsqueeze(0)
            
            noise = randn_tensor(controlnet_cond.shape, device=controlnet_cond.device, dtype=controlnet_cond.dtype)
            noise = noise * torch.sqrt(torch.tensor(noise_level)**2)
            controlnet_cond = controlnet_cond + noise

        

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps[star_index:], start=star_index):
                
                noise_pred = self.tile_step(unet, controlnet_cond, latents, t, image1, image2, added_time_ids, w, device, num_videos_per_prompt, generator, noise_aug_strength, stride=stride, current_step=i, conditioning_scale_forward=conditioning_scale_forward, conditioning_scale_backwrad=conditioning_scale_backwrad)

                

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if i < noise_injection_step_threshold and noise_injection_steps > 0:
                    sigma_t = self.scheduler.sigmas[self.scheduler.step_index]
                    sigma_tm1 = self.scheduler.sigmas[self.scheduler.step_index+1]
                    sigma = torch.sqrt(sigma_t**2-sigma_tm1**2)
                    for j in range(noise_injection_steps):
                        print(f"i={i}, j={j}, _step_index={self.scheduler._step_index}")
                        noise = randn_tensor(latents.shape, device=latents.device, dtype=latents.dtype)
                        noise = noise * sigma
                        latents = latents + noise
                        noise_pred = self.tile_step(unet, controlnet_cond, latents, t, image1, image2, added_time_ids, w, device, num_videos_per_prompt, generator, noise_aug_strength, stride=stride, current_step=i, conditioning_scale_forward=conditioning_scale_forward, conditioning_scale_backwrad=conditioning_scale_backwrad)
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                self.scheduler._step_index += 1

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.tile_decode(latents, num_frames, decode_chunk_size, stride=stride)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

    @torch.no_grad()
    def encode_latents(self, video_latents, encode_chunk_size):
        frames = []
        for i in range(0, video_latents.shape[0], encode_chunk_size):
            frame = self.vae.encode(video_latents[i: i + encode_chunk_size]).latent_dist.mode()
            frames.append(frame)
        frames = torch.cat(frames, dim=0)
        return frames



    @torch.no_grad()
    def tile_encode(self, video_latents, num_frames, encode_chunk_size=14, spatial_n_compress=8, stride=None):
        
        current_height = video_latents.shape[2]
        current_width = video_latents.shape[3]
        if current_height*current_width>576*1024:
            height = 576
            width = 1024
        else:
            height = current_height
            width = current_width
        window_size = [height, width]
        
        
        views, num_h, num_w = self.get_views(video_latents.shape[2], video_latents.shape[3], window_size = [window_size[0], window_size[1]],  
                               stride = [stride[0], stride[1]], random_jitter=False)

        sum_tiles = torch.zeros(1, video_latents.shape[0], 4, video_latents.shape[2]//spatial_n_compress, video_latents.shape[3]//spatial_n_compress).to(dtype=video_latents.dtype, device="cuda")
        num_tiles = torch.zeros(1, video_latents.shape[0], 1, video_latents.shape[2]//spatial_n_compress, video_latents.shape[3]//spatial_n_compress).to(dtype=video_latents.dtype, device="cuda")
        
        for j, (h_start, h_end, w_start, w_end) in enumerate(views):
            latents_for_view = video_latents[:, :, h_start:h_end, w_start:w_end]
            tile_video_frames = self.encode_latents(latents_for_view, encode_chunk_size).unsqueeze(0)*self.vae.config.scaling_factor

            if j % num_w > 0 :

                tile_video_frames = blend_h(sum_tiles[:, :, :, views[j-1][0]//spatial_n_compress:views[j-1][1]//spatial_n_compress, views[j-1][2]//spatial_n_compress: views[j-1][3]//spatial_n_compress], tile_video_frames, (views[j-1][3]-w_start)//spatial_n_compress)

            if j >= num_w:

                tile_video_frames = blend_v(sum_tiles[:, :, :, views[j - num_w][0]//spatial_n_compress:views[j - num_w][1]//spatial_n_compress, views[j - num_w][2]//spatial_n_compress: views[j - num_w][3]//spatial_n_compress], tile_video_frames, (views[j - num_w][1]-h_start)//spatial_n_compress)

            

            sum_tiles[:, :, :, h_start//spatial_n_compress:h_end//spatial_n_compress, w_start//spatial_n_compress:w_end//spatial_n_compress] = tile_video_frames
        
        return sum_tiles

    @torch.no_grad()
    def tile_decode(self, video_latents, num_frames, decode_chunk_size=14, spatial_n_compress=8, stride=None):
        current_height = video_latents.shape[3]*self.vae_scale_factor
        current_width = video_latents.shape[4]*self.vae_scale_factor
        if current_height*current_width>576*1024:
            height = 576
            width = 1024
        else:
            height = current_height
            width = current_width
        window_size = [height, width]
        
        views, num_h, num_w = self.get_views(video_latents.shape[3], video_latents.shape[4], window_size = [window_size[0]//self.vae_scale_factor, window_size[1]//self.vae_scale_factor],  
                               stride = [stride[0]//self.vae_scale_factor, stride[1]//self.vae_scale_factor],
                               random_jitter=False)

        sum_tiles = torch.zeros(video_latents.shape[0], 3, video_latents.shape[1], video_latents.shape[3]*spatial_n_compress, video_latents.shape[4]*spatial_n_compress).to(dtype=torch.float16, device="cuda")
        num_tiles = torch.zeros(video_latents.shape[0], 1, video_latents.shape[1], video_latents.shape[3]*spatial_n_compress, video_latents.shape[4]*spatial_n_compress).to(dtype=torch.float16, device="cuda")

        for j, (h_start, h_end, w_start, w_end) in enumerate(views):
            latents_for_view = video_latents[:, :, :, h_start:h_end, w_start:w_end]
            tile_video_frames = self.decode_latents(latents_for_view, num_frames, decode_chunk_size)

            
            if j % num_w > 0 :
                tile_video_frames = blend_h(sum_tiles[:, :, :, views[j-1][0]*spatial_n_compress:views[j-1][1]*spatial_n_compress, views[j-1][2]*spatial_n_compress: views[j-1][3]*spatial_n_compress], tile_video_frames, (views[j-1][3]-w_start)*spatial_n_compress)

            if j >= num_w:
                tile_video_frames = blend_v(sum_tiles[:, :, :, views[j - num_w][0]*spatial_n_compress:views[j - num_w][1]*spatial_n_compress, views[j - num_w][2]*spatial_n_compress: views[j - num_w][3]*spatial_n_compress], tile_video_frames, (views[j - num_w][1]-h_start)*spatial_n_compress)

            

            sum_tiles[:, :, :, h_start*spatial_n_compress:h_end*spatial_n_compress, w_start*spatial_n_compress:w_end*spatial_n_compress] = tile_video_frames
        
        return sum_tiles


def blend_v(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, -1, 1) / overlap_size).to(
        b.device
    )
    if overlap_size == 0:
        return b
    else:
        b[:, :, :, :overlap_size, :] = (1 - weight_b) * a[:, :, :, -overlap_size:, :] + weight_b * b[:, :, :, :overlap_size, :]
        return b

def blend_h(a: torch.Tensor, b: torch.Tensor, overlap_size: int) -> torch.Tensor:
    weight_b = (torch.arange(overlap_size).view(1, 1, 1, -1) / overlap_size).to(
        b.device
    )
    if overlap_size == 0:
        return b
    else:
        b[:, :, :, :, :overlap_size] = (1 - weight_b) * a[:, :, :, :, -overlap_size:] + weight_b * b[:, :, :, :, :overlap_size]
        return b