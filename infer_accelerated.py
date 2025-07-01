import sys

import os
import torch
import argparse
import copy

from glob import glob
from decord import VideoReader, cpu
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
import gc
import csv
from torchvision.io import write_video
import numpy as np
import cv2

from peft import LoraConfig
from omegaconf import OmegaConf
from torch.nn.functional import interpolate


from src.models.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from src.pipelines.pipeline_vsr_tile_singleforward import FrameInterpolationWithNoiseInjectionPipelineAR
from src.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from src.attn_ctrl.attention_control import AttentionStore, register_temporal_self_attention_control, register_temporal_self_attention_flip_control
from src.models.unet import ControlNetSDVModel
from src.models.controlnet import UNetSpatioTemporalConditionControlNetModel
from src.utils.wavelet_color_fix import wavelet_reconstruction, adaptive_instance_normalization


from SUPIR.util import create_SUPIR_model, PIL2Tensor, Tensor2PIL, convert_dtype
from llava.llava_agent import LLavaAgent
from SUPIR.CKPT_PTH import LLAVA_MODEL_PATH



def read_video(video_path, height=576, width=1024, length=14, star_frame=0):
    video_reader = VideoReader(video_path, ctx=cpu(0))
    frame_num = len(video_reader)
    if length==-1:
        length = frame_num
    star_frame = min(star_frame, frame_num-length)
    frame_indices = [i+star_frame for i in range(length)]
    frames = video_reader.get_batch(frame_indices)
    frames = (torch.tensor(frames.asnumpy()).permute(0, 3, 1, 2).float())/255.0
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.ToTensor() 
    ])
    cropped_resized_tensors = torch.stack([transform(t) for t in frames])
    return cropped_resized_tensors

def unsharp_mask(tensor, sigma=1.0, strength=1.5):
    tensor_np = tensor.permute(0, 2, 3, 1).cpu().numpy() 
    sharpened_frames = []
    for frame in tensor_np:
        blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
        sharpened = cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)
        sharpened_frames.append(sharpened)
    sharpened_tensor = torch.tensor(np.array(sharpened_frames)).permute(0, 3, 1, 2)  # 转换为 (B, C, H, W)
    return sharpened_tensor

def get_hqReference(args, img_path, model, llava_agent, use_llava, SUPIR_device):
    LQ_ips = img_path
    LQ_img, h0, w0 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size)
    LQ_img = LQ_img.unsqueeze(0).to(model.device)[:, :3, :, :]

    # step 1: Pre-denoise for LLaVA, resize to 512
    LQ_img_512, h1, w1 = PIL2Tensor(LQ_ips, upsacle=args.upscale, min_size=args.min_size, fix_resize=512)
    LQ_img_512 = LQ_img_512.unsqueeze(0).to(model.device)[:, :3, :, :]
    clean_imgs = model.batchify_denoise(LQ_img_512)
    clean_PIL_img = Tensor2PIL(clean_imgs[0], h1, w1)

    # step 2: LLaVA
    if use_llava:
        captions = llava_agent.gen_image_caption([clean_PIL_img])
    else:
        captions = ['']
    print(captions)

    
    LQ_img = LQ_img.to(device=model.device)
    print("SUPIR_device:%s, LQ_img=%s" %(SUPIR_device, LQ_img.device))

    sample = model.batchify_sample(LQ_img, captions, num_steps=args.edm_steps, restoration_scale=args.s_stage1, s_churn=args.s_churn,
                                    s_noise=args.s_noise, cfg_scale=args.s_cfg, control_scale=args.s_stage2, seed=args.seed,
                                    num_samples=args.num_samples, p_p=args.a_prompt, n_p=args.n_prompt, color_fix_type=args.color_fix_type,
                                    use_linear_CFG=args.linear_CFG, use_linear_control_scale=args.linear_s_stage2,
                                    cfg_scale_start=args.spt_linear_CFG, control_scale_start=args.spt_linear_s_stage2)
    
    image = interpolate(sample, size=(h0, w0), mode='bicubic')
    image = ((image/2 + 0.5)).cpu().clip(0, 1)
    return image

def main(args):
    weight_dtype = torch.float16

    print("unet path: " + str(args.unet_path))
    print("controlnet path: " + str(args.controlnet_path))

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    ref_unet = ControlNetSDVModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", low_cpu_mem_usage=True, variant='fp16'
    ).to(args.device, dtype=weight_dtype)
    unet = ControlNetSDVModel.from_pretrained(args.unet_path, subfolder="unet", low_cpu_mem_usage=True).to(args.device, dtype=weight_dtype)
    
    controlnet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(args.controlnet_path).to(args.device, dtype=weight_dtype)

    controller_ref= AttentionStore()
    register_temporal_self_attention_control(ref_unet, controller_ref)

    controller = AttentionStore()
    register_temporal_self_attention_flip_control(unet, controller, controller_ref)
    
    vae = AutoencoderKLTemporalDecoder.from_pretrained(args.vae_path if args.vae_path is not None else args.pretrained_model_name_or_path, subfolder="vae").to(args.device, dtype=weight_dtype)
    
    print("vae path:" + str(args.vae_path))
    print("save path: " + str(args.out_path))
    print("forward_scale:%f, backward_scale:%f" %(args.forward_scale, args.backwrad_scale))
    print("min_cfg:%f, max_cfg:%f" %(args.min_cfg, args.max_cfg))
    print("sdedit_noise_rate:" + str(args.sdedit_noise_rate))
    print("use_usm:" + str(args.use_usm))
    print("sr_type:" + str(args.sr_type))


    if args.lora_path is not None:
        print("lora path:" + str(args.lora_path))
        vae_lora_config = LoraConfig(
            r=128,
            lora_alpha=128, # 0 scaling = lora_alpha/r
            init_lora_weights="gaussian",
            target_modules=["conv", "to_k", "to_q", "to_v", "to_out.0"],
        )
        vae.decoder.add_adapter(vae_lora_config)
        model_dict = vae.decoder.state_dict()
        pretrained_dict = torch.load(args.lora_path)
        modified_dict = {
            key.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default'): value
            for key, value in pretrained_dict.items()
        }
        pretrained_dict = {k: v for k, v in modified_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        vae.decoder.load_state_dict(model_dict)

    pipe = FrameInterpolationWithNoiseInjectionPipelineAR.from_pretrained(
        args.pretrained_model_name_or_path, 
        scheduler=noise_scheduler,
        unet=ref_unet,
        controlnet = controlnet,
        vae = vae,
        variant="fp16",
        torch_dtype=torch.float16, 
    )
    
    pipe = pipe.to(device=args.device, dtype=weight_dtype)

    stride = [args.stride_height, args.stride_width]

    

    if args.sr_type=='supir':
        sr_model = create_SUPIR_model('SUPIR/SUPIR_v0.yaml', SUPIR_sign='Q')
        if args.loading_half_params:
            sr_model = sr_model.half()
        if args.use_tile_vae:
            sr_model.init_tile_vae(encoder_tile_size=args.encoder_tile_size, decoder_tile_size=args.decoder_tile_size)
        sr_model.ae_dtype = convert_dtype(args.ae_dtype)
        sr_model.model.dtype = convert_dtype(args.diff_dtype)
        sr_model = sr_model.to(args.device)
        # load LLaVA
        use_llava = not args.no_llava
        if use_llava:
            llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=args.device, load_8bit=args.load_8bit_llava, load_4bit=False)
        else:
            llava_agent = None
    elif args.sr_type == "resshift":
        from resshift.sampler import ResShiftSampler
        configs_sr = OmegaConf.load('resshift/configs/realsr_swinunet_realesrgan256_journal.yaml')
        configs_sr.model.ckpt_path = 'checkpoints/DAM-VSR/resshift_realsrx4_s4_v3.pth'
        configs_sr.diffusion.params.sf = 4
        configs_sr.autoencoder.ckpt_path = 'checkpoints/DAM-VSR/autoencoder_vq_f4.pth'
        chop_stride = 448
        resshift_sampler = ResShiftSampler(
            configs_sr,
            sf=4,
            chop_size=512,
            chop_stride=chop_stride,
            chop_bs=1,
            use_amp=True,
            seed=12345,
            padding_offset=64,
            )
    elif args.sr_type=='invsr':
        from invsr.sampler_invsr import InvSamplerSR
        configs = OmegaConf.load('invsr/configs/sample-sd-turbo.yaml')
        configs.timesteps = [200,]
        configs.sd_pipe.params.cache_dir = 'checkpoints'
        configs.model_start.ckpt_path = 'checkpoints/noise_predictor_sd_turbo_v5.pth'
        configs.bs = 1
        configs.tiled_vae = True
        configs.color_fix = ''
        configs.basesr.chopping.pch_size = 128
        configs.basesr.chopping.extra_bs = 8
        invsr_sampler = InvSamplerSR(configs)
            

    _, ext = os.path.splitext(args.validation_data_dir)
    if ext in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg'}:
        validation_video_paths = [args.validation_data_dir]
    elif os.path.isfile(args.validation_data_dir):
        validation_video_paths = []
        with open(args.validation_data_dir, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                validation_video_paths.append(row['filepath'])
    else:
        validation_video_paths = sorted(glob(os.path.join(args.validation_data_dir, '**.mp4')))


    # run inference
    generator = torch.Generator(device=args.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    

    for video_path in validation_video_paths:
        print(f'start process:{video_path}')
        save_name = video_path.split("/")[-1].replace(".mp4", "")
        frames_all_path = os.path.join(args.out_path, f"{save_name}.mp4")
        if os.path.exists(frames_all_path):
            print(f'continue:{video_path}')
            continue

        video_tensor = read_video(video_path, length=-1)
        # video_tensor = video_tensor[:, :, :144, :256]
        # video_tensor = video_tensor[:, :, :320, :512]

        
        pad_w1, pad_w2, pad_h1, pad_h2 = 0, 0, 0, 0
        if video_tensor.shape[2]<576/args.video_upscale:
            pad_h1 = (576//args.video_upscale - video_tensor.shape[2])//2
            pad_h2 = 576//args.video_upscale - video_tensor.shape[2] - pad_h1
        if video_tensor.shape[3]<1024/args.video_upscale:
            pad_w1 = (1024//args.video_upscale - video_tensor.shape[3])//2
            pad_w2= 1024//args.video_upscale - video_tensor.shape[3] - pad_w1
        video_tensor = F.pad(video_tensor, (pad_w1, pad_w2, pad_h1, pad_h2), 'constant', 0)
        

        results = None
        cond_results = []
        generated = None
        HQ_Refer_dict = {}

        for i in range(0, video_tensor.shape[0], args.frames_chunk - args.overlap):
            if i + args.overlap >= video_tensor.shape[0]:
                break
            l = generated.shape if generated is not None else 0
            if generated is not None and i + args.frames_chunk > video_tensor.shape[0]:
                cur_i = max(video_tensor.shape[0] - args.frames_chunk, 0) # TODO
                cur_overlap = i - cur_i + args.overlap
            else:
                cur_i = i
                cur_overlap = args.overlap
            ref_frames_i = video_tensor[
                cur_i : cur_i + args.frames_chunk
            ].clone()
            if generated is not None:
                try:
                    ref_frames_i[:cur_overlap] = generated[-cur_overlap:]
                except Exception as e:
                    print(e)
                    print(
                        f"i: {i}, l: {l}, cur_i: {cur_i}, cur_overlap: {cur_overlap}, input_frames_i: {ref_frames_i.shape}, generated: {generated.shape}"
                    )
            star_index = cur_i
            end_index = cur_i + args.frames_chunk- 1
            print("processing frame: %d -- %d" % (star_index, (end_index)))
            input_depth = F.interpolate(ref_frames_i, (ref_frames_i.shape[2]*args.video_upscale, ref_frames_i.shape[3]*args.video_upscale), mode="bilinear", align_corners=True)

            
            if star_index not in HQ_Refer_dict:
                if args.sr_type=='supir':
                    ref_image = get_hqReference(args, transforms.ToPILImage()(torch.nn.functional.interpolate(video_tensor[star_index].unsqueeze(0), size=[video_tensor.shape[2]*args.video_upscale, video_tensor.shape[3]*args.video_upscale], mode="bicubic")[0].clamp(0,1)).convert('RGB'), sr_model, llava_agent, use_llava, args.device)
                elif args.sr_type=='resshift':
                    input_img = video_tensor[star_index].unsqueeze(0).to(device=args.device)
                    ref_image = resshift_sampler.inference_prepare(
                        input_img
                        )
                elif args.sr_type=='invsr':
                    input_img = video_tensor[star_index].unsqueeze(0).to(device=args.device)
                    ref_image = invsr_sampler.inference_prepare(
                        input_img
                        )
                    ref_image = torch.tensor(ref_image).permute(0, 3, 1, 2)
                HQ_Refer_dict[star_index] = ref_image.cpu()
            else:
                ref_image = results[star_index].unsqueeze(0)

            
            generated = pipe(unet=unet, image1=ref_image, image2=None, 
                             controlnet_cond=input_depth.unsqueeze(0).to(dtype=weight_dtype, device=controlnet.device),
                             num_inference_steps=args.num_inference_steps, 
                             generator=generator,
                             weighted_average=args.weighted_average,
                             noise_injection_steps=args.noise_injection_steps,
                             noise_injection_ratio= args.noise_injection_ratio,
                             output_type='pt',
                             height=input_depth.shape[2],
                             width=input_depth.shape[3],
                             decode_chunk_size=2,
                             stride=stride,
                             noise_level = args.noise_level,
                             low_res_video = ref_frames_i.unsqueeze(0),
                             min_guidance_scale = args.min_cfg,
                             max_guidance_scale = args.max_cfg,
                             conditioning_scale_forward = args.forward_scale,
                             conditioning_scale_backwrad = args.backwrad_scale,
                             sdedit_noise_rate = args.sdedit_noise_rate,
                             ).frames[0]


            if args.color_fix_type == 'Wavelet':
                generated = wavelet_reconstruction(generated.cpu().float(), input_depth.float()).clamp(0, 1)
            elif args.color_fix_type == 'AdaIn':
                generated = adaptive_instance_normalization(generated.cpu().float(), input_depth.float()).clamp(0, 1)
            else:
                generated = frames_output


            if results is None:
                results = generated
            else:
                blending_weight = torch.tensor(np.linspace(1, 0, cur_overlap)).view(cur_overlap, 1, 1, 1).to(dtype=generated.dtype, device=generated.device)
                generated[:cur_overlap] = blending_weight*results[-cur_overlap:] + (1-blending_weight)*generated[:cur_overlap]
                results = results[:-cur_overlap]
                results = torch.cat((results, generated), dim=0)


        frames_output = results.cpu()
        frames_output = frames_output[:, :, pad_h1*args.video_upscale:frames_output.shape[2]-pad_h2*args.video_upscale, pad_w1*args.video_upscale:frames_output.shape[3]-pad_w2*args.video_upscale]   # pad_w1, pad_w2, pad_h1, pad_h2
        res_processing = frames_output

        if args.use_usm:
            res_processing = unsharp_mask(res_processing.float()).clamp(0, 1)
        frames_all = res_processing
        
        os.makedirs(os.path.dirname(frames_all_path), exist_ok=True)
        frames_all = (frames_all * 255).permute(0, 2, 3, 1).to(dtype=torch.uint8).cpu()
        write_video(frames_all_path, frames_all, fps=7, video_codec="h264", options={"crf": "16"})


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="checkpoints/stable-video-diffusion-img2vid")
    parser.add_argument('--out_path', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=30)
    parser.add_argument('--weighted_average', action='store_true')
    parser.add_argument('--noise_injection_steps', type=int, default=1)
    parser.add_argument('--noise_injection_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--validation_data_dir', type=str)
    parser.add_argument('--condition_type', type=str)
    parser.add_argument('--frames_chunk', type=int, default=14)
    parser.add_argument('--overlap', type=int, default=1)
    parser.add_argument('--sr_type', type=str, default='supir', help="supir, invsr, resshift")
    parser.add_argument("--controlnet_path", type=str, default='checkpoints/DAM-VSR/controlnet')
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--unet_path", type=str, default='checkpoints/DAM-VSR/unet')
    parser.add_argument("--video_upscale", type=int, default=4)
    parser.add_argument("--stride_height", type=int, default=432)
    parser.add_argument("--stride_width", type=int, default=768)
    parser.add_argument("--noise_level", type=float, default=0)
    parser.add_argument("--forward_scale", type=float, default=1.0)
    parser.add_argument("--backwrad_scale", type=float, default=1.0)
    parser.add_argument("--min_cfg", type=float, default=1.00)
    parser.add_argument("--max_cfg", type=float, default=1.00)
    parser.add_argument("--sdedit_noise_rate", type=float, default=0.6)
    parser.add_argument('--use_usm', action='store_true')



    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--SUPIR_sign", type=str, default='Q', choices=['F', 'Q'])
    parser.add_argument("--min_size", type=int, default=1024)
    parser.add_argument("--edm_steps", type=int, default=50)
    parser.add_argument("--s_stage1", type=int, default=-1)
    parser.add_argument("--s_churn", type=int, default=5)
    parser.add_argument("--s_noise", type=float, default=1.003)
    parser.add_argument("--s_cfg", type=float, default=7.5)
    parser.add_argument("--s_stage2", type=float, default=1.)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--a_prompt", type=str,
                        default='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                                'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                                'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                                'hyper sharpness, perfect without deformations.')
    parser.add_argument("--n_prompt", type=str,
                        default='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                                'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                                'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                                'deformed, lowres, over-smooth')
    parser.add_argument("--color_fix_type", type=str, default='Wavelet', choices=["None", "AdaIn", "Wavelet"])
    parser.add_argument("--linear_CFG", action='store_true', default=True)
    parser.add_argument("--linear_s_stage2", action='store_true', default=False)
    parser.add_argument("--spt_linear_CFG", type=float, default=4.0)
    parser.add_argument("--spt_linear_s_stage2", type=float, default=0.)
    parser.add_argument("--ae_dtype", type=str, default="bf16", choices=['fp32', 'bf16'])
    parser.add_argument("--diff_dtype", type=str, default="fp16", choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument("--no_llava", action='store_true', default=True)
    parser.add_argument("--loading_half_params", action='store_true', default=True)
    parser.add_argument("--use_tile_vae", action='store_true', default=True)
    parser.add_argument("--encoder_tile_size", type=int, default=512)
    parser.add_argument("--decoder_tile_size", type=int, default=64)
    parser.add_argument("--load_8bit_llava", action='store_true', default=True)
    args = parser.parse_args()

    main(args)
