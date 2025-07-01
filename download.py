import os
#os.environ['HF_ENDPOINT']="https://hf-mirror.com"
from huggingface_hub import hf_hub_download, snapshot_download

class DAM_download():
    def __init__(self, save_path) -> None:
        self.save_path = save_path

        self.download_model_svd()
        print("download stabilityai/stable-video-diffusion-img2vid")


        self.download_model_sdxl()
        print("download stabilityai/stable-diffusion-xl-base-1.0")
        self.download_model_clip_vit_large_patch14_336()
        print("download openai/clip-vit-large-patch14-336")
        self.download_model_llava()
        print("download liuhaotian/llava-v1.5-13b")
        self.download_model_clip_vit_large_patch14()
        print("download openai/clip-vit-large-patch14")
        self.download_model_CLIP_ViT_bigG_14_laion2B_39B_b160k()
        print("download laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")


        self.download_model_sd_turbo()
        print("download stabilityai/sd-turbo")
        self.download_model_noise_predictor_sd_turbo_v5()
        print("download OAOA/InvSR")
        self.download_model_damvsr()
        print("download Fucius/DAM-VSR")

  
    ##### for ours ######
    def download_model_svd(self):
        REPO_ID = 'stabilityai/stable-video-diffusion-img2vid'
        if not os.path.exists(os.path.join(self.save_path, 'stable-video-diffusion-img2vid')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'stable-video-diffusion-img2vid'), local_dir_use_symlinks=False)
    
    def download_model_damvsr(self):
        REPO_ID = 'Fucius/DAM-VSR'
        if not os.path.exists(os.path.join(self.save_path, 'DAM-VSR')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'DAM-VSR'), local_dir_use_symlinks=False)

    ##### for supir ######
    def download_model_sdxl(self):
        REPO_ID = 'stabilityai/stable-diffusion-xl-base-1.0'
        if not os.path.exists(os.path.join(self.save_path, 'stable-diffusion-xl-base-1.0')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'stable-diffusion-xl-base-1.0'), local_dir_use_symlinks=False)

    def download_model_clip_vit_large_patch14_336(self):
        REPO_ID = 'openai/clip-vit-large-patch14-336'
        if not os.path.exists(os.path.join(self.save_path, 'clip-vit-large-patch14-336')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'clip-vit-large-patch14-336'), local_dir_use_symlinks=False)

    def download_model_llava(self):
        REPO_ID = 'liuhaotian/llava-v1.5-13b'
        if not os.path.exists(os.path.join(self.save_path, 'llava-v1.5-13b')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'llava-v1.5-13b'), local_dir_use_symlinks=False)

    def download_model_clip_vit_large_patch14(self):
        REPO_ID = 'openai/clip-vit-large-patch14'
        if not os.path.exists(os.path.join(self.save_path, 'clip-vit-large-patch14')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'clip-vit-large-patch14'), local_dir_use_symlinks=False)


    def download_model_CLIP_ViT_bigG_14_laion2B_39B_b160k(self):
        REPO_ID = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        if not os.path.exists(os.path.join(self.save_path, 'CLIP-ViT-bigG-14-laion2B-39B-b160k')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'CLIP-ViT-bigG-14-laion2B-39B-b160k'), local_dir_use_symlinks=False)
    
 

    ##### for invsr ######
    def download_model_sd_turbo(self):
        REPO_ID = 'stabilityai/sd-turbo'
        if not os.path.exists(os.path.join(self.save_path, 'sd-turbo')):
            snapshot_download(repo_id=REPO_ID, local_dir=os.path.join(self.save_path, 'sd-turbo'), local_dir_use_symlinks=False)

    def download_model_noise_predictor_sd_turbo_v5(self):
        REPO_ID = 'OAOA/InvSR'
        filename_list = ['noise_predictor_sd_turbo_v5.pth']
        for filename in filename_list:
            local_file = os.path.join(self.save_path, filename)
            if not os.path.exists(local_file):
                hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=self.save_path, local_dir_use_symlinks=False)



if __name__ == '__main__':
    save_path = 'checkpoints'
    os.makedirs(save_path, exist_ok=True)
    down = DAM_download(save_path)
    print("finished download")