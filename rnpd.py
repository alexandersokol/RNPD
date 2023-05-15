import fileinput
import os
import shutil
import sys
import time
from subprocess import call

import requests
from IPython.display import clear_output

INSTALL_PACKAGES = [
    "gdown",
    "numpy==1.23.5",
    "accelerate==0.12.0",
    "gradio_client"
]
GRADIO_PACKAGE = 'gradio==3.28.1'

WORKSPACE_DIR = "/workspace"

REPOSITORY_NAME = 'rnpd-stable-diffusion'
BACKUP_FILENAME = 'sd_backup_rnpd.tar.zst'

SD_DIR = os.path.join(WORKSPACE_DIR, 'sd', )
STABLE_DIFFUSION_DIR = os.path.join(SD_DIR, 'stablediffusion')
WEBUI_DIR = os.path.join(SD_DIR, 'stable-diffusion-webui')

MODELS_DIR = os.path.join(WORKSPACE_DIR, 'models')
CKPT_DIR = os.path.join(MODELS_DIR, 'stable-diffusion')
VAE_DIR = os.path.join(MODELS_DIR, 'vae')
LORA_DIR = os.path.join(MODELS_DIR, 'lora')
HYPERNETWORKS_DIR = os.path.join(MODELS_DIR, 'hypernetworks')
LYCORIS_DIR = os.path.join(MODELS_DIR, 'lycoris')
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, 'embeddings')


def pip_install(arguments: str):
    call(f'pip install --root-user-action=ignore --disable-pip-version-check --no-deps -qq {arguments}',
         shell=True,
         stdout=open('/dev/null', 'w'))


def wget(url: str):
    call(f'wget -q {url}',
         shell=True,
         stdout=open('/dev/null', 'w'))


def unzst(file: str):
    call(f'tar -C / --zstd -xf {file}',
         shell=True,
         stdout=open('/dev/null', 'w'))


def sed(what: str, where: str):
    call(f"sed -i '{what}' {where}", shell=True)


def git_clone(arguments: str):
    call(f"git clone {arguments}",
         shell=True,
         stdout=open('/dev/null', 'w'))


def install_dependencies(force_reinstall: bool):
    if not force_reinstall and os.path.exists('/usr/local/lib/python3.10/dist-packages/safetensors'):
        print('[1;32mDependencies already installed.')
    else:
        print('[1;33mInstalling the dependencies...')
        if not os.path.exists(WORKSPACE_DIR):
            os.mkdir(WORKSPACE_DIR)
        os.chdir(WORKSPACE_DIR)

        packages = ''
        for package in INSTALL_PACKAGES:
            packages += package
            packages += ' '

        if packages:
            packages += '--force-reinstall'
            pip_install(packages)

        if os.path.exists('deps'):
            shutil.rmtree('deps')

        if os.path.exists('diffusers'):
            shutil.rmtree('diffusers')

        if not os.path.exists('cache'):
            os.mkdir('cache')

        os.mkdir('deps')
        os.chdir('deps')
        wget('https://huggingface.co/TheLastBen/dependencies/resolve/main/rnpddeps-t2.tar.zst')
        unzst('rnpddeps-t2.tar.zst')

        pip_install(GRADIO_PACKAGE)

        sed('s@~/.cache@/workspace/cache@', '/usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py')
        os.chdir(WORKSPACE_DIR)

        git_clone('--depth 1 -q --x main https://github.com/TheLastBen/diffusers')

        shutil.rmtree('deps')

        clear_output()
        print('[1;32mDone.')

    os.chdir(WORKSPACE_DIR)
    os.environ['PYTHONWARNINGS'] = 'ignore'


def download_webui():
    if not os.path.isdir(WEBUI_DIR):

        if not os.path.isdir(SD_DIR):
            os.mkdir(SD_DIR)

        os.chdir(SD_DIR)
        git_clone('-q --depth 1 --branch master https://github.com/AUTOMATIC1111/stable-diffusion-webui')


def download_stable_diffusion():
    os.chdir(WORKSPACE_DIR)

    if not os.path.isdir(STABLE_DIFFUSION_DIR):
        wget('-O sd_rep.tar.zst https://huggingface.co/TheLastBen/dependencies/resolve/main/sd_rep.tar.zst')
        call('tar --zstd -xf sd_rep.tar.zst', shell=True)
        call('rm sd_rep.tar.zst', shell=True)


def install_webui(huggingface_token):
    from huggingface_hub import HfApi

    os.chdir(WORKSPACE_DIR)

    if huggingface_token:
        username = HfApi().whoami(huggingface_token)["name"]
        backup = f"https://USER:{huggingface_token}@huggingface.co/datasets/{username}/{REPOSITORY_NAME}/resolve" \
                 f"/main/{BACKUP_FILENAME}"
        response = requests.head(backup)

        if response.status_code == 302:
            print('[1;33mRestoring the SD backup folder...')
            os.chdir(WORKSPACE_DIR)
            webui_path = os.path.join(WORKSPACE_DIR, BACKUP_FILENAME)
            open(webui_path, 'wb').write(requests.get(backup).content)
            unzst(BACKUP_FILENAME)
            os.remove(BACKUP_FILENAME)
        else:
            print('[1;33mBackup not found, using a fresh/existing repo...')
            time.sleep(2)
            download_stable_diffusion()
            download_webui()
    else:
        print('[1;33mInstalling/Updating the repo...')
        os.chdir(WORKSPACE_DIR)
        download_stable_diffusion()
        download_webui()

    os.chdir(WEBUI_DIR)
    call('git reset --hard', shell=True)
    # call('git pull', shell=True)
    os.chdir(WORKSPACE_DIR)
    clear_output()
    print('[1;32mDone.')


def get_start_params(download_sd_model: bool) -> str:
    params = '--disable-console-progressbars      '
    params += ' --no-half-vae'
    params += ' --disable-safe-unpickle'
    params += ' --api'
    params += ' --opt-sdp-attention'
    params += ' --enable-insecure-extension-access'
    params += ' --skip-version-check'
    params += ' --listen'
    params += ' --port 3000'
    params += ' --theme dark'
    params += f' --ckpt-dir {CKPT_DIR}'
    params += f' --vae-dir {VAE_DIR}'
    params += f' --lora-dir {LORA_DIR}'
    params += f' --hypernetwork-dir {HYPERNETWORKS_DIR}'
    params += f' --embeddings-dir {EMBEDDINGS_DIR}'

    lycoris_ext_path = os.path.join(WEBUI_DIR, 'extensions', 'a1111-sd-webui-lycoris')
    if os.path.exists(lycoris_ext_path):
        params += f' --lyco-dir {LYCORIS_DIR}'

    if not os.path.isdir(CKPT_DIR):
        os.makedirs(CKPT_DIR)

    if not os.path.isdir(VAE_DIR):
        os.makedirs(VAE_DIR)

    if not os.path.isdir(LORA_DIR):
        os.makedirs(LORA_DIR)

    if not os.path.isdir(HYPERNETWORKS_DIR):
        os.makedirs(HYPERNETWORKS_DIR)

    if not os.path.isdir(LYCORIS_DIR):
        os.makedirs(LYCORIS_DIR)

    if not os.path.isdir(HYPERNETWORKS_DIR):
        os.makedirs(HYPERNETWORKS_DIR)

    if not download_sd_model:
        params += ' --no-download-sd-model'

    return params


def prepare_initial_model() -> bool:
    has_prepared_models = False

    extensions = ['.safetensors', '.ckpt', '.pt', '.bin']
    for root, dirs, files in os.walk(CKPT_DIR):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                has_prepared_models = True

    if not has_prepared_models and os.path.exists('/workspace/auto-models/SDv1-5.ckpt'):
        shutil.move('/workspace/auto-models/SDv1-5.ckpt', os.path.join(CKPT_DIR, "SDv1-5.ckpt"))
        has_prepared_models = True

    if os.path.isdir('/workspace/auto-models'):
        shutil.rmtree('/workspace/auto-models')

    return has_prepared_models


def webui_config():
    import gradio

    gradio.close_all()

    call(
        'wget -q -O /usr/local/lib/python3.10/dist-packages/gradio/blocks.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/blocks.py',
        shell=True)
    os.chdir(os.path.join(WEBUI_DIR, 'modules'))

    call(
        'wget -q -O paths.py https://raw.githubusercontent.com/TheLastBen/fast-stable-diffusion/main/AUTOMATIC1111_files/paths.py',
        shell=True)
    call(
        "sed -i 's@/content/gdrive/MyDrive/sd/stablediffusion@/workspace/sd/stablediffusion@' /workspace/sd/stable-diffusion-webui/modules/paths.py",
        shell=True)
    call(
        "sed -i 's@\"quicksettings\": OptionInfo(.*@\"quicksettings\": OptionInfo(\"sd_model_checkpoint,  sd_vae, CLIP_stop_at_last_layers, inpainting_mask_weight, initial_noise_multiplier\", \"Quicksettings list\"),@' /workspace/sd/stable-diffusion-webui/modules/shared.py",
        shell=True)
    call("sed -i 's@print(\"No module.*@@' /workspace/sd/stablediffusion/ldm/modules/diffusionmodules/model.py",
         shell=True)

    os.chdir(WEBUI_DIR)

    clear_output()

    podid = os.environ.get('RUNPOD_POD_ID')
    localurl = f"{podid}-3000.proxy.runpod.net"

    for line in fileinput.input('/usr/local/lib/python3.10/dist-packages/gradio/blocks.py', inplace=True):
        if line.strip().startswith('self.server_name ='):
            line = f'            self.server_name = "{localurl}"\n'
        if line.strip().startswith('self.protocol = "https"'):
            line = '            self.protocol = "https"\n'
        if line.strip().startswith('if self.local_url.startswith("https") or self.is_colab'):
            line = ''
        if line.strip().startswith('else "http"'):
            line = ''
        sys.stdout.write(line)

    is_model_prepared = prepare_initial_model()

    return get_start_params(download_sd_model=not is_model_prepared)


def backup(huggingface_token):
    from slugify import slugify
    from huggingface_hub import HfApi, CommitOperationAdd, create_repo

    if huggingface_token == "":
        print('[1;31mA huggingface write token is required')

    else:
        os.chdir('/workspace')

        if os.path.exists('sd'):

            call(
                f"tar "
                f"--exclude='stable-diffusion-webui/models/*/*' "
                f"--exclude='sd-webui-controlnet/models/*'"
                f"--exclude='sd-webui-controlnet/outputs/*'"
                f"--exclude='sd-webui-controlnet/outputs/*/*'"
                f"--exclude='sd-webui-controlnet/outputs/*/*/*'"
                f"--exclude='sd-webui-controlnet/outputs/*/*/*/*'"
                f" --zstd -cf {BACKUP_FILENAME}",
                shell=True)
            api = HfApi()
            username = api.whoami(token=huggingface_token)["name"]

            repo_id = f"{username}/{slugify(REPOSITORY_NAME)}"

            print("[1;32mBacking up...")

            operations = [CommitOperationAdd(path_in_repo=BACKUP_FILENAME,
                                             path_or_fileobj=os.path.join(WORKSPACE_DIR, BACKUP_FILENAME))]

            create_repo(repo_id, private=True, token=huggingface_token, exist_ok=True, repo_type="dataset")

            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message="SD folder Backup",
                token=huggingface_token
            )

            call(f'rm {BACKUP_FILENAME}', shell=True)
            clear_output()

            print("[1;32mDone")

        else:
            print('[1;33mNothing to backup')
