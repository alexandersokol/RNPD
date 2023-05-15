import os
import shutil
import requests
import time
from subprocess import call
from IPython.display import clear_output

INSTALL_PACKAGES = [
    "gdown",
    "numpy==1.23.5",
    "accelerate==0.12.0"
]
GRADIO_PACKAGE = 'gradio==3.28.1'

WORKSPACE_DIR = "/workspace"

REPOSITORY_NAME = 'rnpd-stable-diffusion'
BACKUP_FILENAME = 'sd_backup_rnpd.tar.zst'

SD_DIR = os.path.join(WORKSPACE_DIR, 'sd', )
STABLE_DIFFUSION_DIR = os.path.join(SD_DIR, 'stablediffusion')
WEBUI_DIR = os.path.join(SD_DIR, 'stable-diffusion-webui')


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
        print('[1;33mDone.')

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
        unzst('sd_rep.tar.zst')
        os.remove('sd_rep.tar.zst')


def install_webui(huggingface_token):
    from huggingface_hub import HfApi, CommitOperationAdd, create_repo

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
