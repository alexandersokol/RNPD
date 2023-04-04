# Hello there
import torch
import os
import time
import requests
from IPython.display import clear_output
from subprocess import call
import ipywidgets as widgets


def done():
    done = widgets.Button(
        description='Done!',
        disabled=True,
        button_style='success',
        tooltip='',
        icon='check'
    )
    clear_output()
    display(done)


def bulk_installation(force_reinstall, huggingface_token):
    dependency_installation(force_reinstall)
    download_notebooks()
    prepare_repository(huggingface_token)


def dependency_installation(force_reinstall):
    is_torch_v2 = "2.0.0" in torch.__version__

    if force_reinstall or not os.path.exists('/usr/local/lib/python3.10/dist-packages/safetensors'):
        print('\033[34mInstalling dependencies\033[0m')

        call('pip install --root-user-action=ignore --disable-pip-version-check --no-deps -qq gdown numpy==1.23.5 accelerate==0.12.0 --force-reinstall',
             shell=True, stdout=open('/dev/null', 'w'))

        if os.path.exists('deps'):
            call("rm -r deps", shell=True)
        if os.path.exists('diffusers'):
            call("rm -r diffusers", shell=True)

        call('mkdir deps', shell=True)

        if not os.path.exists('cache'):
            call('mkdir cache', shell=True)

        os.chdir('deps')

        if is_torch_v2:
            call('wget -q https://huggingface.co/TheLastBen/dependencies/resolve/main/rnpddeps-t2.tar.zst',
                 shell=True, stdout=open('/dev/null', 'w'))
            call('tar -C / --zstd -xf rnpddeps-t2.tar.zst',
                 shell=True, stdout=open('/dev/null', 'w'))

        else:
            call('wget -q -i https://github.com/TheLastBen/fast-stable-diffusion/raw/main/Dependencies/rnpddeps.txt',
                 shell=True, stdout=open('/dev/null', 'w'))
            call('dpkg -i *.deb', shell=True,
                 stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
            call('tar -C / --zstd -xf rnpddeps.tar.zst',
                 shell=True, stdout=open('/dev/null', 'w'))
            call('apt-get install libfontconfig1 libgles2-mesa-dev -q=2 --no-install-recommends',
                 shell=True, stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
            call("pip install --root-user-action=ignore -qq gradio==3.23",
                 shell=True, stdout=open('/dev/null', 'w'))

        call("sed -i 's@~/.cache@/workspace/cache@' /usr/local/lib/python3.10/dist-packages/transformers/utils/hub.py", shell=True)
        os.chdir('/workspace')
        call("git clone --depth 1 -q --branch main https://github.com/TheLastBen/diffusers",
             shell=True, stdout=open('/dev/null', 'w'))
        call("rm -r deps", shell=True)

    os.environ['PYTHONWARNINGS'] = 'ignore'

    done()


def download_notebooks():
    print('\033[34mDownloading notebooks\033[0m')

    os.chdir('/workspace')

    if not os.path.exists('notebooks'):
        call('mkdir notebooks', shell=True)

    os.chdir('/workspace/notebooks')
    call('wget -N -q -i https://huggingface.co/datasets/TheLastBen/RNPD/raw/main/Notebooks.txt', shell=True)
    call('rm Notebooks.txt', shell=True)
    os.chdir('/workspace')

    done()


def prepare_repository(huggingface_token):

    from huggingface_hub import HfApi

    os.chdir('/workspace')
    if huggingface_token != "":
        username = HfApi().whoami(huggingface_token)["name"]
        backup = f"https://USER:{huggingface_token}@huggingface.co/datasets/{username}/custom-fast-stable-diffusion/resolve/main/sd_backup_rnpd.tar.zst"
        response = requests.head(backup)
        if response.status_code == 302:
            print('[1;33mRestoring the SD folder...')
            open('/workspace/sd_backup_rnpd.tar.zst',
                 'wb').write(requests.get(backup).content)
            call('tar --zstd -xf sd_backup_rnpd.tar.zst', shell=True)
            call('rm sd_backup_rnpd.tar.zst', shell=True)
        else:
            print('[1;33mBackup not found, using a fresh/existing repo...')
            time.sleep(2)
            if not os.path.exists('/workspace/sd/stablediffusion'):
                call('wget -q -O sd_rep.tar.zst https://huggingface.co/TheLastBen/dependencies/resolve/main/sd_rep.tar.zst', shell=True)
                call('tar --zstd -xf sd_rep.tar.zst', shell=True)
                call('rm sd_rep.tar.zst', shell=True)
            os.chdir('/workspace/sd')
            if not os.path.exists('stable-diffusion-webui'):
                call('git clone -q --depth 1 --branch master https://github.com/AUTOMATIC1111/stable-diffusion-webui', shell=True)

    else:
        print('[1;33mInstalling/Updating the repo...')
        os.chdir('/workspace')
        if not os.path.exists('/workspace/sd/stablediffusion'):
            call('wget -q -O sd_rep.tar.zst https://huggingface.co/TheLastBen/dependencies/resolve/main/sd_rep.tar.zst', shell=True)
            call('tar --zstd -xf sd_rep.tar.zst', shell=True)
            call('rm sd_rep.tar.zst', shell=True)

        os.chdir('/workspace/sd')
        if not os.path.exists('stable-diffusion-webui'):
            call('git clone -q --depth 1 --branch master https://github.com/AUTOMATIC1111/stable-diffusion-webui', shell=True)

    os.chdir('/workspace/sd/stable-diffusion-webui/')
    call('git reset --hard', shell=True)
    print('[1;32m')
    call('git pull', shell=True)
    os.chdir('/workspace')
    clear_output()
    done()