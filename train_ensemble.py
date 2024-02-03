import subprocess


executable = "./dist_train.sh"
config = "config/yolox_m_captcha_config.py"
num_gpu = "4"
for i in range(1, 10):
    workdir = f"./work_dirs/model{i}"
    args = [executable, config, num_gpu, '--work-dir', workdir, '--seed', str(i-1)]
    subprocess.run(args)