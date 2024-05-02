sudo apt update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install wget
sudo apt-get install unzip
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

pip3 install wandb --quiet
pip3 install --upgrade --force-reinstall --no-deps kaggle==1.5.8
pip3 install torch
pip3 install torchsummary
pip3 install pandas
pip3 install scikit-learn
pip3 install matplotlib
pip3 install onnx
pip3 install slugify
pip3 install tqdm


mkdir .kaggle
echo '{"username":"Your Kaggle Username","key":"Your Kaggle Key"}' > .kaggle/kaggle.json
chmod 600 .kaggle/kaggle.json
sudo ln -s ~/.local/bin/kaggle /usr/bin/kaggle
