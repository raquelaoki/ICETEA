sudo apt update
sudo apt install python3 python3-dev python3-venv

#Most updated version of pip
sudo apt-get install wget
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py


python3 --version
pip3 --version

#MY project
cd Documents/GitHub/ICETEA
python3 -m venv env
source env/bin/activate
#pip install -r requirements.txt
#pip install google-cloud-storage
#git clone https://github.com/raquelaoki/icetea

python train_models.py config/setup1.yaml
echo 'DONE!'
