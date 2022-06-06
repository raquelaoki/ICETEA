sudo nvidia-smi

sudo apt-get update
sudo apt install python3 python3-dev python3-venv
sudo apt-get install python3-pip
sudo apt install git
sudo apt-get install libgl1
#sudo apt-get install wget
#sudo apt-get install libxml2

python3 --version
pip3 --version

#MY project
python3 -m venv env
source env/bin/activate
#git clone --branch adding_kaggle https://github.com/raquelaoki/icetea
git clone --branch clean_up https://github.com/raquelaoki/icetea
pip install -r icetea/requirements.txt
#pip3 install --upgrade protobuf==3.19.0

#from tensorflow.python.client import device_lib
#print("DEVICES - ", device_lib.list_local_devices())

python icetea/main.py icetea/config/setup_aipw_gc.yaml icetea/config/small_test.yaml False
python icetea/main.py icetea/config/setup_aipw_gc_test.yaml icetea/config/small_test.yaml False
echo 'DONE!'

python icetea/main.py icetea/config/setup_aipw_gc_dl.yaml icetea/config/setup0.yaml False
python icetea/main.py icetea/config/setup_aipw_gc_dl.yaml icetea/config/setup1.yaml False
python icetea/main.py icetea/config/setup_aipw_gc_dl.yaml icetea/config/setup2.yaml False

#REgresson
#DONE: 7,   9,  19,  32,  39,  48,  61,  66,  74,  86, 92, 105, 111,121, 129,
#  136, 146, 161, 168, 176,
 # NOT USED  187, 192, 199, 210, 220, 228, 239, 245, 252, 262
vim icetea/config/setup_aipw_gc_dl.yaml
vim icetea/config/setup2.yaml
python icetea/main.py icetea/config/setup_aipw_gc_dl.yaml icetea/config/setup2.yaml False

