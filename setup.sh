ln -s /evalcap/coco_caption /videocap/src/evalcap/coco_caption
ln -s /evalcap/cider /videocap/src/evalcap/cider
pip install fvcore ete3 transformers
pip install --upgrade azureml-core
df -h
ls -al

sed -i 's/29500/29501/' /opt/conda/lib/python3.8/site-packages/deepspeed/constants.py
export TORCH_HOME=/models
