from huggingface_hub import hf_hub_download
#下载单个文件
#hf_hub_download(repo_id="lmsys/vicuna-7b-delta-v0", filename="pytorch_model-00001-of-00002.bin", local_dir="/home/nxd/NExT-GPT/vicuna")
#hf_hub_download(repo_id="lmsys/vicuna-7b-delta-v0", filename="pytorch_model-00002-of-00002.bin", local_dir="/home/nxd/NExT-GPT/vicuna")
#下载文件到本地
from huggingface_hub import snapshot_download
#snapshot_download(repo_id="lmsys/vicuna-7b-delta-v0",local_dir="/home/nxd/NExT-GPT/NExT-GPT/ckpt/pretrained_ckpt/vicuna_ckpt/7b_v0")
snapshot_download(repo_id="ChocoWu/nextgpt_7b_tiva_v0",local_dir="/home/nxd/NExT-GPT/NExT-GPT/ckpt/delta_ckpt/nextgpt/7b_tiva_v0")