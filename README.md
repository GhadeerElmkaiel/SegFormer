# Code for running in docker
'''bash
python -m torch.distributed.launch --nproc_per_node=1 segformer/train.py --config segformer/local_configs/segformer/B3/mls.segformer.b3.512x512.sber.fisheye.generate.edges.160k.py --launcher pytorch
'''