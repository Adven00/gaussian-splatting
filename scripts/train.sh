CUDA_VISIBLE_DEVICES=3 python train.py -s data/360/bonsai  -m ./output/thesis/bonsai_obj  --lambda_sparsity_loss 0.2 --eval -i images_4 --start_checkpoint ./output/thesis/bonsai/chkpnt30000.pth --iterations 60000
CUDA_VISIBLE_DEVICES=2 python train.py -s data/360/counter  -m ./output/thesis/counter_obj  --lambda_sparsity_loss 0.2 --eval -i images_4 --start_checkpoint ./output/thesis/counter/chkpnt30000.pth --iterations 60000
CUDA_VISIBLE_DEVICES=0 python train.py -s data/360/kitchen  -m ./output/thesis/kitchen_obj  --lambda_sparsity_loss 0.2 --eval -i images_4 --start_checkpoint ./output/thesis/kitchen/chkpnt30000.pth --iterations 60000
CUDA_VISIBLE_DEVICES=1 python train.py -s data/360/room  -m ./output/thesis/room_obj  --lambda_sparsity_loss 0.2 --eval -i images_4 --start_checkpoint ./output/thesis/room/chkpnt30000.pth --iterations 60000


CUDA_VISIBLE_DEVICES=0 python train.py -s data/360/kitchen  -m ./output/thesis/kitchen  --lambda_sparsity_loss 0.2 --eval -i images_4 --iterations 60000
CUDA_VISIBLE_DEVICES=1 python train.py -s data/360/room  -m ./output/thesis/room  --lambda_sparsity_loss 0.2 --eval -i images_4 --iterations 60000
CUDA_VISIBLE_DEVICES=2 python train.py -s data/360/garden  -m ./output/thesis/garden  --lambda_sparsity_loss 0.2 --eval -i images_4 --iterations 60000


CUDA_VISIBLE_DEVICES=2 python render.py -m output/thesis/counter_obj
CUDA_VISIBLE_DEVICES=3 python render.py -m output/thesis/bonsai_obj
CUDA_VISIBLE_DEVICES=0 python render.py -m output/thesis/kitchen_obj
CUDA_VISIBLE_DEVICES=1 python render.py -m output/thesis/room_obj
CUDA_VISIBLE_DEVICES=1 python render.py -m output/thesis/room_obj
