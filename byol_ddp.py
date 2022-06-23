import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from datetime import datetime
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as autocast, GradScaler

from byol import BYOL
from dataset import ISICUnlabeled
from utils import get_model
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='densenet121')
parser.add_argument('--save_dir', type=str, default='./byol_save')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_gpu', type=int, default=4)
parser.add_argument('--local_rank', type=int)
parser.add_argument('--use_amp', type=bool, default=True)
args = parser.parse_args()

def save_dict_to_text(mydict, path):
    stream = [str(k)+': '+str(v)+'\n' for k,v in mydict.items()]
    stream = reduce(lambda x, y: x+y, stream)
    with open(path, 'w') as f:
        f.writelines(stream)

def main():
    torch.distributed.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # To make sure each process has a different seed
    print(f"loacal_rank {local_rank}: cuda_seed: {torch.cuda.initial_seed()} seed: {torch.initial_seed()}")

    transform=transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
    train_dataset = ISICUnlabeled(transform=transform)
    num_steps = len(train_dataset) // args.batch_size
    if local_rank == 0:
        writer = SummaryWriter(log_dir=args.save_dir)
        print(f"Dataset length: {len(train_dataset)}")
        print(f"Total steps per epoch: {num_steps}")

    train_sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size//args.num_gpu,
                            num_workers=8, sampler=train_sampler)
    
    model = get_model(args.model, 4).cuda(args.local_rank)
    learner = BYOL(
        model,
        image_size = 224,
        hidden_layer = 'avgpool',
        projection_size = 256,           # the projection size
        projection_hidden_size = 4096,   # the hidden dimension of the MLP for both the projection and prediction
        moving_average_decay = 0.99,     # the moving average decay factor for the target encoder, already set at what paper recommends
        use_momentum = False)      
    learner = DDP(learner, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    #opt = torch.optim.SGD(learner.parameters(), lr=0.02* args.batch_size / 256, momentum=0.9, nesterov=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    optimizer = torch.optim.AdamW(learner.parameters(), lr=1e-4 * args.batch_size / 256, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    if args.use_amp:
        scaler = GradScaler()
        if local_rank == 0:
            print('AMP activated')

    if local_rank == 0: # Save ckpt_0 for debug
        print(f"Model type = {args.model}")
        save_dict_to_text(vars(args), f'{args.save_dir}/args.text')
        torch.save(learner.module.online_encoder.net.state_dict(), f'{args.save_dir}/byol_{0}.pth')

    for epoch in range(1, 1000+1):
        train_sampler.set_epoch(epoch)
        if local_rank == 0:
            start_time = time.time()
        for step, data in enumerate(dataloader):
            images = data[0]
            images = images.cuda(non_blocking=True)
            optimizer.zero_grad()
            if args.use_amp:
                with autocast():
                    loss = learner(images)
                loss_value = loss.detach().cpu()
                scaler.scale(loss).backward()
                #total_norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), 2*65536)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = learner(images)
                loss_value = loss.detach().cpu()
                loss.backward()
                #torch.nn.utils.clip_grad_norm_(learner.parameters(), 2)
                optimizer.step()

            if local_rank == 0:
                writer.add_scalar("Loss", loss_value, step + epoch*num_steps)
                writer.add_scalar("LR", optimizer.param_groups[0]['lr'], step + epoch*num_steps)
                if step%50 == 0:
                    print("%s:" % str(datetime.now())[:19] , end=" ")
                    print(f"Epoch {epoch}, Step {step}, Loss: {loss_value}, ", end="")
                    print("time_used: {:.3}".format(time.time()-start_time))
                    start_time = time.time()

        if local_rank == 0:
            scheduler.step()
            if epoch % 10 == 0 :
                torch.save(learner.module.online_encoder.net.state_dict(), f'{args.save_dir}/byol_{epoch}.pth')
            

if __name__ == '__main__':
    main()

