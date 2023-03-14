import argparse
import sys
import os
import torch
import torchvision.transforms as transforms



from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import Generator
from datasets import ImageDataset



parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/screws_A', help='root directory of the dataset')
parser.add_argument('--category', type=str, default="screws_A", help='category of anomaly object')
parser.add_argument('--input_nc', type=int, default=4, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=256, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--weight', type=str, default='screws_A.pth', help='the weight file for generator')
parser.add_argument('--output_path', type=str, default='results', help='path to save the output images')
opt = parser.parse_args()
print(opt)


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Networks
netG_norm2abnl = Generator(opt.input_nc, opt.output_nc)

if opt.cuda:
    netG_norm2abnl.cuda()

# Load state dicts
checkpoint_path = "checkpoints/{}/{}".format(opt.category, opt.weight)
netG_norm2abnl.load_state_dict(torch.load(checkpoint_path))

# Set model's test mode
netG_norm2abnl.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_Img = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
mask = Tensor(opt.batchSize, 1, opt.size, opt.size)

# Dataset loader
transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

mask_transform = [
   transforms.Grayscale(1), 
   transforms.ToTensor()]

dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mask_transform=mask_transform, mode='test'), 
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

results_path = "{}/{}".format(opt.output_path,opt.category)
if not os.path.exists(results_path):
    os.makedirs(results_path)

for i, batch in enumerate(dataloader):

    real_norm = Variable(input_Img.copy_(batch['img_norm']))
    mask = Variable(mask.copy_(batch['mask_norm']))

    input_real_norm = torch.cat((real_norm, mask),dim=1)

    # Generate output
    generated_abnl = 0.45*(netG_norm2abnl(input_real_norm).data + 1.0)
    real_norm = 0.5*((real_norm) +1.0)

    
    # Save image files
    save_image(real_norm, '%s/%04d_norm.png' %(results_path,(i+1)))
    save_image(mask, '%s/%04d_mask.png' %(results_path,(i+1)))
    save_image(generated_abnl, '%s/%04d_generated.png' %(results_path,(i+1)))

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    
sys.stdout.write('\n')
