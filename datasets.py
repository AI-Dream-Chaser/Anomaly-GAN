from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mask_transform= None, mode='test'):
        super(ImageDataset, self).__init__()

        print("loading data...")

        self.root = root
        self.transform = transforms.Compose(transforms_)
        self.mask_transform = transforms.Compose(mask_transform)
        self.normalImg_paths = []  #path of anomaly-free image
        self.mask_paths = [] #path of mask for anomaly-free image

        #Load the txt path of dataset
        if mode == 'test':
            Normaltxt_path = self.root + "/normal" + "/ImageSets/Main/test.txt"  #anomaly-free image

        # Read the txt file to get the picture index
        Normaltxt = open(Normaltxt_path)
        Normal_lines = Normaltxt.readlines()

        self.normal_imgpath = self.root + "/normal" + "/JPEGImages/"
        self.normal_maskpath = self.root + "/normal" +"/mask/"

        for normal_item in Normal_lines:
            normal_name = normal_item.strip().split()[0]

            # Getting the path of img and mask
            self.normalImg_paths.append(self.normal_imgpath + normal_name + '.png')
            self.mask_paths.append(self.normal_maskpath + normal_name + '_label.png')    
      
    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, index, color_format='RGB'):

        img_norm = Image.open(self.normalImg_paths[index])
        mask_norm = Image.open(self.mask_paths[index])

        img_norm = img_norm.convert(color_format)
        mask_norm = mask_norm.convert(color_format)

        if self.transform is not None:
            img_norm = self.transform(img_norm)
            mask_norm = self.mask_transform(mask_norm)
            
        return {"img_norm":img_norm, "mask_norm":mask_norm}
