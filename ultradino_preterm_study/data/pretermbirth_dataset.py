import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T
import torch
import cv2

import itertools

class PretermBirthDatasetBase(torch.utils.data.Dataset):
    """
    Base class for the Preterm Birth Dataset.
    Args:
        split (str): Dataset split to use ('train', 'vali', 'test', 'valitest', 'all').
        csv_dir (str): Path to the CSV file containing metadata and file paths.
        split_index (str): Column name in the CSV file indicating the data split.
    Returns:
        image (Tensor): Ultrasound image tensor.
        image_with_calipers (Tensor): Ultrasound image with calipers tensor.
        segmentation_mask (Tensor): Segmentation mask tensor.
        binary_mask (Tensor): Binary mask tensor.
        segmentation_logits (Tensor): Segmentation logits tensor.
        cpr (int): Unique identifier for the patient.
    """
    def __init__(
        self, 
        split:str = 'train', 
        csv_dir:str = '/data/proto/sPTB-SA-SonoNet/metadata/ASMUS_MICCAI_dataset_splits.csv',
        split_index:str = 'fold_1',
        resample = False,
        **kwargs,
    ):
        super().__init__()

        assert split in ['train', 'vali', 'test', 'valitest', 'all']
        
        # read file name list
        csv = pd.read_csv(csv_dir, low_memory=False)

        if split =='train':
            csv = csv[csv[split_index]=='train']
            #csv = csv[csv['Depth_Groups'] != "7 to 8 mm"]
            
        elif split =='vali':
            csv = csv[csv[split_index]=='vali']
            #csv = csv[csv['Depth_Groups'] != "7 to 8 mm"]
            #csv = csv[csv['physical_delta_x'] < 0.0134]
        elif split =='test':
            csv = csv[csv[split_index]=='test']
            
            #csv = csv[csv['physical_delta_x'] > 0.0134]
            #list_cpr =  [1704831982, 103951402]
            #csv = csv#[csv.cpr.isin(list_cpr)]               
        elif split == 'valitest':
            vali_csv = csv[csv[split_index]=='vali']
            test_csv = csv[csv[split_index]=='test']
            csv = pd.concat((vali_csv, test_csv), axis=0)
            csv = pd.concat((vali_csv, test_csv, csv), axis=0)
        elif split == 'all':
            pass
        else:
            csv = csv[csv[split_index]==split]
        
        ## TO TEST
        #csv = csv.sample(100)
        self.csv = csv.reset_index(drop=True)
        self.df_points = pd.read_csv('/data/proto/Joris/Data/Masks_Cervix/df_points.csv', index_col=-1)
        self.split = split
        self.split_index = split_index
        self.landmarks = ['C_L', 'C_R', 'OB_UL', 'OB_UR', 'OB_DL','OB_DR', 'IB_UL', 'IB_UR', 'IB_DL', 'IB_DR']
        self.resample = resample
        
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]
    
    def __getitem__(self, index):
        
        # This loads objects from csv and sends to preprocessing
        
        if not self.resample:
            # read image
            image_wo_calipers_dir = self._get_attr(index, 'image_dir_clean')
            image = Image.open(image_wo_calipers_dir)
            image = np.asarray(image)
            
            # read mask
            mask_dir = self._get_attr(index, 'mask_filled')
            if mask_dir == "False":
                mask_dir = self._get_attr(index, 'mask_smoothed')
                if mask_dir == "False":
                    mask_dir = self._get_attr(index, 'mask_dir_png')
                    
            segmentation_mask = cv2.imread(mask_dir, 0).astype(np.float32) / 8
            #segmentation_data = np.load(mask_dir, allow_pickle=True).item()
            #segmentation_mask = segmentation_data['zahra']
            #segmentation_mask[segmentation_mask>4] = 0
            
        else:
            # read image
            image_wo_calipers_dir = self._get_attr(index, 'image_clean_resampled')
            image = Image.open(image_wo_calipers_dir)
            image = np.asarray(image)
            
            # read mask
            mask_dir = self._get_attr(index, 'mask_resampled')
            segmentation_mask = cv2.imread(mask_dir, 0).astype(np.float32) / 8
            # Test: predict preterm with ultradino from the mask alone and see
        
        #print(segmentation_mask.shape, image.shape)
        CL = self._get_attr(index, 'cervical_length')
        
        return image, segmentation_mask, image_wo_calipers_dir, CL

    def __len__(self):
        return len(self.csv)

class PretermBirthDataset(PretermBirthDatasetBase):
    """
    PyTorch Dataset class for the Preterm Birth Dataset.
    Args:
        transforms (list): List of Albumentations transformations to apply.
        label_name (str): Column name in the CSV file for the target label.
        class_only (int): If set to 0, only include term births; if set to 1, only include preterm births; if -1, include all.
    Returns:
        data (dict): Dictionary containing processed tensors and metadata.
    Note:
        - Images are normalized to the range [-1, 1] and converted to grayscale.
        - Pixel spacing information is included as additional channels.
        - The dataset can be filtered based on the target label using the `class_only` parameter.
    """
    def __init__(self, 
                 transforms, 
                 label_name:str = 'birth_before_week_37', 
                 input_type:str = 'image',
                 resample: bool = False,
                 class_only:int = -1,
                 **kwargs           
    ):
        super().__init__(resample=resample, **kwargs)

        assert class_only in [-1, 0, 1] # if 0 get only term if 1 get only preterm

        if class_only!=-1:
            self.csv = self.csv [self.csv[label_name] == class_only]
            self.csv = self.csv.reset_index(drop=True)

        self.transforms = transforms
        self.label_name = label_name
        self.input_type = input_type
        
    def __getitem__(self, index):
        
        data = {}
        
        if self.input_type == 'image':
        
            return self.preprocess_image(index)
            
        elif self.input_type == 'mask_only':
        
            return self.preprocess_maskonly(index)
            
        elif self.input_type == "mask_img":
        
            return self.preprocess_mask_img(index)
            
        
            
    def preprocess_image(self, index):
    
        image, segmentation_mask, image_wo_calipers_dir, CL = super().__getitem__(index)
        
        # get pixel spacing for resized images from metadata csv file
        px_spacing_resized = float(self._get_attr(index, 'px_spacing')) 
        py_spacing_resized = float(self._get_attr(index, 'py_spacing'))
        
        
        image = self.transforms(image=image)['image']#, mask=segmentation_mask)
        
        pixel_spacing = torch.tensor([px_spacing_resized, py_spacing_resized]).float()
      
       # get label from metadata csv file
        label = float(self._get_attr(index, f'{self.label_name}'))

        # wrapup
        data = {}
        
        data['label'] = label
        data['CL'] = float(CL)
        
        data['ps'] = pixel_spacing
        data['image'] = image
        data['segmentation_mask'] =  image
        data['image_dir_clean'] = image_wo_calipers_dir

        return data
    
    def get_landmark_tensor(self, image_wo_calipers_dir):
        
        try:
            points = self.df_points.loc[image_wo_calipers_dir]

            landmark_tensor = []
            
            for landmark in self.landmarks:
                #landmark_dict[landmark] = np.array([points[landmark+'_x'], points[landmark+'_y']])
                landmark_tensor.append(np.array([points[landmark+'_x'], points[landmark+'_y']]))
        
            return True, torch.tensor(np.array(landmark_tensor)).float()
        except:
            
            return False, torch.tensor(np.zeros(45)).float()
        
        
    def geometric_encoder(self, points):
        
        N, _ = points.shape
        D = []
        
        for i,j in itertools.combinations(range(N), 2):
            d = torch.norm(points[i] - points[j])
            
            D.append(d)
        
        D = torch.stack(D)
        D = D / D.mean()
        
        return D
        
    def preprocess_maskonly(self, index):
    
        image, segmentation_mask, image_wo_calipers_dir, CL = super().__getitem__(index)
        do, landmark_tensor = self.get_landmark_tensor(image_wo_calipers_dir)
        if do:
            landmark_tensor = self.geometric_encoder(landmark_tensor)
        
        
        # get pixel spacing for resized images from metadata csv file
        px_spacing_resized = float(self._get_attr(index, 'px_spacing')) 
        py_spacing_resized = float(self._get_attr(index, 'py_spacing'))
        
        image = self.transforms(image=segmentation_mask)['image']#.unsqueeze(0)#, mask=segmentation_mask)
        
        
        pixel_spacing = torch.tensor([px_spacing_resized, py_spacing_resized]).float()
      
       # get label from metadata csv file
        label = float(self._get_attr(index, f'{self.label_name}'))

        # wrapup
        data = {}
        
        data['label'] = label
        data['CL'] = float(CL)
        
        data['ps'] = pixel_spacing
        data['image'] = image
        data['segmentation_mask'] =  image
        data['image_dir_clean'] = image_wo_calipers_dir
        
        data['landmarks'] = landmark_tensor

        return data
        
    def preprocess_mask_img(self, index):
    
        image, segmentation_mask, image_wo_calipers_dir, CL = super().__getitem__(index)
        do, landmark_tensor = self.get_landmark_tensor(image_wo_calipers_dir)
        
        if do:
            landmark_tensor = self.geometric_encoder(landmark_tensor)
        
        
        # get pixel spacing for resized images from metadata csv file
        px_spacing_resized = float(self._get_attr(index, 'px_spacing')) 
        py_spacing_resized = float(self._get_attr(index, 'py_spacing'))
        
        inputs = self.transforms(image=image, mask = segmentation_mask)#.unsqueeze(0)#, mask=segmentation_mask)
        image, segmentation_mask = inputs['image'], inputs['mask'].unsqueeze(0)
        
        pixel_spacing = torch.tensor([px_spacing_resized, py_spacing_resized]).float()
      
       # get label from metadata csv file
        label = float(self._get_attr(index, f'{self.label_name}'))

        # wrapup
        data = {}
        
        data['label'] = label
        data['CL'] = float(CL)
        
        data['ps'] = pixel_spacing
        data['image'] = image + segmentation_mask
        data['segmentation_mask'] =  segmentation_mask
        data['image_dir_clean'] = image_wo_calipers_dir
        
        data['landmarks'] = landmark_tensor

        return data
           
    def _get_label(self, index):
        label = int(self._get_attr(index, self.label_name))
        return label
    
    def get_labels(self):
        labels = np.array([self._get_label(l) for l in range(len(self.csv))])
        return labels



if __name__ == "__main__":
    import albumentations as A
    from matplotlib import pyplot as plt
    tfs = [# A.Resize(224, 288),
           # A.CLAHE(p=1),
           # A.HorizontalFlip(p=1),
           #A.RandomBrightnessContrast(p=1),
           # A.Rotate(limit=[-25, 25], p=1, border_mode=0),
           A.VerticalFlip(p=1),
           # A.RandomGamma(p=1),
           A.Resize(224, 288)
           ]


    # label_name = 'birth_before_week_37'
    # class_only = -1
    # label_names = []
    # split_index= 'fold_5'
    # zoom=False
    # sigma=0

    # traindata = PretermBirthImageDataset(split='train', transforms=tfs, label_name=label_name, class_only=class_only, label_names=label_names, split_index=split_index, zoom=zoom, sigma=sigma)
    # print(f"training data: {len(traindata)}")
    # valdata = PretermBirthImageDataset(split='vali', transforms=tfs, label_name=label_name, class_only=class_only, label_names=label_names, split_index=split_index, zoom=zoom, sigma=sigma)
    # print(f"validation data: {len(valdata)}")
    # testdata = PretermBirthImageDataset(split='test', transforms=tfs, label_name=label_name, class_only=class_only, label_names=label_names, split_index=split_index, zoom=zoom, sigma=sigma)
    # print(f"test data: {len(testdata)}")

    label_name = 'birth_before_week_37'
    class_only = -1
    label_names = []
    split_index= 'fold_5B'
    zoom=False
    sigma=0

    csv_dir = '/data/proto/sPTB-SA-SonoNet/metadata/external_testset_all.csv'

    testdata = PretermBirthImageDataset(csv_dir=csv_dir, split='all', transforms=tfs, label_name=label_name, class_only=class_only)
    print(f"test data: {len(testdata)}")




    loader = torch.utils.data.DataLoader(testdata, batch_size=1,shuffle=False)

    for i, data in enumerate(loader):
        if i==1:break
        image, label, mask = data['image'], data['label'], data['binary_mask']
        segmentation_mask = data['segmentation_mask']
        segmentation_logits = data['segmentation_logits']

        print('image',image.shape, image.min(), image.max())
        print('label', label)
        #print('target', target)
        print('mask',mask.shape, mask.min(), mask.max())
        print('dtu mask', segmentation_mask.shape, segmentation_mask.min(), segmentation_mask.max())
        print()

        plt.figure()
        plt.subplot(1,4,1)

        plt.imshow((image.squeeze(0).squeeze(0).numpy()+1)/2, cmap='gray')
        
        plt.subplot(1,4,2)
        plt.imshow((mask.squeeze(0).squeeze(0).numpy()+1)/2, cmap='gray')
        plt.subplot(1,4,3)
        plt.imshow(torch.argmax(segmentation_logits, dim=1).squeeze(0).squeeze(0))
        plt.subplot(1,4,4)
        plt.imshow(segmentation_mask.squeeze(0).squeeze(0))
        plt.show()
