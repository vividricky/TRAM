import numpy as np
import cv2
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

from template import *
from utils.indicator import *


def contrast_augmentation(image,low=1.05,high=1.5):

    random_number = random.uniform(low, high)
    augmenter = iaa.ContrastNormalization(alpha=random_number, per_channel=True)
    augmented_image = augmenter(image=image)
    return augmented_image

def saturation_augmentation(image,low=0.5,high=0.9):

    augmenter = iaa.MultiplySaturation((low, high))
    augmented_image = augmenter.augment_image(image)
    return augmented_image

def colorfulness_augmentation(image,low=0.5,high=0.9):

    height, width, channels = image.shape
    ratio = random.uniform(low, high)
    max_values = np.max(image, axis=(0, 1))
    result = np.zeros_like(image)
    for i in range(channels):
        decrease_ratio = ratio * (max_values[i] / 255.0)
        result[:, :, i] = cv2.multiply(image[:, :, i], decrease_ratio)
    return result
def white_balance_augmentation(image, low_percent=-20, high_percent=20):

    percent = random.uniform(low_percent, high_percent)
    img_float = image.astype(float)
    b, g, r = cv2.split(img_float)

    avg_b = cv2.mean(b)[0]
    avg_g = cv2.mean(g)[0]
    avg_r = cv2.mean(r)[0]

    avg_intensity = (avg_b + avg_g + avg_r) / 3
    desired_avg_intensity = avg_intensity * (1 + percent / 100)

    b = b * (desired_avg_intensity / avg_b)
    g = g * (desired_avg_intensity / avg_g)
    r = r * (desired_avg_intensity / avg_r)

    result = cv2.merge((b, g, r))
    result = cv2.convertScaleAbs(result)
    return result

def facial_augmentation(image,fa):
    mask = image.copy()
    lmk = fa.get_landmarks(image)
    left_random = np.random.randint(-5,5)
    right_random = np.random.randint(-5,5)
    top_random = np.random.randint(-30,10)
    bottom_random = np.random.randint(-10,20)

    x_min, y_min = max(int(lmk[0][0][0])+left_random,0), max(int(int(lmk[0][16][1])- (2/3)*(int(lmk[0][8][1])- int(lmk[0][16][1])))+top_random,0)
    x_max, y_max = min(int(int(lmk[0][16][0])+left_random),image.shape[1]),min(int(lmk[0][8][1])+bottom_random,image.shape[0])

    mask = contrast_augmentation(mask,low=0.6,high=0.75)

    face_region = mask[y_min:y_max,x_min:x_max]
    image[y_min:y_max,x_min:x_max] = face_region

    return image

def adjust_color_balance(image, blue_gain, green_gain, red_gain):


    b, g, r = cv2.split(image)


    b = cv2.addWeighted(b, blue_gain, 0, 0, 0)
    g = cv2.addWeighted(g, green_gain, 0, 0, 0)
    r = cv2.addWeighted(r, red_gain, 0, 0, 0)


    result = cv2.merge((b, g, r))


    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def color_filter_augmentation(image):
    aug = adjust_color_balance(image,0.9,1,1.05)
    return aug

def generate_photometric_attribute_labels(anchor,inference,cls,paper_mask_flag,thres=0.95):
    if cls == 0:

        return [0,0,0,0]


    anchor_contrast = calculate_contrast(anchor)
    anchor_saturation = calculate_saturation(anchor)
    anchor_colorfulness = calculate_colorfulness(anchor)
    anchor_white_balance = calculate_white_balance(anchor)

    inference_contrast = calculate_contrast(inference)
    inference_saturation = calculate_saturation(inference)
    inference_colorfulness = calculate_colorfulness(inference)
    inference_white_balance = calculate_white_balance(inference)


    contrast_flag = inference_contrast/anchor_contrast
    contrast_flag = 1 if contrast_flag < thres else 0
    saturation_flag = inference_saturation/anchor_saturation
    saturation_flag = 1 if saturation_flag < thres else 0
    colorfulness_flag = inference_colorfulness/anchor_colorfulness
    colorfulness_flag = 1 if colorfulness_flag < thres else 0
    white_balance_flag = 1 if inference_white_balance/anchor_white_balance < thres or anchor_white_balance/inference_white_balance <thres  else 0

    return [contrast_flag,saturation_flag,colorfulness_flag,white_balance_flag]


def random_view(anchor,positive,negative,scale=(0.75,1)):


    random_crop = T.RandomResizedCrop(224,scale)
    anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
    anchor = Image.fromarray(anchor)
    positive = cv2.cvtColor(positive, cv2.COLOR_BGR2RGB)
    positive = Image.fromarray(positive)
    negative = cv2.cvtColor(negative, cv2.COLOR_BGR2RGB)
    negative = Image.fromarray(negative)

    anchor_view = random_crop.get_params(anchor, scale, random_crop.ratio)


    anchor_view_convert = F.resized_crop(anchor, *anchor_view, random_crop.size, random_crop.interpolation)


    positive_view_convert = F.resized_crop(positive, *anchor_view, random_crop.size, random_crop.interpolation)
    negative_view_convert = F.resized_crop(negative, *anchor_view, random_crop.size, random_crop.interpolation)

    anchor_view_convert = cv2.cvtColor(np.array(anchor_view_convert), cv2.COLOR_RGB2BGR)
    positive_view_convert = cv2.cvtColor(np.array(positive_view_convert), cv2.COLOR_RGB2BGR)
    negative_view_convert = cv2.cvtColor(np.array(negative_view_convert), cv2.COLOR_RGB2BGR)

    return anchor_view_convert, positive_view_convert,negative_view_convert


class AugDataset(Dataset):

    def __init__(self, data,train = True):
        self.data = data
        self.train = train
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.train:


            anchor_path = self.data.iloc[item, 0]
            positive_path = self.data.iloc[item, 1]
            anchor = cv2.imread(anchor_path)
            anchor = cv2.resize(anchor, (224, 224))
            positive = cv2.imread(positive_path)
            positive = cv2.resize(positive, (224, 224))

            augmentation_functions = ['contrast', 'saturation', 'colorfulness','face_region','white_balance']


            selected_index = random.randint(0, len(augmentation_functions))
            label  = selected_index
            if selected_index == 0:
                target = positive
            elif selected_index == 1:


                target = contrast_augmentation(anchor,low=1.05,high=1.5)
            elif selected_index == 2:


                target = saturation_augmentation(anchor,low=0.5,high=1)
            elif selected_index == 3:


                target = colorfulness_augmentation(anchor,low=0.5,high=1)
            elif selected_index == 4:
                target = facial_augmentation(anchor, self.fa)
            elif selected_index == 5:
                target = white_balance_augmentation(anchor, low_percent=-20, high_percent=20)


            anchor = self.transforms(anchor)
            target = self.transforms(target)
            return anchor, target,label

class ImageDataSet(Dataset):
    def __init__(self, data,thres=0.95,train = True):
        self.data = data
        self.train = train
        self.thres = thres
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.train:


            negative_path = self.data.iloc[item, 0]
            label = self.data.iloc[item, 1]
            anchor_path = self.data.iloc[item, 2]
            positive_path =self.data.iloc[item, 3]

            anchor = cv2.imread(anchor_path)
            positive = cv2.imread(positive_path)
            negative = cv2.imread(negative_path)
            anchor = cv2.resize(anchor, (224, 224))
            positive = cv2.resize(positive, (224, 224))
            negative = cv2.resize(negative, (224, 224))

            paper_mask_flag  = 0
            if label == 3 or label == 4:
                paper_mask_flag = label - 2
            if label > 0:
                label = 1

            real_inf_real_caption,real_inf_fake_caption = caption_generator_pair(anchor,positive,paper_mask_flag,self.thres)
            fake_inf_real_caption,fake_inf_fake_caption = caption_generator_pair(anchor,negative,paper_mask_flag,self.thres)

            indicators_positive = generate_photometric_attribute_labels(anchor,positive,0,paper_mask_flag,self.thres)
            indicators_negative = generate_photometric_attribute_labels(anchor,negative,1,paper_mask_flag,self.thres)

            anchor_view, positive_view,negative_view = random_view(anchor,positive,negative)


            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

            anchor_view = self.transforms(anchor_view)
            positive_view = self.transforms(positive_view)
            negative_view = self.transforms(negative_view)


            return anchor, positive, negative,anchor_view,positive_view,negative_view,label, [real_inf_real_caption,real_inf_fake_caption,fake_inf_real_caption,fake_inf_fake_caption], \
                                                     [indicators_positive,indicators_negative]

        else:
            inference_path = self.data.iloc[item, 0]
            label = self.data.iloc[item, 1]

            paper_mask_flag  = 0
            if label == 3 or label == 4:
                paper_mask_flag = label - 2

            if label > 0:
                label = 1
            reference_path = self.data.iloc[item, 2]
            inf_img = cv2.imread(inference_path)
            inf_img = cv2.resize(inf_img, (224, 224))

            if inf_img is None:
                print(inference_path)
            ref_img = cv2.imread(reference_path)
            ref_img = cv2.resize(ref_img, (224, 224))

            eval_real_template, eval_fake_template = caption_generator_pair(ref_img,inf_img,paper_mask_flag,self.thres)

            inf_img = self.transforms(inf_img)
            ref_img = self.transforms(ref_img)


            template = [eval_real_template,eval_fake_template]
            return inference_path, inf_img, ref_img, label, template

def get_dataset_aug(src1_data, src2_data, src3_data,tgt_data):
  data_root = '/B00120240002/data_list/'
  print('Load Source Data')
  print('Source Data: ', src1_data)
  print('Source Data: ', src2_data)
  print('Source Data: ', src3_data)

  src1_anchor_data = pd.read_csv(data_root+src1_data+'_real.csv',names=['path','label'])
  src1_positive_data = derange_df(src1_anchor_data['path'])
  src1_anchor_data = src1_anchor_data[['path']]
  src1_anchor_data['positive_path'] = src1_positive_data['path']
  batch_size = 3
  src1_anchor_dataloader = DataLoader(
      AugDataset(src1_anchor_data, train=True),
      batch_size=batch_size,
      shuffle=True,
      drop_last=True)

  src2_anchor_data = pd.read_csv(data_root+src2_data+'_real.csv',names=['path','label'])
  src2_positive_data = derange_df(src2_anchor_data['path'])
  src2_anchor_data = src2_anchor_data[['path']]
  src2_anchor_data['positive_path'] = src2_positive_data['path']
  src2_anchor_dataloader = DataLoader(
      AugDataset(src2_anchor_data, train=True),
      batch_size=batch_size,
      shuffle=True,
      drop_last=True)

  src3_anchor_data = pd.read_csv(data_root+src3_data+'_real.csv',names=['path','label'])
  src3_positive_data = derange_df(src3_anchor_data['path'])
  src3_anchor_data = src3_anchor_data[['path']]
  src3_anchor_data['positive_path'] = src3_positive_data['path']
  src3_anchor_dataloader = DataLoader(
      AugDataset(src3_anchor_data, train=True),
      batch_size=batch_size,
      shuffle=True,
      drop_last=True)

  batch_size = 8
  tgt_anchor_data = pd.read_csv(data_root+tgt_data+'_real.csv',names=['path','label'])
  tgt_positive_data = derange_df(tgt_anchor_data['path'])
  tgt_anchor_data = tgt_anchor_data[['path']]
  tgt_anchor_data['positive_path'] = tgt_positive_data['path']
  tgt_anchor_data = DataLoader(
      AugDataset(tgt_anchor_data, train=True),
      batch_size=batch_size,
      shuffle=False)


  return [src1_anchor_dataloader,src2_anchor_dataloader,src3_anchor_dataloader],tgt_anchor_data


def get_dataset(src1_data, src2_data, src3_data,tgt_data,thres=0.9):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'], index_col=False)
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'], index_col=False)
    src3_train_data = pd.read_csv(data_root+src3_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'], index_col=False)

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'], index_col=False)


    train_data = [src1_train_data,src2_train_data,src3_train_data]

    batch_size = 2

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        ImageDataSet(dataframe, thres=thres,train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
        train_dataloader.append(src)

    batch_size = 5
    tgt_dataloader = DataLoader(
      ImageDataSet(tgt_test_data, train=False),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_input(path_list,thres=0.9):
    data_root = '/B00120240002/data_list/'
    tgt_test_data =pd.read_csv(data_root+path_list,names=['inf_path','label','ref_path'])
    batch_size = 5
    tgt_dataloader = DataLoader(
      ImageDataSet(tgt_test_data, train=False),
      batch_size=batch_size,
      shuffle=False)

    return tgt_test_data


def get_dataset_four_to_one(src1_data, src2_data, src3_data,src4_data, tgt_data,thres=0.9):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src3_train_data = pd.read_csv(data_root+src3_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src4_train_data = pd.read_csv(data_root+src4_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data,src3_train_data,src4_train_data]

    batch_size = 2

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        ImageDataSet(dataframe, thres=thres,train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
        train_dataloader.append(src)

    batch_size = 5
    tgt_dataloader = DataLoader(
      ImageDataSet(tgt_test_data, train=False),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_W(src1_data, src2_data, src3_data,tgt_data,thres=0.9):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src3_train_data = pd.read_csv(data_root+src3_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference_ref_W.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data,src3_train_data]

    batch_size = 2

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        ImageDataSet(dataframe, thres=thres,train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
        train_dataloader.append(src)

    batch_size = 5
    tgt_dataloader = DataLoader(
      ImageDataSet(tgt_test_data, train=False),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_one_to_one(src_data,tgt_data,thres=0.9):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    batch_size = 2

    src = DataLoader(
        ImageDataSet(src1_train_data, thres=thres,train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    batch_size = 5
    tgt_dataloader = DataLoader(
      ImageDataSet(tgt_test_data, train=False),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ [src],tgt_dataloader]
    return data_loaders_list


def get_dataset_two_to_one(src1_data, src2_data,tgt_data,thres=0.9):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data]

    train_dataloader =[]

    batch_size = 3
    src = DataLoader(
    ImageDataSet(src1_train_data, thres=thres,train=True),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)
    train_dataloader.append(src)

    batch_size = 3
    src_2 = DataLoader(
    ImageDataSet(src2_train_data, thres=thres,train=True),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)
    train_dataloader.append(src_2)

    batch_size = 5
    tgt_dataloader = DataLoader(
      ImageDataSet(tgt_test_data, train=False),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

class TRAMImagePairDataset(Dataset):
    def __init__(self, data,train = True,thres=0.8,shrink_pixel=5):
        self.data = data
        self.train = train
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms =  T.Compose([
            T.ToTensor(),
        ])
        self.thres = thres
        self.shrink_pixel = shrink_pixel
    def __len__(self):
        return len(self.data)

    def replace_crop_with_face_mask(self,path):
        parts = [p for p in path.split('/') if p]

        if len(parts) > 3 and parts[2].endswith('_crop'):

            new_part = parts[2].replace('_crop', '_face_mask')

            parts.insert(2, new_part)


        new_path = '/' + '/'.join(parts)
        return new_path

    def shrink_mask(self,mask, pixel):
        kernel = np.ones((pixel, pixel), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return eroded


    def __getitem__(self, item):
        if self.train:


            negative_path = self.data.iloc[item, 0]
            label = self.data.iloc[item, 1]
            anchor_path = self.data.iloc[item, 2]
            positive_path =self.data.iloc[item, 3]

            negative_path_mask = self.replace_crop_with_face_mask(negative_path)
            anchor_path_mask = self.replace_crop_with_face_mask(anchor_path)
            positive_path_mask = self.replace_crop_with_face_mask(positive_path)


            anchor = cv2.imread(anchor_path)
            anchor = cv2.resize(anchor,(224,224))
            positive = cv2.imread(positive_path)
            positive = cv2.resize(positive,(224,224))
            negative = cv2.imread(negative_path)
            negative = cv2.resize(negative,(224,224))

            anchor_mask = cv2.imread(anchor_path_mask,cv2.IMREAD_GRAYSCALE)
            anchor_mask = cv2.resize(anchor_mask,(224,224))
            positive_mask = cv2.imread(positive_path_mask,cv2.IMREAD_GRAYSCALE)
            positive_mask = cv2.resize(positive_mask,(224,224))
            negative_mask = cv2.imread(negative_path_mask,cv2.IMREAD_GRAYSCALE)
            negative_mask = cv2.resize(negative_mask,(224,224))

            anchor_mask = self.shrink_mask(anchor_mask,self.shrink_pixel)
            positive_mask = self.shrink_mask(positive_mask,self.shrink_pixel)
            negative_mask = self.shrink_mask(negative_mask,self.shrink_pixel)

            paper_mask_flag  = 0


            real_inf_real_caption,real_inf_fake_caption = caption_generator_pair(anchor,positive,paper_mask_flag,self.thres)
            fake_inf_real_caption,fake_inf_fake_caption = caption_generator_pair(anchor,negative,paper_mask_flag,self.thres)

            indicators_positive = generate_photometric_attribute_labels(anchor,positive,0,paper_mask_flag,self.thres)
            indicators_negative = generate_photometric_attribute_labels(anchor,negative,1,paper_mask_flag,self.thres)


            anchor = self.transforms(anchor)
            positive = self.transforms(positive)
            negative = self.transforms(negative)

            anchor_mask = self.mask_transforms(anchor_mask)
            positive_mask = self.mask_transforms(positive_mask)
            negative_mask = self.mask_transforms(negative_mask)


            return anchor, positive, negative,anchor_mask,positive_mask,negative_mask,label, [real_inf_real_caption,real_inf_fake_caption,fake_inf_real_caption,fake_inf_fake_caption], \
                                                     [indicators_positive,indicators_negative]
        else:
            inference_path = self.data.iloc[item, 0]
            label = self.data.iloc[item, 1]

            paper_mask_flag  = 0


            if label > 0:
                label = 1
            reference_path = self.data.iloc[item, 2]
            inf_img = cv2.imread(inference_path)
            ref_img = cv2.imread(reference_path)

            inference_path_mask = self.replace_crop_with_face_mask(inference_path)
            reference_path_mask = self.replace_crop_with_face_mask(reference_path)

            inf_mask = cv2.imread(inference_path_mask,cv2.IMREAD_GRAYSCALE)
            ref_mask = cv2.imread(reference_path_mask,cv2.IMREAD_GRAYSCALE)

            inf_mask = cv2.resize(inf_mask,(224,224))
            ref_mask = cv2.resize(ref_mask,(224,224))


            inf_mask = self.shrink_mask(inf_mask,self.shrink_pixel)
            ref_mask = self.shrink_mask(ref_mask,self.shrink_pixel)

            eval_real_template, eval_fake_template = caption_generator_pair(ref_img,inf_img,paper_mask_flag,self.thres)

            inf_img = inf_img.astype(np.float32)
            inf_img = cv2.cvtColor(inf_img, cv2.COLOR_BGR2RGB)
            inf_img = Image.fromarray(inf_img.astype(np.uint8)).resize((224, 224))
            inf_img = self.transforms(inf_img)

            ref_img = ref_img.astype(np.float32)
            ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            ref_img = Image.fromarray(ref_img.astype(np.uint8)).resize((224, 224))
            ref_img = self.transforms(ref_img)

            inf_mask = self.mask_transforms(inf_mask)
            ref_mask = self.mask_transforms(ref_mask)


            videoID = '/'.join(inference_path.split('/')[:-1])
            template = [eval_real_template,eval_fake_template]

            return inference_path, inf_img, ref_img, inf_mask, ref_mask, label, template,videoID


class TRAMFewShotDataset(Dataset):
    def __init__(self, data,train = True,thres=0.8,shrink_pixel=5):
        self.data = data
        self.train = train
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transforms =  T.Compose([
            T.ToTensor(),
        ])
        self.thres = thres
        self.shrink_pixel = shrink_pixel
    def __len__(self):
        return len(self.data)

    def replace_crop_with_face_mask(self,path):
        parts = [p for p in path.split('/') if p]

        if len(parts) > 3 and parts[2].endswith('_crop'):

            new_part = parts[2].replace('_crop', '_face_mask')

            parts.insert(2, new_part)


        new_path = '/' + '/'.join(parts)
        return new_path

    def shrink_mask(self,mask, pixel):
        kernel = np.ones((pixel, pixel), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        return eroded


    def __getitem__(self, item):

        inference_path = self.data.iloc[item, 0]
        label = self.data.iloc[item, 1]

        paper_mask_flag = 0

        if label > 0:
            label = 1


        reference_paths = self.data.iloc[item, 2:].dropna().tolist()
        num_refs = len(reference_paths)


        inf_img = cv2.imread(inference_path)
        ref_imgs = [cv2.imread(p) for p in reference_paths]


        inference_path_mask = self.replace_crop_with_face_mask(inference_path)
        reference_path_masks = [
            self.replace_crop_with_face_mask(p)
            for p in reference_paths
        ]


        inf_mask = cv2.imread(inference_path_mask, cv2.IMREAD_GRAYSCALE)
        ref_masks = [
            cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            for p in reference_path_masks
        ]


        inf_mask = cv2.resize(inf_mask, (224, 224))
        ref_masks = [cv2.resize(m, (224, 224)) for m in ref_masks]


        inf_mask = self.shrink_mask(inf_mask, self.shrink_pixel)


        eval_real_template, eval_fake_template = caption_generator_pair(
            ref_imgs[0], inf_img, paper_mask_flag, self.thres
        )


        def process_img(img):
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            return self.transforms(img)

        inf_img = process_img(inf_img)
        ref_imgs = [process_img(img) for img in ref_imgs]


        inf_mask = self.mask_transforms(inf_mask)
        ref_masks = [self.mask_transforms(m) for m in ref_masks]


        videoID = '/'.join(inference_path.split('/')[:-1])
        template = [eval_real_template, eval_fake_template]

        return (
            inference_path,
            inf_img,
            ref_imgs,
            inf_mask,
            ref_masks,
            label,
            template,
            videoID
        )
def get_dataset_one_to_one_mask(src1_data, src2_data,tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data]

    batch_size = batch_size

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        TRAMImagePairDataset(dataframe, thres=thres,train=True,shrink_pixel=shrink_pixel),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
        train_dataloader.append(src)

    batch_size = 16
    tgt_dataloader = DataLoader(
      TRAMImagePairDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_one_to_one_mask_wo_celeba(src1_data,tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])
    train_data = [src1_train_data]

    batch_size = batch_size

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        TRAMImagePairDataset(dataframe, thres=thres,train=True,shrink_pixel=shrink_pixel),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
        train_dataloader.append(src)

    batch_size = 16
    tgt_dataloader = DataLoader(
      TRAMImagePairDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)

    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_one_to_one_mask_WH_inference(src1_data,tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data = pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])
    tgt_test_data['ref_path'] = tgt_test_data['ref_path'][0]


    train_data = [src1_train_data]

    batch_size = batch_size

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        TRAMImagePairDataset(dataframe, thres=thres,train=True,shrink_pixel=shrink_pixel),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
        train_dataloader.append(src)

    batch_size = 16
    tgt_dataloader = DataLoader(
      TRAMImagePairDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)

    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_two_to_one_mask(src1_data, src2_data,src3_data,tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src3_train_data = pd.read_csv(data_root+src3_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data,src3_train_data]

    batch_size = batch_size

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        TRAMImagePairDataset(dataframe, thres=thres,train=True,shrink_pixel=shrink_pixel),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
        train_dataloader.append(src)

    batch_size = 16
    tgt_dataloader = DataLoader(
      TRAMImagePairDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_dataset_two_to_one_mask_wo_celeba(src1_data, src2_data,tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])


    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data]

    batch_size = batch_size

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        TRAMImagePairDataset(dataframe, thres=thres,train=True,shrink_pixel=shrink_pixel),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
        train_dataloader.append(src)

    batch_size = 16
    tgt_dataloader = DataLoader(
      TRAMImagePairDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list

def get_tram_dataloaders(src1_data, src2_data,src3_data,src4_data,tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    src1_train_data = pd.read_csv(data_root+src1_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src2_train_data = pd.read_csv(data_root+src2_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src3_train_data = pd.read_csv(data_root+src3_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])
    src4_train_data = pd.read_csv(data_root+src4_data+'_train_reference.csv',names=['path','label','ref_path','inf_real_path'])

    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference.csv',names=['inf_path','label','ref_path'])


    train_data = [src1_train_data,src2_train_data,src3_train_data,src4_train_data]

    batch_size = batch_size

    train_dataloader =[]
    for dataframe in train_data:
        src = DataLoader(
        TRAMImagePairDataset(dataframe, thres=thres,train=True,shrink_pixel=shrink_pixel),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
        train_dataloader.append(src)

    batch_size = 8
    tgt_dataloader = DataLoader(
      TRAMImagePairDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = [ train_dataloader,tgt_dataloader]
    return data_loaders_list


def get_tram_few_shot_dataloader(tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'
    tgt_test_data =pd.read_csv(data_root+tgt_data+'_all_reference_5_shots.csv',names=['inf_path','label','ref_path_0','ref_path_1','ref_path_2','ref_path_3','ref_path_4'])


    batch_size = 4
    tgt_dataloader = DataLoader(
      TRAMFewShotDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = tgt_dataloader
    return data_loaders_list

def get_tram_multi_anchor_dataloader(tgt_data,thres=0.9,shrink_pixel=5,batch_size=2):
    data_root = '/B00120240002/data_list/'

    tgt_test_data = pd.read_csv(data_root + tgt_data + '_all_reference_30_shots.csv',
                    header=None)


    num_cols = tgt_test_data.shape[1]

    columns = ['inf_path', 'label'] + [
        f'ref_path_{i}' for i in range(num_cols - 2)
    ]

    tgt_test_data.columns = columns

    batch_size = 1
    tgt_dataloader = DataLoader(
      TRAMFewShotDataset(tgt_test_data, train=False,shrink_pixel=shrink_pixel),
      batch_size=batch_size,
      shuffle=False)


    data_loaders_list = tgt_dataloader
    return data_loaders_list
