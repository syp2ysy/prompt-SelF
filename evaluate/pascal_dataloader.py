
import os
from PIL import Image
from scipy.io import loadmat
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from mae_utils import PURPLE, YELLOW
from tqdm import tqdm 
from prompt_self_utils import  extract_image_feature, extract_image_feature_what
import cv2

def create_grid_from_images_old(canvas, support_img, support_mask, query_img, query_mask):
   canvas[:, :support_img.shape[1], :support_img.shape[2]] = support_img
   canvas[:, -query_img.shape[1]:, :query_img.shape[2]] = query_img
   canvas[:, :support_img.shape[1], -support_img.shape[2]:] = support_mask
   canvas[:, -query_img.shape[1]:, -support_img.shape[2]:] = query_mask
   return canvas


def apply_edge_detection(mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                   [-2., 0., 2.],
                                   [-1., 0., 1.]], device=mask.device).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                   [0., 0., 0.],
                                   [1., 2., 1.]], device=mask.device).unsqueeze(0).unsqueeze(0)

    grad_x = F.conv2d(mask, sobel_kernel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_kernel_y, padding=1)

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_magnitude = (grad_magnitude - grad_magnitude.min()) / (grad_magnitude.max() - grad_magnitude.min() + 1e-8)
    edge_mask = (grad_magnitude > 0.1).float()
    edge_mask = edge_mask.squeeze(0)
    return edge_mask


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, image_transform, mask_transform, padding: bool = 1, use_original_imgsize: bool = False, flipped_order: bool = False,
                 reverse_support_and_query: bool = False, random: bool = False, ensemble: bool = False, purple: bool = False, prompt_search: bool = False,
                 what_makes: bool = False, edge_detection: bool = False, inpainting: bool = False):
        self.fold = fold
        self.nfolds = 4
        self.flipped_order = flipped_order
        self.nclass = 20
        self.padding = padding
        self.random = random
        self.ensemble = ensemble
        self.purple = purple
        self.use_original_imgsize = use_original_imgsize
        self.datapth = datapath
        self.prompt_search = prompt_search
        self.what_makes = what_makes
        self.edge_detection = edge_detection
        self.inpainting = inpainting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.img_path = os.path.join(datapath, 'VOCdevkit/VOC2012/JPEGImages/')
        self.ann_path = os.path.join(datapath, 'VOCdevkit/VOC2012/SegmentationClassAug/')
        self.image_transform = image_transform
        self.reverse_support_and_query = reverse_support_and_query
        self.mask_transform = mask_transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        
        if self.prompt_search:
            import timm
            print("Loading feature model...")
            self.feature_model_input_size = 518
            self.feature_model_name = 'vit_large_patch14_reg4_dinov2.lvd142m'

            # Load feature extraction model
            feature_model_path = os.path.join('../model_pth', 'timm', self.feature_model_name, 'pytorch_model.bin')
            if not os.path.exists(feature_model_path):
                raise FileNotFoundError(f"Feature model checkpoint not found: {feature_model_path}")

            self.feature_model = timm.create_model(
                model_name=self.feature_model_name,
                pretrained=False,
                checkpoint_path=str(feature_model_path)
            ).eval().to(self.device)
            print('Feature extraction model loaded.')

            # Prepare cache directory
            if self.what_makes:
                self.cache_dir = os.path.join(self.datapth, 'VOCdevkit/VOC2012/cache_what', self.feature_model_name)
                self.feature_model_name = 'vit_large_patch14_clip_336.openai_ft_in12k_in1k'
                # self.feature_model_input_size = 336
            else:
                self.cache_dir = os.path.join(self.datapth, 'VOCdevkit/VOC2012/cache', self.feature_model_name)
            
            os.makedirs(self.cache_dir, exist_ok=True)

            # Build class_to_train_features and class_to_train_image_paths
            self.class_to_train_features = {}  # {class_id: [features]}
            self.class_to_train_image_paths = {}  # {class_id: [image_names]}

            # Read training image names and their class IDs
            train_fold_file = os.path.join(self.datapth, 'splits/pascal/trn/fold%d.txt' % self.fold)
            if not os.path.exists(train_fold_file):
                raise FileNotFoundError(f"Train fold file not found: {train_fold_file}")

            # Build class_to_image_names mapping
            class_to_image_names = {}
            with open(train_fold_file, 'r') as f:
                for line in f:
                    if '__' in line:
                        img_name, class_id_str = line.strip().split('__')
                        class_id = int(class_id_str) - 1  # Adjust class ID to 0-based index
                        class_to_image_names.setdefault(class_id, []).append(img_name)

            print("Loading training image features...")
            # Added progress bar for the outer loop
            for class_id, image_names in tqdm(class_to_image_names.items(), desc="Classes"):
                features_list = []
                image_names_list = []
                # Inner loop progress bar remains unchanged
                for img_name in tqdm(image_names, desc=f"Class {class_id}"):
                    img_path = os.path.join(self.img_path, img_name + '.jpg')
                    feature_path = os.path.join(self.cache_dir, img_name + '.npy')
                    if os.path.exists(feature_path):
                        feature = np.load(feature_path)
                    else:
                        img = Image.open(img_path).convert("RGB")
                        img_resized = img.resize((self.feature_model_input_size, self.feature_model_input_size))
                        img_resized_np = np.array(img_resized) / 255.0
                        if self.what_makes:
                            feature = extract_image_feature_what(self.feature_model, img_resized_np, self.device)
                        else:
                            feature = extract_image_feature(self.feature_model, img_resized_np, self.device)
                        np.save(feature_path, feature)
                    features_list.append(feature)
                    image_names_list.append(img_name)
                if features_list:
                    self.class_to_train_features[class_id] = np.vstack(features_list)  # Shape (N, feature_dim)
                    self.class_to_train_image_paths[class_id] = image_names_list  # Shape (N,)  

    def __len__(self):
        return  len(self.img_metadata) # 1000

    def create_grid_from_images(self, support_img, support_mask, query_img, query_mask, arrangement):
        img_height = support_img.shape[1]
        img_width = support_img.shape[2]
        padding = self.padding
        canvas = torch.ones((support_img.shape[0],
                             2 * img_height + 2 * padding,
                             2 * img_width + 2 * padding))
        images_dict = {
            'support_img': support_img,
            'support_mask': support_mask,
            'query_img': query_img,
            'query_mask': query_mask
        }
        arrangement_mapping = {
            'A1': ['support_img', 'query_img', 'support_mask', 'query_mask'],
            'A2': ['support_mask', 'query_mask', 'support_img', 'query_img'],
            'A3': ['query_mask', 'support_mask', 'query_img', 'support_img'],
            'A4': ['query_img', 'support_img', 'query_mask', 'support_mask'],
            'A5': ['support_mask', 'support_img', 'query_mask', 'query_img'],
            'A6': ['query_mask', 'query_img', 'support_mask', 'support_img'],
            'A7': ['query_img', 'query_mask', 'support_img', 'support_mask'],
            'A8': ['support_img', 'support_mask', 'query_img', 'query_mask']
        }

        mapping = arrangement_mapping[arrangement]
        imgs = [images_dict[name] for name in mapping]

        # Place images in canvas
        # Top-left
        canvas[:, :img_height, :img_width] = imgs[0]
        # Bottom-left
        canvas[:, -img_height:, :img_width] = imgs[1]
        # Top-right
        canvas[:, :img_height, -img_width:] = imgs[2]
        # Bottom-right
        canvas[:, -img_height:, -img_width:] = imgs[3]

        return canvas
    
    def create_grid_from_images_inpainting(self, support_img_inpainting, support_img, query_img_inpainting, query_img, arrangement):
        img_height = support_img.shape[1]
        img_width = support_img.shape[2]
        padding = self.padding
        canvas = torch.ones((support_img.shape[0],
                            2 * img_height + 2 * padding,
                            2 * img_width + 2 * padding))
        images_dict = {
            'support_img': support_img_inpainting,
            'support_mask': support_img,
            'query_img': query_img_inpainting,
            'query_mask': query_img
        }
        arrangement_mapping = {
            'A1': ['support_img', 'query_img', 'support_mask', 'query_mask'],
            'A2': ['support_mask', 'query_mask', 'support_img', 'query_img'],
            'A3': ['query_mask', 'support_mask', 'query_img', 'support_img'],
            'A4': ['query_img', 'support_img', 'query_mask', 'support_mask'],
            'A5': ['support_mask', 'support_img', 'query_mask', 'query_img'],
            'A6': ['query_mask', 'query_img', 'support_mask', 'support_img'],
            'A7': ['query_img', 'query_mask', 'support_img', 'support_mask'],
            'A8': ['support_img', 'support_mask', 'query_img', 'query_mask']
        }

        mapping = arrangement_mapping[arrangement]
        imgs = [images_dict[name] for name in mapping]

        # Place images in canvas
        # Top-left
        canvas[:, :img_height, :img_width] = imgs[0]
        # Bottom-left
        canvas[:, -img_height:, :img_width] = imgs[1]
        # Top-right
        canvas[:, :img_height, -img_width:] = imgs[2]
        # Bottom-right
        canvas[:, -img_height:, -img_width:] = imgs[3]

        return canvas

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_name, class_sample_query, class_sample_support = self.sample_episode(idx)
        query_img, query_cmask, support_img, support_cmask, org_qry_imsize = self.load_frame(query_name, support_name)

        if self.image_transform:
            query_img = self.image_transform(query_img)
        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask, class_sample_query, purple=self.purple)
        if self.mask_transform:
            query_mask = self.mask_transform(query_mask)

        if self.image_transform:
            support_img = self.image_transform(support_img)
        support_mask, support_ignore_idx = self.extract_ignore_idx(support_cmask, class_sample_support, purple=self.purple)
        if self.mask_transform:
            support_mask = self.mask_transform(support_mask)
            
        # Apply edge detection if required
        if self.edge_detection:
            query_mask = apply_edge_detection(query_mask)
            support_mask = apply_edge_detection(support_mask)

        if self.inpainting:
            # Process support_img to create support_img_inpainting
            C_s, H_s, W_s = support_img.shape
            x_start_s = W_s // 2 - W_s // 8
            x_end_s = W_s // 2 + W_s // 8
            y_start_s = H_s // 2 - H_s // 8
            y_end_s = H_s // 2 + H_s // 8
            support_img_inpainting = support_img.clone()
            support_img_inpainting[:, y_start_s:y_end_s, x_start_s:x_end_s] = 0

            # Process query_img to create query_img_inpainting
            C_q, H_q, W_q = query_img.shape
            x_start_q = W_q // 2 - W_q // 8
            x_end_q = W_q // 2 + W_q // 8
            y_start_q = H_q // 2 - H_q // 8
            y_end_q = H_q // 2 + H_q // 8
            query_img_inpainting = query_img.clone()
            query_img_inpainting[:, y_start_q:y_end_q, x_start_q:x_end_q] = 0

            if self.ensemble:
                grids = []
                arrangements = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
                for arr in arrangements:
                    grid = self.create_grid_from_images_inpainting(support_img_inpainting, support_img, query_img_inpainting, query_img, arrangement=arr)
                    grids.append(grid)
                grid = tuple(grids)
            else:
                grid = self.create_grid_from_images_inpainting(support_img_inpainting, support_img, query_img_inpainting, query_img, arrangement='A1')
        else:
            if self.ensemble:
                grids = []
                arrangements = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
                for arr in arrangements:
                    grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, arrangement=arr)
                    grids.append(grid)
                grid = tuple(grids)
            else:
                grid = self.create_grid_from_images(support_img, support_mask, query_img, query_mask, arrangement='A1')

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_name,
            'query_ignore_idx': query_ignore_idx,
            'org_query_imsize': org_qry_imsize,
            'support_img': support_img,
            'support_mask': support_mask,
            'support_name': support_name,
            'support_ignore_idx': support_ignore_idx,
            'class_id': torch.tensor(class_sample_query),
            'grid': grid
        }

        return batch

    def extract_ignore_idx(self, mask, class_id, purple):
        mask = np.array(mask)
        boundary = np.floor(mask / 255.)
        if not purple:
            mask[mask != class_id + 1] = 0
            mask[mask == class_id + 1] = 255
            return Image.fromarray(mask), boundary
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x,y] != class_id + 1:
                    color_mask[x, y] = np.array(PURPLE)
                else:
                    color_mask[x, y] = np.array(YELLOW)
        return Image.fromarray(color_mask), boundary


    def load_frame(self, query_name, support_name):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_img = self.read_img(support_name)
        support_mask = self.read_mask(support_name)
        org_qry_imsize = query_img.size

        return query_img, query_mask, support_img, support_mask, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = Image.open(os.path.join(self.ann_path, img_name) + '.png')
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        """Returns the index of the query, support, and class."""
        query_name, class_sample = self.img_metadata[idx]
        if self.prompt_search:
            # Use prompt-search to select support image
            class_id = class_sample  # Adjust class ID to 0-based index
            if class_id in self.class_to_train_features and self.class_to_train_features[class_id].size > 0:
                # Get features and image names
                train_features = self.class_to_train_features[class_id]  # Shape (N, feature_dim)
                train_image_names = self.class_to_train_image_paths[class_id]  # List of image names

                # Extract feature of query image
                query_img_path = os.path.join(self.img_path, query_name + '.jpg')
                query_img = Image.open(query_img_path).convert("RGB")
                img_resized = query_img.resize((self.feature_model_input_size, self.feature_model_input_size))
                img_resized_np = np.array(img_resized) / 255.0

                if self.what_makes:
                    query_feature = extract_image_feature_what(self.feature_model, img_resized_np, self.device)
                else:
                    query_feature = extract_image_feature(self.feature_model, img_resized_np, self.device)# (1, feature_dim)

                # Compute similarities
                similarities = np.dot(query_feature, train_features.T)  # (1, N)
                idx_max = np.argmax(similarities, axis=1)[0]
                support_name = train_image_names[idx_max]
                support_class = class_sample
            else:
                # Fallback to random sampling if no training images are found
                print(f"No training images found for class {class_sample}, using random support image.")
                support_class = class_sample
                support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
                while query_name == support_name:
                    support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
        else:
            if not self.random:
                support_class = class_sample
            else:
                support_class = np.random.choice(
                    [k for k in self.img_metadata_classwise.keys() if self.img_metadata_classwise[k]], 1, replace=False)[0]
            while True:  # Keep sampling support set if query == support
                support_name = np.random.choice(self.img_metadata_classwise[support_class], 1, replace=False)[0]
                if query_name != support_name:
                    break
        return query_name, support_name, class_sample, support_class

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
        return class_ids_val

    def build_img_metadata(self):
        datapth = self.datapth
        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(datapth, 'splits/pascal/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        img_metadata = read_metadata('val', self.fold)
       
        print('Total (val) images are : %d' % len(img_metadata))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise