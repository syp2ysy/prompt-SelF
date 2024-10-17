import os
from tqdm import trange
import pascal_dataloader
from evaluate_detection.box_ops import to_rectangle
from evaluate_detection.canvas_ds import CanvasDataset
import torchvision
from mae_utils import *
import argparse
from pathlib import Path
from segmentation_utils import *
import cv2
import numpy as np
import torch
from prompt_self_utils import  get_query_mask_position, get_query_mask_coordinates, extract_predicted_query_mask

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training, Evaluate PASCAL', add_help=False)
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--output_dir', default='../output/pascal_prompt_self/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--base_dir', default='../Datasets', help='pascal base dir')
    parser.add_argument('--seed', default=321, type=int)
    parser.add_argument('--t', default=[0, 0, 0], type=float, nargs='+')
    parser.add_argument('--task', default='segmentation', choices=['segmentation', 'detection'])
    parser.add_argument('--ckpt', default="../model_pth/3400.pth", help='model checkpoint')
    parser.add_argument('--dataset_type', default='pascal',
                        choices=['pascal', 'pascal_det'])
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--purple', default=0, type=int)
    parser.add_argument('--flip', default=0, type=int)
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of canvases')
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--what_makes', action='store_true')
    parser.add_argument('--edge_detection', action='store_true')
    parser.add_argument('--inpainting', action='store_true')
    return parser

def _generate_result_for_canvas(args, model, canvas, arrangement):
    """canvas 已经在正确的范围内。"""
    ids_shuffle, len_keep = generate_mask_for_evaluation(arrangement)
    _, im_paste, _ = generate_image(
        canvas.unsqueeze(0).to(args.device),
        model,
        ids_shuffle.to(args.device),
        len_keep,
        device=args.device
    )
    canvas = torch.einsum('chw->hwc', canvas)
    canvas = torch.clip(
        (canvas.cpu().detach() * imagenet_std + imagenet_mean) * 255,
        0, 255
    ).int().numpy()
    assert canvas.shape == im_paste.shape, (canvas.shape, im_paste.shape)
    return np.uint8(canvas), np.uint8(im_paste)


def evaluate(args):
    log_dir = os.path.join(args.output_dir, 'logs')
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(log_dir, 'log.txt'), 'w') as log:
        log.write(str(args) + '\n')
        
    padding = 1
    args.padding = padding  # Add padding to args to use in other functions
    image_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), interpolation=3),
         torchvision.transforms.ToTensor()])
    mask_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((224 // 2 - padding, 224 // 2 - padding), interpolation=3),
         torchvision.transforms.ToTensor()])
    
    # Define resize size
    resize_size = (224 // 2 - padding, 224 // 2 - padding)
    
    ds = {
        'pascal': pascal_dataloader.DatasetPASCAL,
        'pascal_det': CanvasDataset
    }[args.dataset_type](args.base_dir, fold=args.fold, image_transform=image_transform, mask_transform=mask_transform,
                         flipped_order=args.flip, purple=args.purple, ensemble=args.ensemble, prompt_search=args.search,
                         what_makes=args.what_makes, edge_detection=args.edge_detection, inpainting=args.inpainting)
    
    model = prepare_model(args.ckpt, arch=args.model)
    _ = model.to(args.device)
    
    # Initialize evaluation metrics
    eval_dict = {'iou': 0, 'color_blind_iou': 0, 'accuracy': 0}
    
    for idx in trange(len(ds)):

        data_sample = ds[idx]
        canvas = data_sample['grid']
        # Preprocess canvas
        if args.dataset_type != 'pascal_det':
            if isinstance(canvas, tuple):
                # For ensemble case, ensure each canvas is preprocessed the same way
                canvas = tuple(((c - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]) for c in canvas)
            else:
                canvas = (canvas - imagenet_mean[:, None, None]) / imagenet_std[:, None, None]
        
        if isinstance(canvas, tuple):  # Ensemble case
            generated_query_masks = []
            arrangements = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
            for i, (c, arrangement) in enumerate(zip(canvas, arrangements)):
                # Model prediction
                original_image, generated_result = _generate_result_for_canvas(args, model, c, arrangement)
                # Save images
                if args.output_dir:
                    Image.fromarray(np.uint8(original_image)).save(
                        os.path.join(args.output_dir, f'original_{str(idx).zfill(3)}_A{i+1}.png'))
                    Image.fromarray(np.uint8(generated_result)).save(
                        os.path.join(args.output_dir, f'generated_{str(idx).zfill(3)}_A{i+1}.png'))
                # Extract predicted query_mask
                query_mask = extract_predicted_query_mask(generated_result, arrangement, args)
                generated_query_masks.append(query_mask)
                # Save original_image for later
                if i == 0:
                    target = original_image  # For metric calculation
            # Average all query_masks
            generated_query_masks = np.stack(generated_query_masks, axis=0)
            averaged_query_mask = np.mean(generated_query_masks, axis=0)
            # Apply threshold
            if args.inpainting:
                # 如果 inpainting 为 True，直接取平均值，并对每个像素值取整
                averaged_query_mask = np.rint(averaged_query_mask).astype(np.uint8)
            elif args.edge_detection:
                                # 对每个通道应用阈值，然后检查任意通道是否大于阈值
                averaged_query_mask = (averaged_query_mask > 102).astype(np.uint8)

                # 如果任何一个通道满足条件，则将该像素设置为白色
                averaged_query_mask = (averaged_query_mask.sum(axis=-1) > 0).astype(np.uint8) * 255  # Apply threshold
                # averaged_query_mask = round_image(averaged_query_mask, [WHITE, BLACK])
            else:
                # Apply threshold if inpainting is False
                averaged_query_mask = (averaged_query_mask > 127).astype(np.uint8) * 255  # Apply threshold

            # === 新增功能开始 ===
            # 创建一个新的图像，按照 A1 的排列方式
            averaged_image = target.copy()
            position = get_query_mask_position('A1')  # Use 'A1' to ensure consistent cropping
            y_start, y_end, x_start, x_end = get_query_mask_coordinates(position, averaged_image.shape, args)
            # 调整 averaged_query_mask 的尺寸
            averaged_query_mask_resized = cv2.resize(averaged_query_mask, (x_end - x_start, y_end - y_start))
            # 确保有三个通道
            if averaged_query_mask_resized.ndim == 2:
                averaged_query_mask_resized = np.repeat(averaged_query_mask_resized[:, :, np.newaxis], 3, axis=2)
            # 将 averaged_query_mask 放置到图像的右下角
            averaged_image[y_start:y_end, x_start:x_end, :] = averaged_query_mask_resized
            # # 应用 round_image 函数
            # if args.purple:
            #     averaged_image = round_image(averaged_image, [YELLOW, PURPLE], t=args.t)
            # else:
            #     averaged_image = round_image(averaged_image, [WHITE, BLACK], t=args.t)
            # 保存新的图像
            if args.output_dir:
                Image.fromarray(np.uint8(averaged_image)).save(
                    os.path.join(args.output_dir, f'generated_{str(idx).zfill(3)}_ensemble.png'))
            # === 新增功能结束 ===
            
            # Place the averaged_query_mask back into the image
            ours = target.copy()
            position = get_query_mask_position('A1')  # Use 'A1' to ensure consistent cropping
            y_start, y_end, x_start, x_end = get_query_mask_coordinates(position, ours.shape, args)
            # Resize averaged_query_mask to fit
            averaged_query_mask_resized = cv2.resize(averaged_query_mask, (x_end - x_start, y_end - y_start))
            # Ensure three channels
            if averaged_query_mask_resized.ndim == 2:
                averaged_query_mask_resized = np.repeat(averaged_query_mask_resized[:, :, np.newaxis], 3, axis=2)
            # Place into ours
            ours[y_start:y_end, x_start:x_end, :] = averaged_query_mask_resized
            # Apply round_image function
            if args.purple:
                # ours = round_image(ours, [YELLOW, PURPLE], t=args.t)
                # target = round_image(target, [YELLOW, PURPLE])
                fg_color = YELLOW
                bg_color = PURPLE
            else:
                # ours = round_image(ours, [WHITE, BLACK], t=args.t)
                # target = round_image(target, [WHITE, BLACK])
                fg_color = WHITE
                bg_color = BLACK
            do_crop = True
        else:  # Non-ensemble case
            # Model prediction
            original_image, generated_result = _generate_result_for_canvas(args, model, canvas, 'A1')
            if args.output_dir:
                Image.fromarray(np.uint8(original_image)).save(
                    os.path.join(args.output_dir, f'original_{idx}.png'))
                Image.fromarray(np.uint8(generated_result)).save(
                    os.path.join(args.output_dir, f'generated_{idx}.png'))
            # Extract predicted query_mask
            arrangement = 'A1'  # Default to 'A1'
            query_mask = extract_predicted_query_mask(generated_result, arrangement, args)
            # Place query_mask back into the image
            ours = original_image.copy()
            position = get_query_mask_position('A1')
            y_start, y_end, x_start, x_end = get_query_mask_coordinates(position, ours.shape, args)
            # Resize query_mask if necessary
            if query_mask.shape[0] != (y_end - y_start) or query_mask.shape[1] != (x_end - x_start):
                query_mask = cv2.resize(query_mask, (x_end - x_start, y_end - y_start))
            # Ensure three channels
            if query_mask.ndim == 2:
                query_mask = np.repeat(query_mask[:, :, np.newaxis], 3, axis=2)
            # Place into ours
            ours[y_start:y_end, x_start:x_end, :] = query_mask
            # Apply round_image function
            if args.purple:
                ours = round_image(ours, [YELLOW, PURPLE], t=args.t)
                target = round_image(original_image, [YELLOW, PURPLE])
                fg_color = YELLOW
                bg_color = PURPLE
            else:
                ours = round_image(ours, [WHITE, BLACK], t=args.t)
                target = round_image(original_image, [WHITE, BLACK])
                fg_color = WHITE
                bg_color = BLACK
            do_crop = True
        
        # **Ensure data types are consistent before visualization**
        if isinstance(ours, torch.Tensor):
            ours = ours.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        ours = ours.astype(np.uint8)
        target = target.astype(np.uint8)
        
        # Visualization (after ensuring ours and target are NumPy arrays)
        # Draw red and blue rectangles
        if args.output_dir:
            vis_image = ours.copy()
            # Draw red rectangle around query_mask
            position = get_query_mask_position('A1')
            y_start, y_end, x_start, x_end = get_query_mask_coordinates(position, vis_image.shape, args)
            cv2.rectangle(vis_image, (x_start, y_start), (x_end - 1, y_end - 1), (0, 0, 255), 2)  # Red rectangle
            # Draw blue rectangles around prompt_img and prompt_mask
            # According to 'A1' arrangement, prompt_img and prompt_mask are at top-left and top-right
            img_height = (vis_image.shape[0] - 2 * padding) // 2
            img_width = (vis_image.shape[1] - 2 * padding) // 2
            # Top-left
            cv2.rectangle(vis_image, (0, 0), (img_width - 1, img_height - 1), (255, 0, 0), 2)  # Blue rectangle
            # Top-right
            cv2.rectangle(vis_image, (img_width + 2 * padding, 0), (vis_image.shape[1] - 1, img_height - 1), (255, 0, 0), 2)  # Blue rectangle
            # Save visualization
            Image.fromarray(vis_image).save(os.path.join(args.output_dir, f'visualization_{idx}.png'))
        
        # Ensure 'ours' and 'target' have the same shape
        if ours.shape != target.shape:
            print("Error: 'ours' and 'target' have different shapes.")
            print(f"'ours' shape: {ours.shape}, 'target' shape: {target.shape}")
            continue  # Skip this sample
        
        # Compute metrics
        current_metric = calculate_metric(args, target, ours, fg_color=fg_color, bg_color=bg_color, do_crop=do_crop)
        
        # Check for NaN in IoU
        if np.isnan(current_metric['iou']):
            print(f"IoU is NaN at index {idx}")
            # Add debugging code here if needed
            
        # Log metrics
        with open(os.path.join(log_dir, 'log.txt'), 'a') as log:
            log.write(str(idx) + '\t' + str(current_metric) + '\n')
        for i, j in current_metric.items():
            if not np.isnan(j):
                eval_dict[i] += (j / len(ds))
    
    with open(os.path.join(log_dir, 'log.txt'), 'a') as log:
        log.write('all\t' + str(eval_dict) + '\n')
    
    # Print final evaluation metrics
    print(f'Final evaluation metrics: {eval_dict}')


if __name__ == '__main__':
    args = get_args()

    args = args.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
