import torch
import torch.nn.functional as F
import numpy as np

def get_query_mask_position(arrangement):
    mapping = {
        'A1': 'bottom-right',
        'A2': 'bottom-left',
        'A3': 'top-left',
        'A4': 'top-right',
        'A5': 'top-right',
        'A6': 'top-left',
        'A7': 'bottom-left',
        'A8': 'bottom-right'
    }
    return mapping[arrangement]

def get_query_mask_coordinates(position, canvas_shape, args):
    padding = args.padding if hasattr(args, 'padding') else 1
    img_height = (canvas_shape[0] - 2 * padding) // 2
    img_width = (canvas_shape[1] - 2 * padding) // 2

    y_positions = {
        'top-left': (0, img_height),
        'top-right': (0, img_height),
        'bottom-left': (img_height + 2 * padding, canvas_shape[0]),
        'bottom-right': (img_height + 2 * padding, canvas_shape[0])
    }
    x_positions = {
        'top-left': (0, img_width),
        'bottom-left': (0, img_width),
        'top-right': (img_width + 2 * padding, canvas_shape[1]),
        'bottom-right': (img_width + 2 * padding, canvas_shape[1])
    }
    y_start, y_end = y_positions[position]
    x_start, x_end = x_positions[position]
    return y_start, y_end, x_start, x_end


def extract_predicted_query_mask(im_paste, arrangement, args):
    position = get_query_mask_position(arrangement)
    y_start, y_end, x_start, x_end = get_query_mask_coordinates(position, im_paste.shape[:2], args)
    query_mask = im_paste[y_start:y_end, x_start:x_end, :]
    return query_mask

@torch.no_grad()
def extract_image_feature(model, img_np, device):
    """
    Extract feature from an image using the feature extraction model.

    Args:
        model (torch.nn.Module): Feature extraction model.
        img_np (np.ndarray): Input image array.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Extracted and normalized feature vector.
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    img = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    output = model.forward_features(img_tensor)
    feature = F.normalize(output[:, 0], dim=1).cpu().numpy()  # Using [CLS] token
    return feature  # 

@torch.no_grad()
def extract_image_feature_what(model, img_np, device):
    """
    Extract feature from an image using the feature extraction model.

    Args:
        model (torch.nn.Module): Feature extraction model.
        img_np (np.ndarray): Input image array.
        device (torch.device): Computation device.

    Returns:
        np.ndarray: Extracted and normalized feature vector.
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    img = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    output = model.forward_features(img_tensor)
    output = model.forward_head(output,pre_logits=True)
    feature = output.cpu().numpy()  # Using [CLS] token
    return feature  # 