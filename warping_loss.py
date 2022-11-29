import cv2
import numpy as np
from scipy import ndimage
import imageio
from utilities import RobustCrossEntropyLoss, SoftDiceLoss, softmax_helper, DC_and_CE_loss
import torch
import cc3d

def decide_simple_point_2D(gt, x, y):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2]

    ## check local topology
    
    number_fore, _ = cv2.connectedComponents(patch, 4)
    number_back, _ = cv2.connectedComponents(1-patch, 8)

    label = (number_fore-1) * (number_back-1)

    # try:
    #     patch[1][1] = 0
    #     number_fore, label_fore = cv2.connectedComponents(patch, 4)
    #     label_fore_4 = np.unique([label_fore[0,1], label_fore[1,0], label_fore[2,1], label_fore[1,2]])

    #     patch_reverse = 1 - patch
    #     patch_reverse[1][1] = 0
    #     number_back, label_back = cv2.connectedComponents(patch_reverse, 8)
    #     label_back_8 = np.unique([label_back[0,0], label_back[0,1], label_back[0,2], label_back[1,0], label_back[1,2], label_back[2,0], label_back[2,1], label_back[2,2]])
    #     label = len(np.nonzero(label_fore_4)[0]) * len(np.nonzero(label_back_8)[0])
    # except:
    #         label = 0
    #         pass

    ## flip the simple point
    if (label == 1):
        gt[x,y] = 1 - gt[x,y]

    return gt

def decide_simple_point_3D(gt, x, y, z):
    """
    decide simple points
    """

    ## extract local patch
    patch = gt[x-1:x+2, y-1:y+2, z-1:z+2]

    ## check local topology
    if patch.shape[0] != 0 and patch.shape[1] != 0 and patch.shape[2] != 0:
        try:
            _, number_fore = cc3d.connected_components(patch, 6, return_N = True)
            _, number_back = cc3d.connected_components(1-patch, 26, return_N = True)
        except:
            number_fore = 0
            number_back = 0
            pass
        label = number_fore * number_back

        ## flip the simple point
        if (label == 1):
            gt[x,y,z] = 1 - gt[x,y,z]

    return gt

def update_simple_point(distance, gt):
    non_zero = np.nonzero(distance)
    # indice = np.argsort(-distance, axis=None) 
    indice = np.unravel_index(np.argsort(-distance, axis=None), distance.shape)

    if len(gt.shape) == 2:
        for i in range(len(non_zero[0])):
            # check the index is correct
            # diff_distance[indices[len(non_zero_list[0]) - i - 1]//gt.shape[1], indices[len(non_zero_list[0]) - i - 1]%gt.shape[1]]
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_2D(gt, x, y)
    else:
        for i in range(len(non_zero[0])):
            # check the index is correct
            # diff_distance[indices[len(non_zero_list[0]) - i - 1]//gt.shape[1], indices[len(non_zero_list[0]) - i - 1]%gt.shape[1]]
            x = indice[0][len(non_zero[0]) - i - 1]
            y = indice[1][len(non_zero[0]) - i - 1]
            z = indice[2][len(non_zero[0]) - i - 1]

            gt = decide_simple_point_3D(gt, x, y, z)
    return gt


def warping_loss(y_pred, y_gt):
    """
    Calculate the warping loss of the predicted image and ground truth image 
    Args:
        pre:   The likelihood pytorch tensor for neural networks.
        gt:   The groundtruth of pytorch tensor.
    Returns:
        warping_loss:   The warping loss value (tensor)
    """
    ## compute false positive and false negative

    # if ()
    loss = 0
    soft_dice_args = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
    train_loss_func = DC_and_CE_loss(soft_dice_args, {})
    sdl = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_args)
    ce_loss = RobustCrossEntropyLoss()

    if (len(y_pred.shape) == 4):
        B, C, H, W = y_pred.shape

        pre = softmax_helper(y_pred)
        import pdb; pdb.set_trace()
        pre = torch.argmax(pre, dim=1)
        y_gt = torch.unsqueeze(y_gt[:,0,:,:], dim =1)
        gt = torch.squeeze(y_gt, dim=1)
        
        pre = pre.cpu().detach().numpy().astype('uint8')
        gt = gt.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy()
        gt_copy = gt.copy()

        critical_points = np.zeros((B,H,W))
        for i in range(B):
            false_positive = ((pre_copy[i,:,:] - gt_copy[i,:,:]) == 1).astype(int)
            false_negative = ((gt_copy[i,:,:] - pre_copy[i,:,:]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance_gt = ndimage.distance_transform_edt(gt_copy[i,:,:]) * false_negative  # shrink gt while keep connected
            false_positive_distance_gt = ndimage.distance_transform_edt(1 - gt_copy[i,:,:]) * false_positive  # grow gt while keep unconnected
            gt_warp = update_simple_point(false_negative_distance_gt, gt_copy[i,:,:])
            gt_warp = update_simple_point(false_positive_distance_gt, gt_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(pre_copy[i,:,:]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(1-pre_copy[i,:,:]) * false_negative # grow gt while keep unconnected
            pre_warp = update_simple_point(false_positive_distance_pre, pre_copy[i,:,:])
            pre_warp = update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i,:,:] = np.logical_or(np.not_equal(pre[i,:,:], gt_warp), np.not_equal(gt[i,:,:], pre_warp)).astype(int)
    else:
        B, C, H, W, Z = y_pred.shape

        pre = softmax_helper(y_pred)
        pre = torch.argmax(pre, dim=1)
        y_gt = torch.unsqueeze(y_gt[:,0,:,:,:], dim =1)
        gt = torch.squeeze(y_gt, dim=1)
        
        pre = pre.cpu().detach().numpy().astype('uint8')
        gt = gt.cpu().detach().numpy().astype('uint8')

        pre_copy = pre.copy()
        gt_copy = gt.copy()

        critical_points = np.zeros((B,H,W,Z))
        for i in range(B):
            false_positive = ((pre_copy[i,:,:,:] - gt_copy[i,:,:,:]) == 1).astype(int)
            false_negative = ((gt_copy[i,:,:,:] - pre_copy[i,:,:,:]) == 1).astype(int)

            ## Use distance transform to determine the flipping order
            false_negative_distance = ndimage.distance_transform_edt(gt_copy[i,:,:,:]) * false_negative
            false_positive_distance = ndimage.distance_transform_edt(1 - gt_copy[i,:,:,:]) * false_positive
            gt_warp = update_simple_point(false_negative_distance, gt_copy[i,:,:,:])
            gt_warp = update_simple_point(false_positive_distance, gt_warp)

            false_positive_distance_pre = ndimage.distance_transform_edt(pre_copy[i,:,:,:]) * false_positive  # shrink pre while keep connected
            false_negative_distance_pre = ndimage.distance_transform_edt(1-pre_copy[i,:,:,:]) * false_negative # grow gt while keep unconnected
            pre_warp = update_simple_point(false_positive_distance_pre, pre_copy[i,:,:,:])
            pre_warp = update_simple_point(false_negative_distance_pre, pre_warp)

            critical_points[i,:,:] = np.logical_or(np.not_equal(pre[i,:,:,:], gt_warp), np.not_equal(gt[i,:,:,:], pre_warp)).astype(int)

    loss = ce_loss(y_pred * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda(), y_gt * torch.unsqueeze(torch.from_numpy(critical_points), dim=1).cuda()) * len(np.nonzero(critical_points)[0])
    return loss
