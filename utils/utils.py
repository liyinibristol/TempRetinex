import os
import numpy as np
import torch
import shutil
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import utils.flow_viz as flow_viz
import torch.nn.functional as F
from scipy import interpolate


def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = torch.nn.functional.conv2d(img, filter1, stride=2, groups=c)
    output2 = torch.nn.functional.conv2d(img, filter2, stride=2, groups=c)
    return output1, output2


def gauss_cdf(x):
    return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.))))


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = torch.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1, ).cuda()
    # kern1d=torch.diff(torch.erf(x/math.sqrt(2.0)))/2.0
    kern1d = torch.diff(gauss_cdf(x))
    kernel_raw = torch.sqrt(torch.outer(kern1d, kern1d))
    kernel = kernel_raw / torch.sum(kernel_raw)
    # out_filter=kernel.unsqueeze(2).unsqueeze(3).repeat(1,1,channels,1)
    out_filter = kernel.view(1, 1, kernlen, kernlen)
    out_filter = out_filter.repeat(channels, 1, 1, 1)
    return out_filter


class LocalMean(torch.nn.Module):
    def __init__(self, patch_size=5):
        super(LocalMean, self).__init__()
        self.patch_size = patch_size
        self.padding = self.patch_size // 2

    def forward(self, image):
        image = torch.nn.functional.pad(image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        patches = image.unfold(2, self.patch_size, 1).unfold(3, self.patch_size, 1)
        return patches.mean(dim=(4, 5))


def blur(x):
    device = x.device
    kernel_size = 21
    padding = kernel_size // 2
    kernel_var = gauss_kernel(kernel_size, 1, x.size(1)).to(device)
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), mode='reflect')
    return torch.nn.functional.conv2d(x_padded, kernel_var, padding=0, groups=x.size(1))


def padr_tensor(img):
    pad = 2
    pad_mod = torch.nn.ConstantPad2d(pad, 0)
    img_pad = pad_mod(img)
    return img_pad


def calculate_local_variance(train_noisy):
    b, c, w, h = train_noisy.shape
    avg_pool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
    noisy_avg = avg_pool(train_noisy)
    noisy_avg_pad = padr_tensor(noisy_avg)
    train_noisy = padr_tensor(train_noisy)
    unfolded_noisy_avg = noisy_avg_pad.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy = train_noisy.unfold(2, 5, 1).unfold(3, 5, 1)
    unfolded_noisy_avg = unfolded_noisy_avg.reshape(unfolded_noisy_avg.shape[0], -1, 5, 5)
    unfolded_noisy = unfolded_noisy.reshape(unfolded_noisy.shape[0], -1, 5, 5)
    noisy_diff_squared = (unfolded_noisy - unfolded_noisy_avg) ** 2
    noisy_var = torch.mean(noisy_diff_squared, dim=(2, 3))
    noisy_var = noisy_var.view(b, c, w, h)
    return noisy_var


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def save_all(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def show_pic(pic, name, path):
    pic_num = len(pic)
    for i in range(pic_num):
        img = pic[i]
        image_numpy = img[0].cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
            im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im)
        elif image_numpy.shape[0] == 1:
            im = Image.fromarray(np.clip(image_numpy[0] * 255.0, 0, 255.0).astype('uint8'))
            img_name = name[i]
            plt.subplot(5, 6, i + 1)
            plt.xlabel(str(img_name))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(im, plt.cm.gray)
    plt.savefig(path)


def sequential_judgment(img_path, last_img_path, is_reverse=False):
    is_new_seq = False

    assert os.path.exists(img_path)
    if not os.path.exists(last_img_path): return True
    img_dir, img_name = os.path.split(img_path)
    last_img_dir, last_img_name = os.path.split(last_img_path)
    img_idx = int(os.path.splitext(img_name)[0].split('.')[-1])
    last_img_idx = int(os.path.splitext(last_img_name)[0].split('.')[-1])

    if img_dir != last_img_dir:
        is_new_seq = True
    elif img_idx != (last_img_idx + 1) and not is_reverse:
        is_new_seq = True
    elif img_idx != (last_img_idx - 1) and is_reverse:
        is_new_seq = True
    # print("last img idx: ", last_img_idx, " img idx: ", img_idx, "is_new_seq: ", is_new_seq)
    return is_new_seq


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey(5)


def viz_flo(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)

    return flo


def warp_img(flow, img1, img2, dst_size=[1080, 1920]):
    # flow: img1 -> img2

    h, w = flow.shape[:2]
    h_dst, w_dst = dst_size[:2]
    h_scale = float(h_dst / h)
    w_scale = float(w_dst / w)

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # flow[..., 0] 是水平位移，flow[..., 1] 是垂直位移
    # 新的坐标位置 = 原始坐标 + 位移
    map_x = (grid_x + flow[..., 0]) * h_scale
    map_y = (grid_y + flow[..., 1]) * w_scale

    map_x = cv2.resize(map_x, [w_dst, h_dst])
    map_y = cv2.resize(map_y, [w_dst, h_dst])
    # 使用 remap 进行warp
    warped_image = cv2.remap(img1, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    overlap_image = cv2.addWeighted(warped_image, 0.5, img2, 0.5, 0)

    return warped_image, overlap_image


def warp_tensor(flow, img1, img2=None):
    B, C, H, W = flow.shape
    h_dst, w_dst = img1.shape[-2:]
    h_scale = float(h_dst) / float(H)
    w_scale = float(w_dst) / float(W)

    # Extend the grid and use optical flow
    grid_y, grid_x = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32))
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1).to(img1.device)  # [B, H, W]
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1).to(img1.device)  # [B, H, W]

    # 计算新的坐标
    map_x = (grid_x + flow[:, 0, :, :]) * h_scale
    map_y = (grid_y + flow[:, 1, :, :]) * w_scale
    map_x = F.interpolate(map_x.unsqueeze(1), (h_dst, w_dst), mode='bilinear')
    map_y = F.interpolate(map_y.unsqueeze(1), (h_dst, w_dst), mode='bilinear')

    # 使用 F.grid_sample 进行图像变换（注意：grid_sample 输入的坐标是 [-1, 1] 范围）
    grid = torch.stack((map_x / ((w_dst - 1) / 2) - 1, map_y / ((h_dst - 1) / 2) - 1), dim=-1)  # 归一化到 [-1, 1]
    grid = grid.squeeze(1)  # 形状变为 [B, H, W, 2]

    # 对图像进行采样
    warped_tensor = F.grid_sample(img1, grid, mode='bilinear', padding_mode='zeros')

    if img2 is not None:
        # 重叠图像
        overlap_tensor = 0.5 * warped_tensor + 0.5 * img2  # 计算重叠图像
    else:
        overlap_tensor = None

    return warped_tensor, overlap_tensor


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """

    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy

    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def cvt_ts2np(t):
    t = t.detach()
    n = t.squeeze().permute((1, 2, 0)).cpu().numpy()
    return cv2.cvtColor(n, cv2.COLOR_RGB2BGR)


def hist_match_rgb(source, template):
    """
    进行RGB图像的直方图匹配，使 source 图像的直方图接近 template 图像
    """
    matched = np.zeros_like(source)

    for i in range(3):  # 处理 R, G, B 三个通道
        src_hist, _ = np.histogram(source[:, :, i].flatten(), 256, [0, 256])
        tmpl_hist, _ = np.histogram(template[:, :, i].flatten(), 256, [0, 256])

        src_cdf = src_hist.cumsum()
        src_cdf = src_cdf / src_cdf[-1]  # 归一化

        tmpl_cdf = tmpl_hist.cumsum()
        tmpl_cdf = tmpl_cdf / tmpl_cdf[-1]  # 归一化

        mapping = np.interp(src_cdf, tmpl_cdf, np.arange(256))

        matched[:, :, i] = cv2.LUT(source[:, :, i], mapping.astype(np.uint8))

    return matched


def histogram_match_tensor(src, temp):
    B, C, H, W = src.shape
    hm = torch.zeros_like(src)
    for b in range(B):
        source = src[b]
        template = temp[b]

        # 确保输入是浮点类型
        source = source.float()
        template = template.float()

        # 使用 reshape 而不是 view，并确保张量是连续的
        source_flat = source.reshape(-1).contiguous()
        template_flat = template.reshape(-1).contiguous()

        # 获取源和模板的排序值
        source_values, source_idx = torch.sort(source_flat)
        template_values, _ = torch.sort(template_flat)

        # 计算分位数位置
        source_quantiles = torch.linspace(0, 1, steps=len(source_values), device=source.device)
        template_quantiles = torch.linspace(0, 1, steps=len(template_values), device=template.device)

        # 找到最接近的分位数位置
        indices = torch.searchsorted(template_quantiles, source_quantiles)

        # 处理边界情况
        indices = torch.clamp(indices, 0, len(template_values) - 1)

        # 执行映射
        matched_values = template_values[indices]

        # 恢复原始顺序
        matched = torch.zeros_like(source_values)
        matched[source_idx] = matched_values

        hm[b] = matched.reshape(source.shape)

    return hm


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    # img = cv2.resize(img, [640,360])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def get_cross_kernel(kernel_size=3):
    """
    生成十字形 Kernel
    Args:
        kernel_size: 必须是奇数（如3,5,7...）
    Returns:
        kernel: (1, 1, kernel_size, kernel_size)
    """
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    center = kernel_size // 2
    kernel = torch.zeros((kernel_size, kernel_size))

    # 十字形：中心行 + 中心列
    kernel[center, :] = 1  # 中心行
    kernel[:, center] = 1  # 中心列

    return kernel.view(1, 1, kernel_size, kernel_size)  # 调整为4D Tensor


def erosion_with_kernel(input_tensor, kernel_size=3):
    B, C, H, W = input_tensor.shape
    kernel = get_cross_kernel(kernel_size).to(input_tensor.device)
    pad_size = kernel_size // 2

    # 将输入 reshape 为 [B*C, 1, H, W]
    input_reshaped = input_tensor.view(B * C, 1, H, W)

    # Padding
    padded = F.pad(input_reshaped,
                   (pad_size, pad_size, pad_size, pad_size),
                   mode='constant', value=1)  # [B*C, 1, H+pad*2, W+pad*2]

    # Unfold [B*C, K*K, H*W]
    unfolded = F.unfold(padded, kernel_size)  # 分组卷积等效

    # 仅保留 Kernel 中1的位置的值
    kernel_mask = kernel.view(1, -1).bool()  # (1, K*K)
    selected_values = unfolded[:, kernel_mask.squeeze(0)]  # (1, num_ones, H*W)

    # 取最小值
    eroded = selected_values.min(dim=1)[0]  # [B*C, H*W]
    eroded_img = eroded.view(B * C, 1, H, W)  # [B*C, 1, H, W]

    # 恢复原始形状
    eroded_tensor = eroded_img.view(B, C, H, W)
    return eroded_tensor


def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    # img: [B, C, H, W]
    # sample_coords: [B, 2, H, W] in image scale
    if sample_coords.size(1) != 2:  # [B, H, W, 2]
        sample_coords = sample_coords.permute(0, 3, 1, 2)

    b, _, h, w = sample_coords.shape

    # Normalize to [-1, 1]
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1

    grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, H, W, 2]

    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)

    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)  # [B, H, W]

        return img, mask

    return img


def calc_alpha(input_Y):
    cdf_threshold = 0.98
    B, H, W = input_Y.shape

    # 展平每个batch的H和W维度
    flattened = input_Y.view(B, -1)  # [B, H*W]

    # 对每个batch进行排序
    sorted_values, _ = torch.sort(flattened, dim=1)  # [B, H*W]

    # 计算CDF的位置（假设均匀分布）
    # 这里我们直接使用排序后的值对应的位置比例作为CDF
    # 例如，第i个元素的CDF约为i / (H*W)
    # 我们需要找到第一个位置，其中 (i / (H*W)) >= cdf_threshold
    # 即 i >= cdf_threshold * (H*W)
    target_index = int(cdf_threshold * (H * W))
    target_index = min(target_index, H * W - 1)
    values_at_threshold = sorted_values[:, target_index].unsqueeze(-1)  # [B，1]

    alpha = cdf_threshold * 0.8 / values_at_threshold

    return values_at_threshold, alpha


def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    b, c, h, w = feature.size()
    assert flow.size(1) == 2

    grid = coords_grid(b, h, w, flow.device) + flow  # [B, 2, H, W]

    return bilinear_sample(feature, grid, padding_mode=padding_mode,
                           return_mask=mask)


def forward_backward_consistency_check(fwd_flow, bwd_flow,
                                       alpha=0.01,
                                       beta=0.5
                                       ):
    # fwd_flow, bwd_flow: [B, 2, H, W]
    # alpha and beta values are following UnFlow (https://arxiv.org/abs/1711.07837)
    assert fwd_flow.dim() == 4 and bwd_flow.dim() == 4
    assert fwd_flow.size(1) == 2 and bwd_flow.size(1) == 2
    flow_mag = torch.norm(fwd_flow, dim=1) + torch.norm(bwd_flow, dim=1)  # [B, H, W]

    warped_bwd_flow = flow_warp(bwd_flow, fwd_flow)  # [B, 2, H, W]
    warped_fwd_flow = flow_warp(fwd_flow, bwd_flow)  # [B, 2, H, W]

    diff_fwd = torch.norm(fwd_flow + warped_bwd_flow, dim=1)  # [B, H, W]
    diff_bwd = torch.norm(bwd_flow + warped_fwd_flow, dim=1)

    threshold = alpha * flow_mag + beta

    fwd_occ = (diff_fwd > threshold).float()  # [B, H, W]
    bwd_occ = (diff_bwd > threshold).float()

    return fwd_occ.unsqueeze(1), bwd_occ.unsqueeze(1)


if __name__ == '__main__':
    src_pth = r"/home/ub24017/MyCodes/Zero_Mamba/results/occlusion_test/00005.png"
    tmp_pth = r"/home/ub24017/MyCodes/Zero_Mamba/results/occlusion_test/00012.png"
    src = load_image(src_pth).unsqueeze(0)
    tmp = load_image(tmp_pth).unsqueeze(0)
    import time

    t1 = time.time()
    hm = histogram_match_tensor(src, tmp)
    t2 = time.time()
    print(t2 - t1)
    hm_v = cvt_ts2np(hm).astype(np.uint8)
    pass