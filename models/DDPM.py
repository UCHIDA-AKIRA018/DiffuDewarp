import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from simplex import Simplex_CLASS
from softmax_splatting.softsplat import softsplat
from torchvision.transforms.functional import gaussian_blur
from sklearn.metrics import roc_auc_score,roc_curve,auc,average_precision_score
import cv2
from matplotlib.colors import Normalize
import logging
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F
import sys


def get_beta_schedule(num_diffusion_steps, name="cosine"):
    betas = []
    if name == "cosine":
        max_beta = 0.999
        f = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        for i in range(num_diffusion_steps):
            t1 = i / num_diffusion_steps
            t2 = (i + 1) / num_diffusion_steps
            betas.append(min(1 - f(t2) / f(t1), max_beta))
        betas = np.array(betas)
    elif name == "linear":
        scale = 1000 / num_diffusion_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    else:
        raise NotImplementedError(f"unknown beta schedule: {name}")
    return betas


def extract(arr, timesteps, broadcast_shape, device):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape).to(device)


def mean_flat(tensor):
    return torch.mean(tensor, dim=list(range(1, len(tensor.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL Divergence between two gaussians

    :param mean1:
    :param logvar1:
    :param mean2:
    :param logvar2:
    :return: KL Divergence between N(mean1,logvar1^2) & N(mean2,logvar2^2))
    """
    return 0.5 * (-1 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretised_gaussian_log_likelihood(x, means, log_scales):
    """
        Compute the log-likelihood of a Gaussian distribution discretizing to a
        given image.
        :param x: the target images. It is assumed that this was uint8 values,
                  rescaled to the range [-1, 1].
        :param means: the Gaussian mean Tensor.
        :param log_scales: the Gaussian log stddev Tensor.
        :return: a tensor like x of log probabilities (in nats).
        """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))

    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
            x < -0.999,
            log_cdf_plus,
            torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
            )
    assert log_probs.shape == x.shape
    return log_probs


def generate_simplex_noise(
        Simplex_instance, x, t, frequency, random_param=False, octave=6, persistence=0.8,
        in_channels=1
        ):
    noise = torch.empty(x.shape).to(x.device)
    for i in range(in_channels):
        Simplex_instance.newSeed()
        if random_param:
            param = random.choice(
                    [(2, 0.6, 16), (6, 0.6, 32), (7, 0.7, 32), (10, 0.8, 64), (5, 0.8, 16), (4, 0.6, 16), (1, 0.6, 64),
                     (7, 0.8, 128), (6, 0.9, 64), (2, 0.85, 128), (2, 0.85, 64), (2, 0.85, 32), (2, 0.85, 16),
                     (2, 0.85, 8),
                     (2, 0.85, 4), (2, 0.85, 2), (1, 0.85, 128), (1, 0.85, 64), (1, 0.85, 32), (1, 0.85, 16),
                     (1, 0.85, 8),
                     (1, 0.85, 4), (1, 0.85, 2), ]
                    )
            # 2D octaves seem to introduce directional artifacts in the top left
            noise[:, i, ...] = torch.unsqueeze(
                torch.from_numpy(
                    Simplex_instance.rand_3d_fixed_T_octaves(
                            x.shape[:,i,:,:], t.detach().cpu().numpy(), param[0], param[1],
                            param[2]
                            )
                    ).to(x.device), 0
                ).repeat(x.shape[0], 1, 1, 1)
        else:
            for b in range(x.shape[0]):
                noise[b, i, ...] = torch.unsqueeze(
                    torch.from_numpy(
                            Simplex_instance.rand_3d_fixed_T_octaves(
                                    x.shape[-2:], t[b:b+1].detach().cpu().numpy(), octave,
                                    persistence, frequency[b]
                                    )
                            ).to(x.device), 0
                    )
    return noise


class GaussianDiffusionModel:
    def __init__(
            self,
            img_size,
            betas,
            img_channels=1,
            loss_weight='none',  # prop t / uniform / None
            mode="DiffusionAD",  # DiffusionAD / DiffuDewarp
            ):
        super().__init__()

        self.simplex = Simplex_CLASS()
        if mode == "DiffusionAD":
            self.noise_fn = lambda x, t: torch.randn_like(x)
        elif mode == "DiffuDewarp":
            self.noise_fn = lambda x, t, frequency: generate_simplex_noise(self.simplex, x, t, frequency, False, in_channels=3, octave=2)

        self.img_size = img_size
        self.img_channels = img_channels
        self.num_timesteps = len(betas)

        if loss_weight == 'prop-t':
            self.weights = np.arange(self.num_timesteps, 0, -1)
        elif loss_weight == "uniform":
            self.weights = np.ones(self.num_timesteps)

        self.loss_weight = loss_weight
        alphas = 1 - betas
        self.betas = betas
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_betas = np.sqrt(betas)

        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:],0.0)


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        self.gauss_blur = T.GaussianBlur(kernel_size=31, sigma=3)

    def get_flow_image(self,flow):
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        # フローの方向と速度を計算
        magnitude, angle = cv2.cartToPolar(flow[:,:,0].numpy(), flow[:,:,1].numpy())
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # HSVイメージをBGRに変換して表示
        flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
         # 凡例を描画するための円形データを作成
        legend_size = 50  # 小さめの凡例のサイズ
        center_x, center_y = legend_size // 2, legend_size // 2
        hue_legend = np.zeros((legend_size, legend_size, 3), dtype=np.uint8)

        for y in range(legend_size):
            for x in range(legend_size):
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                if distance < legend_size // 2:  # 円の内側のみ処理
                    angle = np.arctan2(dy, dx)
                    angle_deg = (angle * 180 / np.pi) % 360
                    hue_legend[y, x, 0] = int(angle_deg / 2)  # HSVでHueは0-180にスケール
                    hue_legend[y, x, 1] = 255  # 彩度
                    hue_legend[y, x, 2] = 255  # 明度

        # HSVイメージをBGRに変換
        hue_legend_bgr = cv2.cvtColor(hue_legend, cv2.COLOR_HSV2BGR)

        # 凡例を画像右上に埋め込む
        legend_position_x = flow_visualization.shape[1] - legend_size - 20  # 右端から20px内側
        legend_position_y = 20  # 上端から20px下
        output = flow_visualization.copy()
        output[
            legend_position_y : legend_position_y + legend_size,
            legend_position_x : legend_position_x + legend_size,
        ] = hue_legend_bgr

        return output

    def show_flow_on_image(self,image,flow,min_magnitude=4.0,spacing=4):
        # # Extract flow components
        flow_x = flow[0, 0, :, :].cpu().numpy() * 128
        flow_y = flow[0, 1, :, :].cpu().numpy() * 128

        # # Compute the magnitude of the flow
        magnitude = np.sqrt(flow_x**2 + flow_y**2)

        # # Create a grid for quiver arrows
        Y, X = np.mgrid[0:flow_x.shape[0], 0:flow_x.shape[1]]

        # # Apply spacing and filter based on magnitude
        Y = Y[::spacing, ::spacing]
        X = X[::spacing, ::spacing]
        flow_x = flow_x[::spacing, ::spacing]
        flow_y = flow_y[::spacing, ::spacing]
        magnitude = magnitude[::spacing, ::spacing]

        # # Filter arrows based on minimum magnitude
        mask = magnitude >= min_magnitude
        X = X[mask]
        Y = Y[mask]
        U = flow_x[mask]
        V = flow_y[mask]

        fig, ax = plt.subplots()
        ax.set_xticks([])  # x軸の目盛りを非表示
        ax.set_yticks([])  # y軸の目盛りを非表示
        ax.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        # ax.quiver(X, Y, U, V, color='red', angles='xy', scale_units='xy', scale=1, headlength=2 ,headwidth=1)
        mag, ang = cv2.cartToPolar(U, V, angleInDegrees=True)
        # カラーマップの範囲を制御するためにNormalizeを使用
        norm = Normalize(vmin=0, vmax=1)

        if ang is None:
            logging.info(ang)
        else:
            ang = (-ang - 120) % 360
            ang = ang / 360.0
            ax.quiver(X, Y, U, V, ang, cmap='hsv', angles='xy', scale_units='xy', scale=1, headlength=2 ,headwidth=1, norm = norm)

        # 円形カラーバー（完全な円）を作成
        circle_size = 70  # 円のサイズ
        center = circle_size // 2
        theta = np.linspace(0, 2 * np.pi, circle_size)
        x, y = np.meshgrid(np.arange(circle_size), np.arange(circle_size))
        distance = np.sqrt((x - center)**2 + (y - center)**2)

        # カラーマップ用の角度データ
        angle = (-np.arctan2(y - center, x - center) * 180 ) / np.pi - 120
        angle[angle < 0] += 360

        # 円形マスクの適用
        circle_mask = distance <= (circle_size // 2)
        circle_hsv = np.zeros((circle_size, circle_size, 4), dtype=np.uint8)  # RGBA形式
        circle_hsv[..., 0] = (angle / 2).astype(np.uint8)  # Hue
        circle_hsv[..., 1] = 255  # Saturation
        circle_hsv[..., 2] = 255  # Brightness
        circle_hsv[..., 3] = (circle_mask * 255).astype(np.uint8)  # アルファチャンネル（透明度）

        # HSVからRGBAに変換
        circle_rgb = cv2.cvtColor(circle_hsv[..., :3], cv2.COLOR_HSV2RGB)  # 最初の3チャンネルのみ使用
        circle_rgba = np.dstack((circle_rgb, circle_hsv[..., 3]))  

        # 保存されたプロットに円を追加
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        flow_visualization = Image.open(buf).convert("RGBA")
        buf.close()

        # 円形カラーバーを右上に配置
        circle_img = Image.fromarray(circle_rgba, mode="RGBA")
        overlay_x = flow_visualization.width - circle_img.width - 30  # 右端から20px内側
        overlay_y = 30  # 上端から20px下
        flow_visualization.paste(circle_img, (overlay_x, overlay_y), circle_img)

        plt.close(fig)  # Close the figure
        flow_visualization = np.array(flow_visualization)
        
        return flow_visualization

    def show_image(self,image,file_name):
        # PyTorchテンソルをNumPy配列に変換
        if isinstance(image, torch.Tensor):
            # PyTorch Tensor の場合は .to() を使って CPU に移動
            image_tmp = image.to('cpu').detach().numpy().copy()
        elif isinstance(image, np.ndarray):
            # NumPy 配列の場合はそのままコピー
            image_tmp = image.copy()

        # image_tmp = image.to('cpu').detach().numpy().copy()
        if image_tmp.shape[0] == 1:
            image_tmp = image_tmp[0].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        # else:
        #     image_tmp = image_tmp.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        
        # グレースケールの場合、最後の次元を削除
        if image_tmp.shape[-1] == 1:
            image_tmp = image_tmp[:, :, 0]
        
        # 値を0〜255の範囲にスケーリングし、uint8型に変換
        image_tmp = (image_tmp - image_tmp.min()) / (image_tmp.max() - image_tmp.min()) * 255
        image_tmp = image_tmp.astype(np.uint8)

        # RGB -> BGRに変換（カラーチャンネルの反転）
        if image_tmp.ndim == 3 and image_tmp.shape[2] == 3:  # カラー画像の場合
            image_tmp = cv2.cvtColor(image_tmp, cv2.COLOR_RGB2BGR)
        
        # OpenCVで画像を保存
        cv2.imwrite(file_name + '.png', image_tmp)

    def show_flow(self,flow,file_name):

        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255

        # フローの方向と速度を計算
        magnitude, angle = cv2.cartToPolar(flow[:, :, 0].numpy(), flow[:, :, 1].numpy())
        hsv[..., 0] = angle * 180 / np.pi / 2
        normalized_magnitude = np.clip(magnitude * 128 * 30, 0, 255)
        hsv[..., 2] = normalized_magnitude.astype(np.uint8)

        # HSVイメージをBGRに変換
        flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 凡例を描画するための円形データを作成
        legend_size = 50  # 小さめの凡例のサイズ
        center_x, center_y = legend_size // 2, legend_size // 2
        hue_legend = np.zeros((legend_size, legend_size, 3), dtype=np.uint8)

        for y in range(legend_size):
            for x in range(legend_size):
                dx = x - center_x
                dy = y - center_y
                distance = np.sqrt(dx**2 + dy**2)
                if distance < legend_size // 2:  # 円の内側のみ処理
                    angle = np.arctan2(dy, dx)
                    angle_deg = (angle * 180 / np.pi) % 360
                    hue_legend[y, x, 0] = int(angle_deg / 2)  # HSVでHueは0-180にスケール
                    hue_legend[y, x, 1] = 255  # 彩度
                    hue_legend[y, x, 2] = 255  # 明度

        # HSVイメージをBGRに変換
        hue_legend_bgr = cv2.cvtColor(hue_legend, cv2.COLOR_HSV2BGR)

        # 凡例を画像右上に埋め込む
        legend_position_x = flow_visualization.shape[1] - legend_size - 20  # 右端から20px内側
        legend_position_y = 20  # 上端から20px下
        output = flow_visualization.copy()
        output[
            legend_position_y : legend_position_y + legend_size,
            legend_position_x : legend_position_x + legend_size,
        ] = hue_legend_bgr

        # 画像を保存
        plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0)
        plt.close()

    def sample_t_with_weights(self, b_size, device):
        p = self.weights / np.sum(self.weights)
        indices_np = np.random.choice(len(p), size=b_size, p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / len(p) * p[indices_np]
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights

    def predict_x_0_from_eps(self, x_t, t, eps):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device) * eps)

    def predict_eps_from_x_0(self, x_t, t, pred_x_0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape, x_t.device) * x_t
                - pred_x_0) \
               / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape, x_t.device)

    def q_mean_variance(self, x_0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0
        )
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape, x_0.device)
        log_variance = extract(
                self.log_one_minus_alphas_cumprod, t, x_0.shape, x_0.device
                )
        return mean, variance, log_variance

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """

        # mu (x_t,x_0) = \frac{\sqrt{alphacumprod prev} betas}{1-alphacumprod} *x_0
        # + \frac{\sqrt{alphas}(1-alphacumprod prev)}{ 1- alphacumprod} * x_t
        posterior_mean = (extract(self.posterior_mean_coef1, t, x_t.shape, x_t.device) * x_0
                          + extract(self.posterior_mean_coef2, t, x_t.shape, x_t.device) * x_t)

        # var = \frac{1-alphacumprod prev}{1-alphacumprod} * betas
        posterior_var = extract(self.posterior_variance, t, x_t.shape, x_t.device)
        posterior_log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape, x_t.device)
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(self, model, x_t, t, estimate_noise=None):
        """
        Finds the mean & variance from N(x_{t-1}; mu_theta(x_t,t), sigma_theta (x_t,t))

        :param model:
        :param x_t:
        :param t:
        :return:
        """
        if estimate_noise == None:
            estimate_noise = model(x_t, t)

        # fixed model variance defined as \hat{\beta}_t - could add learned parameter
        model_var = np.append(self.posterior_variance[1], self.betas[1:])
        model_logvar = np.log(model_var)
        model_var = extract(model_var, t, x_t.shape, x_t.device)
        model_logvar = extract(model_logvar, t, x_t.shape, x_t.device)

        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        model_mean, _, _ = self.q_posterior_mean_variance(
                pred_x_0, x_t, t
                )
        return {
            "mean":         model_mean,
            "variance":     model_var,
            "log_variance": model_logvar,
            "pred_x_0":     pred_x_0,
            }

    def sample_p(self, model, x_t, t, denoise_fn="gauss"):
        out = self.p_mean_variance(model, x_t, t)
        # noise = torch.randn_like(x_t)
        if denoise_fn == "gauss":
            noise = torch.randn_like(x_t) 
        else:
            noise = denoise_fn(x_t, t)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}

    def forward_backward(
            self, model, x, see_whole_sequence="half", t_distance=None, denoise_fn="gauss",
            ):
        assert see_whole_sequence == "whole" or see_whole_sequence == "half" or see_whole_sequence == None

        if t_distance == 0:
            return x.detach()

        if t_distance is None:
            t_distance = self.num_timesteps
        seq = [x.cpu().detach()]
        if see_whole_sequence == "whole":

            for t in range(int(t_distance)):
                t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
                # noise = torch.randn_like(x)
                noise = self.noise_fn(x, t_batch).float()
                with torch.no_grad():
                    x = self.sample_q_gradual(x, t_batch, noise)

                seq.append(x.cpu().detach())
        else:
            t_tensor = torch.tensor([t_distance - 1], device=x.device).repeat(x.shape[0])
            x = self.sample_q(
                    x, t_tensor,
                    self.noise_fn(x, t_tensor).float()
                    )
            if see_whole_sequence == "half":
                seq.append(x.cpu().detach())

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x.device).repeat(x.shape[0])
            with torch.no_grad():
                out = self.sample_p(model, x, t_batch, denoise_fn)
                x = out["sample"]
            if see_whole_sequence:
                seq.append(x.cpu().detach())

        return x.detach() if not see_whole_sequence else seq

    def sample_q(self, x_0, t, noise):
        """
            q (x_t | x_0 )

            :param x_0:
            :param t:
            :param noise:
            :return:
        """
        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape, x_0.device) * x_0 +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape, x_0.device) * noise)

    def sample_q_gradual(self, x_t, t, noise):
        """
        q (x_t | x_{t-1})
        :param x_t:
        :param t:
        :param noise:
        :return:
        """
        return (extract(self.sqrt_alphas, t, x_t.shape, x_t.device) * x_t +
                extract(self.sqrt_betas, t, x_t.shape, x_t.device) * noise)

    def calc_vlb_xt(self, model, x_0, x_t, t, estimate_noise=None):
        # find KL divergence at t
        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_0, x_t, t)
        output = self.p_mean_variance(model, x_t, t, estimate_noise)
        kl = normal_kl(true_mean, true_log_var, output["mean"], output["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretised_gaussian_log_likelihood(
                x_0, output["mean"], log_scales=0.5 * output["log_variance"]
                )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        nll = torch.where((t == 0), decoder_nll, kl)
        return {"output": nll, "pred_x_0": output["pred_x_0"]}

    def calc_loss(self, model, x_0, t):

        noise = self.noise_fn(x_0, t).float()
        x_t = self.sample_q(x_0, t, noise)
        estimate_noise = model(x_t, t)
        loss = {}
        loss["loss"] = mean_flat((estimate_noise - noise).square())
        
        return loss, x_t, estimate_noise

   
    def norm_guided_one_step_denoising(self, model, x_0, anomaly_label,args):
        # two-scale t
        normal_t = torch.randint(0, args["less_t_range"], (x_0.shape[0],),device=x_0.device)
        noisier_t = torch.randint(args["less_t_range"],self.num_timesteps,(x_0.shape[0],),device=x_0.device)
        
        normal_loss, x_normal_t, estimate_noise_normal = self.calc_loss(model, x_0, normal_t)
        noisier_loss, x_noiser_t, estimate_noise_noisier = self.calc_loss(model, x_0, noisier_t)
        
        pred_x_0_noisier = self.predict_x_0_from_eps(x_noiser_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)   

        # Only calculate the noise loss of normal samples according to formula 9.
        loss = (normal_loss["loss"]+noisier_loss["loss"])[anomaly_label==0].mean()
        # When the batch size is small, it may lead to an entire batch consisting solely of abnormal samples
        # If they are all abnormal samples, set loss to 0.
        if torch.isnan(loss):
            loss.fill_(0.0)

        estimate_noise_hat = estimate_noise_normal - extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_normal_t.shape, x_0.device) * args["condition_w"] * (pred_x_t_noisier-x_normal_t)
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)

        return loss,pred_x_0_norm_guided,normal_t,x_normal_t,x_noiser_t


    def norm_guided_one_step_denoising_eval(self, model, x_0, normal_t,noisier_t,args):

        
        normal_loss, x_normal_t, estimate_noise_normal = self.calc_loss(model, x_0, normal_t)
        noisier_loss, x_noisier_t, estimate_noise_noisier = self.calc_loss(model, x_0, noisier_t)

        pred_x_0_noisier = self.predict_x_0_from_eps(x_noisier_t, noisier_t, estimate_noise_noisier).clamp(-1, 1)
        pred_x_t_noisier = self.sample_q(pred_x_0_noisier, normal_t, estimate_noise_normal)    

        loss = (normal_loss["loss"]+noisier_loss["loss"]).mean()
        pred_x_0_normal = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_normal).clamp(-1, 1)
        estimate_noise_hat = estimate_noise_normal - extract(self.sqrt_one_minus_alphas_cumprod, normal_t, x_0.shape, x_0.device) * args["condition_w"] * (pred_x_t_noisier-x_normal_t)
        pred_x_0_norm_guided = self.predict_x_0_from_eps(x_normal_t, normal_t, estimate_noise_hat).clamp(-1, 1)
        
        return loss,pred_x_0_norm_guided,pred_x_0_normal,pred_x_0_noisier,x_normal_t,x_noisier_t,pred_x_t_noisier  
    

    def noise_t(self, model, x_0, t,args):
        loss, x_t, estimate_noise = self.calc_loss(model, x_0, t)
        loss = (loss["loss"] ).mean()
        pred_x_0 = self.predict_x_0_from_eps(x_t, t, estimate_noise).clamp(-1, 1)
        return loss,pred_x_0,x_t


    def new_warping_denoising_batch(self, model, x_0, args, perlin_mask=None, perlin_mask_sec=None, object_mask=None, another_image=None, t=None, mode="train"):
        # model: unet
        # x_0: train_image (no-synanomaly) torch.Size([16, 3, 256, 256])
        # perlin_mask: torch.Size([16, 256, 256, 1])
        # object_mask 前景箇所(テクスチャはランダムな場所に出る) torch.Size([16, 256, 256]) 
        # t: time
        # mode: train or eval

        # datasetで定義されているtextureはランダムなマスクが入ってくる (carpet内にあるもの)
        object_mask = (object_mask > 0).float().unsqueeze(3)  # torch.Size([16, 256, 256, 1])
        
        if t == None:
            t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=x_0.device)

        if args['subclass'] == 'clip':  
            x_0, object_mask  = self.apply_random_transformations(x_0, object_mask.permute(0,3,1,2).repeat(1, 3, 1, 1))
            object_mask = object_mask.permute(0,2,3,1)[:,:,:,:1]

        # anomaly_labels 0 good 1 anomaly(clip) 3 anomaly(MVtec)
        if args['subclass'] != 'clip': 
            anomaly_labels = torch.ones_like(t)*3
        elif args['subclass'] == 'clip': 
            anomaly_labels = torch.randint(0, 2, (x_0.shape[0],), device=x_0.device)
        
        target_areas = torch.randint(50, 1250, (x_0.shape[0],)) / args['img_size'][0] / args['img_size'][0] * x_0.shape[2] * x_0.shape[2] # Adjust area range as needed

        if args['subclass'] != 'clip': 
            object_mask_cropped = object_mask
        elif args['subclass'] == 'clip': 
            # x_0をobejctmaskでcropし、それ以外は真っ黒に
            # x_0: train_image (no-synanomaly) torch.Size([16, 3, 256, 256])
            x_0 = torch.zeros_like(x_0) + x_0 * object_mask.permute(0,3,1,2).repeat(1, 3, 1, 1)
            back_image = torch.zeros_like(x_0)  # 背景は真っ黒
            object_mask_cropped = self.make_line_mask(object_mask)
            
        # clipのためのマスクと基準点 
        mask, target_point = self.make_random_mask_for_bend(object_mask_cropped, target_areas)
        if args['subclass'] != 'clip': 
            mask = perlin_mask
            back_image = x_0.clone()

        # ラベルが0ならなしに　
        mask = mask * ((anomaly_labels == 1) | (anomaly_labels == 3)).view(x_0.shape[0], 1, 1, 1)
        
        degrees = torch.rand(x_0.shape[0]) * (90 - 0)
        radians = degrees * (torch.pi / 180)  # ラジアンに変換
        # 正負の符号をランダムに決定
        sign = torch.where(torch.rand(x_0.shape[0]) < 0.5, -1, 1)  # ±1の符号
        max_angle = (radians * sign).to(x_0.device) * (anomaly_labels != 3)  

        angle_t = max_angle * t / self.num_timesteps
        angle_t_1 = max_angle * torch.clamp(t-1, min=0) / self.num_timesteps

        if args['subclass'] != 'clip': 
            frequency = torch.full((t.shape[0],), 32) / args['img_size'][0] * x_0.shape[2]
            noise_rate = torch.full((t.shape[0],), 10) / args['img_size'][0] * x_0.shape[2]
        elif args['subclass'] == 'clip':  
            frequency = torch.full((t.shape[0],), 64) / args['img_size'][0] * x_0.shape[2]
            noise_rate = torch.full((t.shape[0],), 6) / args['img_size'][0] * x_0.shape[2]
            
        noise = self.noise_fn(x_0, t, frequency.numpy()).float()
        
        isDeformation = torch.full((noise.shape[0],), True, device=x_0.device)
        if args['subclass'] != 'clip': 
            isDeformation = (torch.rand(noise.shape[0]) < 0.5).to(x_0.device)

        noise = noise[:,:2, :, :] * isDeformation.float().view(noise.shape[0], 1, 1, 1)  # (16, 1, 1, 1)
        max_noise = noise / noise_rate.view(t.shape[0], 1, 1, 1).to(x_0.device)
        max_noise = max_noise * ((anomaly_labels == 1) | (anomaly_labels == 3)).view(x_0.shape[0], 1, 1, 1)

        noise_t = max_noise/ self.num_timesteps * t.view(-1, 1, 1, 1).to(x_0.device)
        noise_t_1 = max_noise/ self.num_timesteps * torch.clamp(t-1, min=0).view(-1, 1, 1, 1).to(x_0.device)
        flow_t = noise_t.permute(0, 2, 3, 1)
        flow_t_1 = noise_t_1.permute(0, 2, 3, 1)
        
        # x_t, x_t_1の作成
        if args['subclass'] != 'clip': 
            straight_mask = torch.ones(x_0.shape[0]) > 0.5 #全てstraight_mask
        elif args['subclass'] == 'clip':  
            straight_mask = (torch.rand(x_0.shape[0]) > 0.5) # 50%の確率でTrueまたはFalseを生成

        x_t, mask_x_t, flow_x_t_from_x_0  = self.make_anomaly_bend(x_0, mask, target_point, angle_t, back_image, flow_t, straight_mask)
        x_t_1, _, _ = self.make_anomaly_bend(x_0, mask, target_point, angle_t_1, back_image, flow_t_1, straight_mask)
        
        
        if args['anomaly_color']:
            perlin_mask_sec = perlin_mask_sec.squeeze(-1)  # [16, 256, 256]
            perlin_mask_sec[mask_x_t.squeeze(1) > 0] = 0
            perlin_mask_sec[mask.squeeze(-1) > 0] = 0
            perlin_mask_sec[object_mask.squeeze(-1) == 0] = 0
            perlin_mask_sec_tmp = perlin_mask_sec.unsqueeze(-1).clone()
            perlin_mask_sec = perlin_mask_sec.unsqueeze(-1) * (anomaly_labels != 0).view(x_0.shape[0], 1, 1, 1)

            max_alpha = torch.ones(x_0.shape[0], device=x_0.device).view(-1, 1, 1, 1)*0.5
            alpha_t = max_alpha
            x_t = self.make_anomaly_color(x_t,perlin_mask_sec,another_image, alpha_t)
            x_t_1 = self.make_anomaly_color(x_t_1,perlin_mask_sec,another_image,alpha_t)

        # 形状を 16, 3, 256, 256 にリサイズ
        x_0 = torch.nn.functional.interpolate(x_0, size=(args['img_size'][0], args['img_size'][1]), mode='bilinear', align_corners=False)
        x_t = torch.nn.functional.interpolate(x_t, size=(args['img_size'][0], args['img_size'][1]), mode='bilinear', align_corners=False)
        x_t_1 = torch.nn.functional.interpolate(x_t_1, size=(args['img_size'][0], args['img_size'][1]), mode='bilinear', align_corners=False)
        flow_x_t_from_x_0 = torch.nn.functional.interpolate(flow_x_t_from_x_0, size=(args['img_size'][0], args['img_size'][1]), mode='bilinear', align_corners=False)
        mask_x_t = torch.nn.functional.interpolate(mask_x_t, size=(args['img_size'][0], args['img_size'][1]), mode='bilinear', align_corners=False)

        if args['subclass'] == 'clip':  
            flow_T = max_noise.permute(0, 2, 3, 1)
            x_T, mask_x_T, flow_x_T_from_x_0 = self.make_anomaly_bend(x_0, mask, target_point, max_angle, back_image, flow_T, straight_mask)
            x_t_test, mask_x_t_test  = self.make_pre_image(x_T, flow_x_T_from_x_0, self.num_timesteps/(self.num_timesteps - t) , mask_x_T)
            flow_x_t_from_x_0_test, _  = self.make_pre_image(flow_x_T_from_x_0, flow_x_T_from_x_0, self.num_timesteps/(self.num_timesteps - t) , mask_x_T)
        
            swap_mask = (torch.rand(x_t.shape[0]) < 0.5).to(x_t.device)  # Trueの箇所が入れ替え対象
            x_t = torch.where(swap_mask.view(-1, 1, 1, 1), x_t_test, x_t)
            mask_x_t = torch.where(swap_mask.view(-1, 1, 1, 1), mask_x_t_test, mask_x_t)
            flow_x_t_from_x_0 = torch.where(swap_mask.view(-1, 1, 1, 1), flow_x_t_from_x_0_test, flow_x_t_from_x_0)

        # for i in range(t.shape[0]):
        #     self.show_image(x_0[i:i+1,:,:,:], f"test_final_{i}_{anomaly_labels[i]}_x_0")
        #     self.show_image(x_t_1[i:i+1,:,:,:], f"test_final_{i}_{anomaly_labels[i]}_x_{t[i].item()}_1_test")
        #     self.show_image(x_t[i:i+1,:,:,:], f"test_final_{i}_{anomaly_labels[i]}_x_{t[i].item()}_test")
        #     self.show_flow(flow_x_t_from_x_0[i,:,:,:].permute(1,2,0).to('cpu'), f"test_final_{i}_{anomaly_labels[i]}_flow_test")
        #     self.show_image(mask_x_t[i:i+1,:,:,:].repeat(1, 3, 1, 1) , f"test_final_{i}_{anomaly_labels[i]}_mask")
        # sys.exit()

        # modelによる推定
        estimate_x_t_1 = model(x_t, t)

        if args['output_type'] == 'flow_from_0&x_0&x_pre_t':
            estimate_x_0 = estimate_x_t_1[:,:3,:,:]
            estimate_flow_x_t_from_x_0 = estimate_x_t_1[:,3:,:,:]
            estimate_x_t_1, estimate_mask_x_t_1  = self.make_pre_image(x_t, estimate_flow_x_t_from_x_0, t, mask_x_t)
            estimate_x_t_1 = self.interporate_pre_image(estimate_x_t_1,estimate_mask_x_t_1,estimate_x_0,mask_x_t)
        elif args['output_type'] == 'flow_from_0&x_0':
            estimate_x_0 = estimate_x_t_1[:,:3,:,:]
            estimate_flow_x_t_from_x_0 = estimate_x_t_1[:,3:,:,:]
        
        # loss計算
        loss_flow = mean_flat((estimate_flow_x_t_from_x_0 - flow_x_t_from_x_0).abs())
        if args['output_type'] == 'flow_from_0&x_0':
            loss = loss_flow + mean_flat((estimate_x_0 - x_0).abs())
        elif args['output_type'] == 'flow_from_0&x_0&x_pre_t':
            loss = loss_flow + mean_flat((estimate_x_t_1 - x_t_1).abs())+ mean_flat((estimate_x_0 - x_0).abs())

        return loss.mean(), x_t, mask_x_t, estimate_x_0

    
    def backwarp(self, target, flow):
        grid_h, grid_w = target.shape[2], target.shape[3]
        grid = torch.meshgrid(torch.linspace(-1, 1, grid_h), torch.linspace(-1, 1, grid_w))
        grid = torch.stack((grid[1], grid[0]), 2)  # PyTorchのgridは (y, x) の順番であるため、(x, y) にする必要がある
        grid = grid.unsqueeze(0).repeat(target.shape[0], 1, 1, 1).to(target.device)   
        # [batch,256,256,2]

        # グリッドをFlowで変形する
        warped_grid = grid + flow[0].permute(1,2,0)
        
        warped_grid = warped_grid.clamp(-1, 1) # グリッドを -1 から 1 の範囲にクリップする

        target = target
        # 4. 変形したグリッドを使って画像をサンプリングする
        target = torch.nn.functional.grid_sample(target, warped_grid, align_corners=True)

        return target


    def apply_dilation(self, object_mask, kernel_size=3, iterations=1):
        # カーネルの定義（正方形の構造要素）
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=object_mask.device)
        
        # マスクのチャンネル次元の追加
        object_mask = object_mask.permute(0,3,1,2).float()  # [16, 256, 256, 1] -> [16, 1, 256, 256]
        
        # 膨張を複数回実行する場合
        for _ in range(iterations):
            object_mask = torch.nn.functional.conv2d(object_mask, kernel, padding=kernel_size // 2)
            object_mask = (object_mask > 0).float()  # 0/1のバイナリに戻す

        # 元の形状に戻す
        return object_mask.permute(0,2,3,1)  # [16, 1, 256, 256] -> [16, 256, 256]


    def apply_random_transformations(self, images, masks):
        """
        画像とマスクに同一のランダム変換 (回転・並進) を適用
        """
        transformed_images = []
        transformed_masks = []
        
        for image, mask in zip(images, masks):
            # ランダムな変換パラメータを生成
            angle = random.uniform(-10, 10)  # 回転角度
            max_dx = 0.1 * image.shape[2]    # 並進量 (x方向)
            max_dy = 0.1 * image.shape[1]    # 並進量 (y方向)
            translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
            
            # 同じ変換を画像とマスクに適用
            transformed_image = F.affine(image, angle=angle, translate=translate, scale=1.0, shear=(0.0, 0.0))
            transformed_mask = F.affine(mask, angle=angle, translate=translate, scale=1.0, shear=(0.0, 0.0))
            
            transformed_images.append(transformed_image)
            transformed_masks.append(transformed_mask)
        
        return torch.stack(transformed_images), torch.stack(transformed_masks)


    def make_line_mask(self, mask):
        # mask shape: (batchsize, 256, 256, 1) のPyTorchテンソル
        batch_size = mask.shape[0]
        result_mask = mask.clone()  # 結果を格納するためのマスクをコピー

        # バッチ毎に処理
        for i in range(batch_size):
            # (バッチ内の)単一のマスクを取得
            single_mask = result_mask[i, :, :, 0].cpu().numpy().astype(np.uint8)*255  # PyTorchテンソル -> NumPy配列

            # Sobelフィルタによるエッジ検出
            sobel_x = cv2.Sobel(single_mask, cv2.CV_32F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(single_mask, cv2.CV_32F, 0, 1, ksize=5)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_magnitude = np.abs(sobel_magnitude)

            # エッジを二値化
            binary_image = np.where(sobel_magnitude > 200, 255, 0).astype(np.uint8)

            # ハフ変換で線分検出
            minLineLength = 100
            maxLineGap = 120
            threshold = 50
            lines = cv2.HoughLinesP(binary_image, 1, np.pi/180, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
            
            # 全てのラインを描画
            img_with_lines = cv2.cvtColor(single_mask, cv2.COLOR_GRAY2RGB)  # グレースケール -> カラー画像
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 線を描画

            if lines is not None:
                # 線分の角度を計算
                def calculate_angle(line):
                    x1, y1, x2, y2 = line[0]
                    return np.arctan2(y2 - y1, x2 - x1)

                angles = [calculate_angle(line) for line in lines]
                mean_angle = np.mean(angles)

                # 平均角度から外れた線分を除外
                angle_tolerance = np.deg2rad(2)
                lines = [line for line, angle in zip(lines, angles) if abs(angle - mean_angle) <= angle_tolerance]

                img_with_lines = cv2.cvtColor(single_mask, cv2.COLOR_GRAY2RGB)  # グレースケール -> カラー画像
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 線を描画

                # # 描画した画像を保存
                # cv2.imwrite(f"test_{i}_lines_th.png", img_with_lines)

                if len(lines) >= 2:
                    # 線分の距離を計算する関数
                    def line_distance(line1, line2):
                        x1, y1, x2, y2 = line1[0]
                        x3, y3, x4, y4 = line2[0]
                        return min(np.hypot(x1 - x3, y1 - y3), np.hypot(x2 - x4, y2 - y4),
                                np.hypot(x1 - x4, y1 - y4), np.hypot(x2 - x3, y2 - y3))

                    # 条件に合う2本の線分をランダムに選択
                    min_distance, max_distance = 5, 8
                    line1 = random.choice(lines)
                    line2 = next((l for l in lines if line1 is not l and min_distance <= line_distance(line1, l) <= max_distance), None)

                    if line2 is not None:
                        # 2本の線分の端点を取得
                        (x1, y1, x2, y2), (x3, y3, x4, y4) = line1[0], line2[0]

                        # マスク作成
                        mask = np.zeros_like(single_mask)
                        pts = np.array([(x1, y1), (x2, y2), (x4, y4), (x3, y3)], dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)

                        # 生成したマスクを元のマスクに適用
                        single_mask = cv2.bitwise_and(single_mask, mask)
            
            # もし線分が見つからなければ、そのまま元のマスクを保持
            result_mask[i, :, :, 0] = torch.from_numpy(single_mask)/255

        return result_mask


    def interporate_pre_image(self,x_t_1,x_t_1_mask,x_0,x_t_mask):
        # x_t_1: train_image (no-synanomaly) torch.Size([16, 3, 256, 256])
        # x_t_1_mask: train_image (no-synanomaly) torch.Size([16, 1, 256, 256])
        # x_0: train_image (no-synanomaly) torch.Size([16, 3, 256, 256])
        # x_t_mask: train_image (no-synanomaly) torch.Size([16, 1, 256, 256])
        exclusive_mask = (x_t_mask == 1) & (x_t_1_mask == 0)  # mask areas in x_t_mask only
        # Step 2: Expand mask dimensions to match x_t_1's channels if needed
        exclusive_mask_expanded = exclusive_mask.repeat(1, x_t_1.shape[1], 1, 1)
        # Step 3: Replace masked areas in x_t_1 with values from x_0
        x_t_1_interpolated = torch.where(exclusive_mask_expanded, x_0, x_t_1)

        return x_t_1_interpolated


    def make_pre_image(self, x_t, flow_t_from_0, t, target_mask):
        # z = ~target_mask.bool() * 9 + 1
        z = target_mask.bool()
        # t が 0 のインデックスを取得
        mask = t == 0
        # 結果用のテンソルをゼロで初期化
        flow_t_from_0 = flow_t_from_0 * x_t.shape[3] / 2

        flow_scale = torch.zeros_like(flow_t_from_0)
        flow_scale[~mask] = flow_t_from_0[~mask] / t[~mask].view(-1, 1, 1, 1)
        
        x_t_1 = softsplat(tenIn=x_t, tenFlow=flow_scale, tenMetric=(z).clip(0.001, 1.0), strMode='linear')
        x_t_1_mask = softsplat(tenIn=target_mask, tenFlow=flow_scale, tenMetric=(z).clip(0.001, 1.0), strMode='linear')
        return x_t_1, x_t_1_mask


    def make_random_mask_for_bend(self, object_mask, target_areas):
        # object_mask is of shape [16, 256, 256, 1]
        batch_size, height, width, _ = object_mask.shape
        object_mask = object_mask.squeeze(-1)  # Shape [16, 256, 256]

        # Initialize the final mask and last updated points
        final_mask = torch.zeros_like(object_mask)
        last_updated_points = [None] * batch_size

        # Step 1: Initialize with a random point for each batch item
        for b in range(batch_size):
            non_zero_indices = torch.nonzero(object_mask[b])  # Get non-zero indices for this batch item
            if non_zero_indices.size(0) > 0:
                random_idx = torch.randint(0, non_zero_indices.size(0), (1,)).item()
                selected_point = non_zero_indices[random_idx]
                final_mask[b, selected_point[0], selected_point[1]] = 1  # Set the initial point
                last_updated_points[b] = selected_point
            else:
                last_updated_points[b] = torch.tensor([0, 0], dtype=torch.long, device=object_mask.device)

        # Calculate the current areas
        current_areas = final_mask.sum(dim=[1, 2])

        # Track which batches are still expanding
        active_batches = current_areas < target_areas.to(current_areas.device)

        while active_batches.any():
            # Dilate the mask for all active batches
            expanded_masks = torch.nn.functional.max_pool2d(
                final_mask.unsqueeze(1), kernel_size=15, stride=1, padding=7
            ).squeeze(1)
            # Ensure expanded mask stays within the original object mask
            new_mask = expanded_masks * object_mask

            new_mask = torch.where(
                active_batches.unsqueeze(1).unsqueeze(2), new_mask, final_mask
            )

            # Find newly added points (where final_mask was 0 but new_mask is 1)
            newly_added = (new_mask - final_mask) > 0

            # If no new points are added for any active batch, break
            if not newly_added.any():
                break

            # Update the final mask only for active batches
            final_mask = new_mask

            # Update current areas and active_batches
            current_areas = final_mask.sum(dim=[1, 2])

            # For each active batch, sample one of the newly added points
            for b in range(batch_size):
                if active_batches[b] == True:  # Process only active batches
                    new_points = torch.nonzero(newly_added[b])  # Get newly added points
                    if new_points.size(0) > 0:
                        random_idx = torch.randint(0, new_points.size(0), (1,)).item()
                        last_updated_points[b] = new_points[random_idx]  # Store the sampled point

            active_batches = current_areas < target_areas.to(current_areas.device)

        # Reshape final mask to [16, 256, 256, 1] to match the input format
        final_mask = final_mask.unsqueeze(-1)

        return final_mask, last_updated_points

    
    def make_anomaly_color(self, image, mask, back_image, alpha):
        """
        mask 部分に対して image と back_image を alpha に基づいてアルファブレンディングします。

        Args:
            image (torch.Tensor): 元画像 [B, 3, H, W]
            mask (torch.Tensor): マスク画像 [B, H, W, 1]
            back_image (torch.Tensor): 背景画像 [B, 3, H, W]
            alpha (torch.Tensor): アルファ値 [B] (バッチごとに異なる)

        Returns:
            torch.Tensor: アルファブレンディング後の画像
        """
        dtype = image.dtype  # `image` のデータ型に統一する
        mask = mask.to(dtype)
        back_image = back_image.to(dtype)
        alpha = alpha.to(dtype)

        # mask の次元を調整 [B, H, W, 1] -> [B, 1, H, W] に変換
        mask = mask.permute(0, 3, 1, 2)  # [B, 1, H, W]

        # マスク領域で image と back_image をブレンディング
        blended = (1 - alpha) * image + alpha * back_image

        # mask を使って元の画像に合成
        result = mask * blended + (1 - mask) * image

        return result


    def make_anomaly_bend(self, image, mask, points, angles, back_image, flow, random_mask):
        """
        Perform masked background replacement, rotate the image with distance-based angles, 
        and reapply the rotated mask areas.
        Args:
            image: Tensor of shape [batch_size, 3, 256, 256]
            mask: Tensor of shape [batch_size, 256, 256, 1]
            points: Tensor of shape [batch_size, 2] (rotation centers for each batch)
            angles: Tensor of shape [batch_size] (base rotation angles in radians)
            back_image: Tensor of shape [batch_size, 3, 256, 256]
            flow: Tensor of shape [batch_size, 256, 256, 2] (additional noise flow)
        Returns:
            output_image: Tensor of shape [batch_size, 3, 256, 256]
            rotated_mask: Tensor of shape [batch_size, 1, 256, 256]
            final_flow: Tensor of shape [batch_size, 2, 256, 256]
        """

        batch_size, _, height, width = image.shape

        # Create a grid of normalized coordinates [-1, 1]
        base_grid = torch.meshgrid(torch.linspace(-1, 1, height, device=image.device),
                                torch.linspace(-1, 1, width, device=image.device))
        base_grid = torch.stack((base_grid[1], base_grid[0]), 2)  # (x, y)
        base_grid = base_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch_size, height, width, 2]

        # Normalize points to [-1, 1]
        points = torch.stack(points).float().to(image.device)
        points = 2.0 * points / torch.tensor([width - 1, height - 1], device=image.device) - 1.0  # [batch_size, 2]

        # Compute distance from each pixel to the rotation center
        distances = torch.sqrt((base_grid[..., 0] - points[:, 0].view(-1, 1, 1))**2 +
                            (base_grid[..., 1] - points[:, 1].view(-1, 1, 1))**2)  # [batch_size, height, width]
        distances[random_mask] = torch.ones_like(mask).squeeze(-1)[random_mask]
        
        masked_distances = distances * mask.squeeze(-1)  # [batch_size, height, width]

        # max_distance = masked_distances.max(dim=2)[0].max(dim=1)[0]
        max_distance = torch.max(masked_distances.view(batch_size, -1), dim=1).values
        max_distance = torch.where(max_distance == 0, torch.tensor(1.0, device=max_distance.device), max_distance)

        # Scale angles based on distance
        scaled_angles = angles.view(-1, 1, 1) * distances / max_distance.view(-1, 1, 1) # [batch_size, height, width]

        # Create empty grid to store the transformed coordinates
        transformed_grid = base_grid.clone()

        # Apply rotation per pixel by using the distance-dependent angle
        cos_vals = torch.cos(scaled_angles)
        sin_vals = torch.sin(scaled_angles)

        # Apply the rotation transformation to the grid
        transformed_grid[..., 0] = base_grid[..., 0] * cos_vals - base_grid[..., 1] * sin_vals
        transformed_grid[..., 1] = base_grid[..., 0] * sin_vals + base_grid[..., 1] * cos_vals

        # Step 1: Replace masked area with background
        mask = mask.permute(0, 3, 1, 2)  # [batch_size, 1, 256, 256]
        replaced_image = image * (1 - mask) + back_image * mask

        # Step 2: Apply the flow noise
        transformed_grid = transformed_grid + flow  # Add flow noise

        # Step 3: Use grid_sample for image and mask
        rotated_image = torch.nn.functional.grid_sample(image, transformed_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        rotated_mask = torch.nn.functional.grid_sample(mask.float(), transformed_grid, mode='nearest', padding_mode='zeros', align_corners=False)

        # Step 4: Combine rotated areas with replaced image
        output_image = replaced_image * (1 - rotated_mask) + rotated_image * rotated_mask

        # Step 5: Compute final flow
        final_flow = (transformed_grid - base_grid).permute(0, 3, 1, 2) * rotated_mask

        return output_image, rotated_mask, final_flow


    def warping_denoising_ite_eval(self, model, x_t, t_distance, args, object_mask):
        seq = [x_t.cpu().detach()]
        seq_x_0 = [x_t.cpu().detach()]
        
        object_mask = (object_mask > 0).float().unsqueeze(3)  # Trueの場合は1、Falseの場合は0に変換 
        object_mask = self.apply_dilation(object_mask).permute(0,3,1,2)

        if args['subclass'] == 'clip': 
            x_t = torch.zeros_like(x_t) + x_t * object_mask.repeat(1,3,1,1)

        seq = [x_t.cpu().detach()]
        seq_x_0 = [x_t.cpu().detach()]

        x_T = x_t.clone()
        object_mask_T = object_mask.clone()
        flow = torch.zeros_like(x_t)[:,:2,:,:]
        flow_list = []

        for t in range(int(t_distance) - 1, -1, -1):
            t_batch = torch.tensor([t], device=x_t.device).repeat(x_t.shape[0])

            with torch.no_grad():
                estimate = model(x_t, t_batch)
                estimate_x_0 = estimate[:,:3,:,:]
                estimate_flow_x_t_from_x_0 = estimate[:,3:,:,:]

                # 大きさの計算
                magnitude = torch.norm(estimate_flow_x_t_from_x_0, dim=1)  # dim=0で各ピクセルの大きさを計算
                # 閾値を設定
                threshold = (3.0/(int(t_distance) - t) + 1.0) / (128*5) # 例: 0.5という閾値

                flow_zero_mask = (magnitude < threshold).unsqueeze(0)
                # 大きさが閾値以下のフローを0に設定
                estimate_flow_x_t_from_x_0[flow_zero_mask.repeat(1,2,1,1)] = 0
                flow_list.append(estimate_flow_x_t_from_x_0)

                estimate_x_t_1, estimate_mask_x_t_1  = self.make_pre_image(x_t, estimate_flow_x_t_from_x_0, t_batch, object_mask)
                estimate_x_t_1 = self.interporate_pre_image(estimate_x_t_1,estimate_mask_x_t_1,estimate_x_0, object_mask)
                x_t = estimate_x_t_1
                object_mask = estimate_mask_x_t_1

                seq_x_0.append(estimate_x_0.cpu().detach())
            seq.append(x_t.cpu().detach())

        flow_all = torch.zeros_like(x_t)[:,:2,:,:]

        for t in range(int(t_distance)):
            flow = flow_list[-t]/(t + 1)
            flow_x = flow[:, 0, :, :]
            flow_y = flow[:, 1, :, :]
            magnitude = torch.sqrt(flow_x**2 + flow_y**2)
            mask = (magnitude > 0).unsqueeze(1).float()
            flow = flow*mask

            pre_mask, _ = self.make_pre_image(mask, flow, torch.ones_like(t_batch, dtype=torch.int), mask)
            flow_all = self.backwarp(flow_all,flow) * ((mask == 1) | (pre_mask == 0)) + flow

        flow_all = flow_all * object_mask_T
    

        x_0, object_mask  = self.make_pre_image(x_T, flow_all, torch.ones_like(t_batch, dtype=torch.int), object_mask_T)
        flow_x = flow_all[:, 0, :, :]
        flow_y = flow_all[:, 1, :, :]
        magnitude = torch.sqrt(flow_x**2 + flow_y**2)
        mask = (magnitude > 0).unsqueeze(1).float()
        flow_all_inverce, _  = self.make_pre_image(flow_all, flow_all, torch.ones_like(t_batch, dtype=torch.int), mask)
        return estimate_x_0, seq, seq_x_0, flow_all, flow_all_inverce, x_T