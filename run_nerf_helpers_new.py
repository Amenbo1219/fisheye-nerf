import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
            # self.r_linear = nn.Linear(W//2, 1)
            # self.g_linear = nn.Linear(W//2, 1)
            # self.b_linear = nn.Linear(W//2, 1)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)
            rgb = self.rgb_linear(h)
            # Split_RGB
            # r = self.r_linear(h)
            # g = self.g_linear(h)
            # b = self.b_linear(h)
            # rgb = torch.cat([r,g,b],-1)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))
def get_rays_fisyeye(H, W, K, c2w,f_eq=302):
    # Rays_Dは光線のθφω(向き情報)，これはあくまでワールド座標系
    # Rays_Oは光線のxzy位置，これもワールド座標系
    # Dirs:W,Hの要素を取り出して，各配列に入れて，レンズの歪みを加算したもの
    # 光線の一つなので，θにはWの焦点距離の影響，φにはHの焦点距離の影響，ωは1が移入される．
    # 球面座標系でのサンプリング

    cx, cy = W / 2, H / 2
    
    # 画像座標の生成
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), 
                          torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # 画像平面上の半径 r (中心からの距離)
    r = torch.sqrt((i - cx) ** 2 + (j - cy) ** 2)
    
    # 等距離モデル: θ = r / f
    theta = r / f_eq  # [H, W]

    # 光線の方向を計算
    x = torch.sin(theta) * (i - cx) / (r + 1e-6)  # 0割防止
    y = torch.sin(theta) * (j - cy) / (r + 1e-6)
    z = torch.cos(theta)

    # レイ方向ベクトル
    # dirs = np.stack([x, -y, -z], -1)  # [H, W, 3]
    dirs = torch.stack([-1*x,-1*y,-1*z], -1)  # [H, W, 3]

    # カメラ座標系からワールド座標系に変換
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)  # 回転行列適用

    # 光線の原点 (全てカメラの原点)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    pixel_coords = torch.stack([i, j], axis=-1)  # [H, W, 2]

    return rays_o, rays_d, pixel_coords

def get_rays_np_fisyeye(H, W, K, c2w):
    # 画像の中心
    cx, cy = W / 2, H / 2
    f_eq = K[0][0]
    
    # 画像座標の生成
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), 
                       np.arange(H, dtype=np.float32), indexing='xy')
    
    # 画像平面上の半径 r (中心からの距離)
    r = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)

    # 等距離モデル: θ = r / f
    theta = r / f_eq  # [H, W]

    # 光線の方向を計算
    x = np.sin(theta) * (i - cx) / (r + 1e-6)  # 0割防止
    y = np.sin(theta) * (j - cy) / (r + 1e-6)
    z = np.cos(theta)

    # レイ方向ベクトル
    # dirs = np.stack([x, -y, -z], -1)  # [H, W, 3]
    dirs = torch.stack([-1*x,-1*y,-1*z], -1)  # [H, W, 3]
    # dirs = np.stack([x, y, z], -1)  # [H, W, 3]

    # カメラ座標系からワールド座標系に変換
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], axis=-1)  # 回転行列適用

    # 光線の原点 (全てカメラの原点)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape)
    pixel_coords = torch.stack([i, j], axis=-1)  # [H, W, 2]

    return rays_o, rays_d, pixel_coords

# def get_rays_sp(H, W, K, c2w):
#     # Rays_Dは光線のθφω(向き情報)，これはあくまでワールド座標系
#     # Rays_Oは光線のxzy位置，これもワールド座標系
#     # Dirs:W,Hの要素を取り出して，各配列に入れて，レンズの歪みを加算したもの
#     # 光線の一つなので，θにはWの焦点距離の影響，φにはHの焦点距離の影響，ωは1が移入される．
#     # 球面座標系でのサンプリング

#     i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing='xy')
#     u = (2 * i / W) - 1 # -1→w→.1 に拡大
#     v = (2 * j / H) -1  # 垂直軸
#     theta = u * torch.pi # -pi→w→+pi に拡大
#     phi = v * (torch.pi / 2) # pi/2→h→-pi/2 に拡大

#     x = torch.sin(theta) * torch.cos(phi)
#     y = torch.cos(theta) * torch.cos(phi)
#     z = torch.sin(phi)

#     dirs = torch.stack([x, y, z], -1)  # [H, W, 3]
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # rays_d = torch.sum(dirs[..., np.newaxis, :] , -2)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d


# def get_rays_np_sp(H, W, K, c2w):
    
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     u = (2 * i / W) - 1 # -1→w→.1 に拡大
#     v = (2 * j / H) -1  # 垂直軸
#     theta = u * np.pi # -pi→w→+pi に拡大
#     phi = v * (np.pi / 2) # pi/2→h→-pi/2 に拡大

#     x = np.sin(theta) * np.cos(phi)
#     y = np.cos(theta) * np.cos(phi)
#     z = np.sin(phi)

#     # レイの方向ベクトルを作成
#     dirs = np.stack([x, y, z], -1)  # [H, W, 3]
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # rays_d = np.sum(dirs[..., np.newaxis, :] , -2)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d

# # Ray helpers
# def get_rays_roll(H, W, K, c2w):
#     # Rays_Dは光線のθφω(向き情報)，これはあくまでワールド座標系
#     # Rays_Oは光線のxzy位置，これもワールド座標系
#     # Dirs:W,Hの要素を取り出して，各配列に入れて，レンズの歪みを加算したもの
#     # 光線の一つなので，θにはWの焦点距離の影響，φにはHの焦点距離の影響，ωは1が移入される．
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
#     i = i.t()
#     j = j.t()

#     # ピクセル座標を[-1, 1]に正規化
#     u = (2 * i / W) - 1  # 水平軸
#     v = (2 * j / H) -1  # 垂直軸

#     # 角度と高さに変換
#     theta = u * torch.pi  # 水平角
#     phi = v  # 高さ（-1から1の範囲）

#     x = torch.sin(theta)
#     y = torch.cos(theta)
#     z = phi

#     dirs = torch.stack([x,y,z], dim=-1) # del360-v3
#     # dirs = torch.stack([torch.cos(phi),-torch.cos(theta)*torch.sin(phi), -torch.cos(theta)*-torch.sin(phi)], -1) # del360-v2
#     # i, j = torch.meshgrid(torch.linspace(0, THETA-1, THETA), torch.linspace(0, PHI-1, PHI))  # pytorch's meshgrid has indexing='ij'
#     # dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  
#     # Rotate ray directions from camera frame to the world frame
#     # rays_d = torch.sum(dirs[..., np.newaxis, :] , -2)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d


# def get_rays_np_roll(H, W, K, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')


#     # ピクセル座標を[-1, 1]に正規化
#     u = (2 * i / W) - 1  # 水平軸
#     v = (2 * j / H) -1  # 垂直軸

#     # 角度と高さに変換
#     theta = u * np.pi  # 水平角
#     phi = v  # 高さ（-1から1の範囲）

#     x = np.sin(theta)
#     y = np.cos(theta)
#     z = phi
#     dirs = np.stack([x,y,z], -1) # del360-v3
#  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1) 
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d

# # Ray helpers
# def get_rays(H, W, K, c2w):
#     # Rays_Dは光線のθφω(向き情報)，これはあくまでワールド座標系
#     # Rays_Oは光線のxzy位置，これもワールド座標系
#     # Dirs:W,Hの要素を取り出して，各配列に入れて，レンズの歪みを加算したもの
#     # 光線の一つなので，θにはWの焦点距離の影響，φにはHの焦点距離の影響，ωは1が移入される．
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
#     i = i.t()
#     j = j.t()
#     theta = (i/W)*2*torch.pi # del360-v2
#     phi = (j/H-1/2)*2*torch.pi # del360-v2
#     x = np.cos(theta) * np.cos(phi)
#     y = np.cos(phi) * np.sin(theta)
#     z = np.sin(phi)

#     dirs = torch.stack([x,y,z], -1) # del360-v3
#     # i, j = torch.meshgrid(torch.linspace(0, THETA-1, THETA), torch.linspace(0, PHI-1, PHI))  # pytorch's meshgrid has indexing='ij'
#     # dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)

#     # Rotate ray directions from camera frame to the world frame
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # rays_d = torch.sum(dirs[..., np.newaxis, :] , -2)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d



# def get_rays_np(H, W, K, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     theta = (i/W)*2*np.pi # del360-v2
#     phi = (j/H)*2*np.pi # del360-v2
#     x = np.cos(theta) * np.cos(phi)
#     y = np.cos(phi) * np.sin(theta)
#     z = np.sin(phi)
    
#     dirs = np.stack([x,y,z], -1) # del360-v3
#     # dirs = np.stack([np.asin(n_i/torch.cos(n_j)),-np.ones_like(i),np.asin(n_j),], -1) # del360-v3
#     # dirs[np.signbit(dirs)] = 1 # del360-v3
#     # dirs = np.stack([(i-K[0][2])/K[0][0]*np.pi, -(j-K[1][2])/K[1][1]*np.pi/2, -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    
#     # rays_d = np.sum(dirs[..., np.newaxis, :] , -2)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d
# def get_rays(H, W, K, c2w):
#     # Rays_Dは光線のθφω(向き情報)，これはあくまでワールド座標系
#     # Rays_Oは光線のxzy位置，これもワールド座標系
#     # Dirs:W,Hの要素を取り出して，各配列に入れて，レンズの歪みを加算したもの
#     # 光線の一つなので，θにはWの焦点距離の影響，φにはHの焦点距離の影響，ωは1が移入される．
#     i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
#     i = i.t()
#     j = j.t()
#     # print(i,j)
#     dirs = torch.stack([(i-K[0][2])/K[0][0]*torch.sin(i*np.pi), -(j-K[1][2])/K[1][1]*torch.sin(j*np.pi), -torch.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     # x_rotate =c2w[:3,:3][]
#     rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3] , -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = c2w[:3,-1].expand(rays_d.shape)
#     return rays_o, rays_d


# def get_rays_np(H, W, K, c2w):
#     i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
#     dirs = torch.stack([(i-K[0][2])/K[0][0]*np.sin(i*np.pi), -(j-K[1][2])/K[1][1]*np.sin(j*np.pi), -torch.ones_like(i)], -1)
#     # print(i,j)
#     # dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
#     # Rotate ray directions from camera frame to the world frame
#     rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
#     # Translate camera frame's origin to the world frame. It is the origin of all rays.
#     rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
#     return rays_o, rays_d

def ndc_rays_pinhole(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    # 近接平面に光線原点をシフト
    # 1. 原点（rays_o）を near 平面に投影
    rays_o = rays_o + near * rays_d
    return rays_o, rays_d
# Hierarchical sampling (section 5.2)
# def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
#     # Get pdf
#     weights = weights + 1e-5 # prevent nans
#     pdf = weights / torch.sum(weights, -1, keepdim=True)
#     cdf = torch.cumsum(pdf, -1)
#     cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

#     # Take uniform samples
#     if det:
#         u = torch.linspace(0., 1., steps=N_samples)
#         u = u.expand(list(cdf.shape[:-1]) + [N_samples])
#     else:
#         u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

#     # Pytest, overwrite u with numpy's fixed random numbers
#     if pytest:
#         np.random.seed(0)
#         new_shape = list(cdf.shape[:-1]) + [N_samples]
#         if det:
#             u = np.linspace(0., 1., N_samples)
#             u = np.broadcast_to(u, new_shape)
#         else:
#             u = np.random.rand(*new_shape)
#         u = torch.Tensor(u)

#     # Invert CDF
#     u = u.contiguous()
#     inds = torch.searchsorted(cdf, u, right=True)
#     below = torch.max(torch.zeros_like(inds-1), inds-1)
#     above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
#     inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

#     # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
#     matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
#     cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
#     bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

#     denom = (cdf_g[...,1]-cdf_g[...,0])
#     denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
#     t = (u-cdf_g[...,0])/denom
#     samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

#     return samples

def sample_pdf(bins, weights, N_samples, det=False, pytest=False, depth_bias=True):
    # Prevent nans
    weights = weights + 1e-5  # avoid zero-division or NaN

    # --- ① オプション：深度バイアス（近距離を強調） ---
    if depth_bias:
        # bins: [N_rays, N_bins+1] → 中心点 [N_rays, N_bins]
        depth_center = 0.5 * (bins[..., 1:] + bins[..., :-1])
        bias = 1.0 / (depth_center + 1e-6)  # closer = higher weight
        weights = weights * bias

    # Normalize to get pdf
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (N_rays, N_bins+1)

    # Sample u
    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        # 非線形分布で近くを高密度に（オプション）
        u = u ** 2  # 近距離を強調
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)

    # Pytest用（numpy固定値）
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.tensor(u, dtype=torch.float32, device=weights.device)

    # CDFの逆関数でサンプル生成
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)  # (N_rays, N_samples, 2)

    # Gather cdf & bins
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    # 線形補間でサンプル位置を求める
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
def sample_pdf_solid_angle(bins, weights, ray_pixel_coords, cx, cy, f_eq, N_samples, det=True):
    """
    Args:
        bins: [N_rays, N_bins+1] - サンプル間の深度区間
        weights: [N_rays, N_bins] - 各区間の重み
        ray_pixel_coords: [N_rays, 2] - 各レイの (i, j) ピクセル座標
        cx, cy: float - 画像中心（W/2, H/2）
        f_eq: float - 等距離射影の焦点距離
        N_samples: int - サンプル数
        det: bool - deterministic sampling
    """

    # 防NaN用の微小値
    weights = weights + 1e-5

    # --- (1) Solid Angle バイアスの計算 ---
    # ピクセル距離 r = sqrt((i-cx)^2 + (j-cy)^2)
    dx = ray_pixel_coords[:, 0] - cx
    dy = ray_pixel_coords[:, 1] - cy
    r = torch.sqrt(dx ** 2 + dy ** 2) + 1e-6  # avoid div 0

    # θ = r / f_eq, bias = 1 / sin(θ)
    theta = r / f_eq
    sin_theta = torch.sin(theta)
    solid_angle_bias = 1.0 / (sin_theta + 1e-6)  # [N_rays]

    # バイアスを weights に乗算（ブロードキャスト）
    weights = weights * solid_angle_bias[:, None]  # [N_rays, N_bins]

    # --- (2) 通常の sample_pdf 流れ ---
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        u = u ** 2  # optional: u^2で近距離集中
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=weights.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
