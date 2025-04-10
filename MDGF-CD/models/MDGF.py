
from models.SCAM import  SCAM
from models.modules import UpsampleConvLayer, ResidualBlock, ConvLayer, GlobalQueryGen, to_shape, to_the_shape
from models.FGFM import  FMM, HFFM, LFFM

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

try:
    import os, sys
    kernel_path = os.path.abspath(os.path.join('..'))
    sys.path.append(kernel_path)
    from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

except:
    WindowProcess = None
    WindowProcessReverse = None
    print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")


class PatchEmbed(nn.Module):

    def __init__(self, img_size=256, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging(nn.Module):

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DWindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, f, mask=None):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(f).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GDWindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, f, q_global, mask=None, glo_msa=False):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(f).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        if glo_msa:
            B = q_global.shape[0]
            B_dim = torch.div(B_, B, rounding_mode='floor')
            q_global = q_global.repeat(1, B_dim, 1, 1, 1)
            q_global = q_global.reshape(B_, self.num_heads, N, C // self.num_heads)
            q_global = q_global * self.scale
            attn1 = (q_global @ k.transpose(-2, -1))
        else:
            attn1 = 0

        q = q * self.scale
        attn2 = (q @ k.transpose(-2, -1))

        attn = attn1 + attn2

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DGDWindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dwmsa1 = DWindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.dwmsa2 = DWindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.gdwmsa1 = GDWindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.gdwmsa2 = GDWindowAttention(dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)

    def forward(self, x, y, f, x_q, y_q, mask=None, glo_msa = False):
        x = self.dwmsa1(x, f)
        y = self.dwmsa2(y, f)
        f1 = torch.abs(x-y)
        x = self.gdwmsa1(x, f1, x_q, mask, glo_msa)
        y = self.gdwmsa2(y, f1, y_q, mask, glo_msa)
        return x, y


class DiffFormerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, glo_msa=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm11 = norm_layer(dim)
        self.norm12 = norm_layer(dim)
        self.attn = DGDWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.glo_msa = glo_msa

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm21 = norm_layer(dim)
        self.norm22 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, y, q_global1, q_global2):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut1 = x
        x = self.norm11(x)
        x = x.view(B, H, W, C)
        shortcut2 = y
        y = self.norm12(y)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
                y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
            else:
                x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
                y_windows = WindowProcess.apply(y, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            shifted_y = y
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        f_windows = torch.abs(x_windows - y_windows)
        # DW-MSA/GDW-MSA
        x_attn_windows , y_attn_windows = self.attn(x_windows, y_windows, f_windows, q_global1, q_global2, mask=self.attn_mask, glo_msa=self.glo_msa)  # nW*B, window_size*window_size, C

        # merge windows
        x_attn_windows = x_attn_windows.view(-1, self.window_size, self.window_size, C)
        y_attn_windows = y_attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(x_attn_windows, self.window_size, H, W)  # B H' W' C
                shifted_y = window_reverse(y_attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(x_attn_windows, B, H, W, C, self.shift_size, self.window_size)
                y = WindowProcessReverse.apply(y_attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(x_attn_windows, self.window_size, H, W)  # B H' W' C
            shifted_y = window_reverse(y_attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)
        x = shortcut1 + self.drop_path1(x)
        y = shortcut2 + self.drop_path2(y)

        # FFN
        x = x + self.drop_path1(self.mlp1(self.norm21(x)))
        y = y + self.drop_path2(self.mlp2(self.norm22(y)))

        return x, y


class BasicLayer(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, img_size=256, use_scam=True,
                 embed_dim=96,mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.embed_dim = embed_dim
        self.use_scam = use_scam
        self.num_heads = num_heads

        # build blocks
        self.blocks = nn.ModuleList([
            DiffFormerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 glo_msa=False if (i % 2 == 0) else True,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        self.q_global_gen1 = GlobalQueryGen(dim, input_resolution[0], img_size, window_size, num_heads)
        self.q_global_gen2 = GlobalQueryGen(dim, input_resolution[0], img_size, window_size, num_heads)

        if use_scam:
            self.scam = SCAM(self.dim, self.embed_dim)
        else:
            self.scam = None

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x1, x2, outs1, outs2, scamout):
        q_global1 = self.q_global_gen1(to_shape(x1, self.input_resolution))
        q_global2 = self.q_global_gen2(to_shape(x2, self.input_resolution))

        for blk in self.blocks:
            if self.use_checkpoint:
                x1 = checkpoint.checkpoint(blk, x1)
                x2 = checkpoint.checkpoint(blk, x2)
            else:
                x1, x2 = blk(x1, x2, q_global1, q_global2)

        if self.use_scam:
            f1 = self.scam(to_shape(x1, self.input_resolution))
            f2 = self.scam(to_shape(x2, self.input_resolution))
            f = torch.abs(f1 - f2)
            scamout.append(f)

        outs1.append(to_the_shape(x1))
        outs2.append(to_the_shape(x2))

        if self.downsample is not None:
            x1 = self.downsample(x1)
            x2 = self.downsample(x2)
        return  x1, x2, outs1, outs2, scamout


class MDGFcd(nn.Module):

    def __init__(self, img_size=256, patch_size=4, in_chans=3, num_classes=2,
                 embed_dim=96, depths=[6, 6, 6, 6 ], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                              input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                patches_resolution[1] // (2 ** i_layer)),
                                              depth=depths[i_layer],
                                              num_heads=num_heads[i_layer],
                                              window_size=window_size,
                                              use_scam=True if (i_layer < 1) else False,
                                              mlp_ratio=self.mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop_rate, attn_drop=attn_drop_rate,
                                              drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                              norm_layer=norm_layer,
                                              downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                              use_checkpoint=use_checkpoint,
                                              fused_window_process=fused_window_process)

            self.layers.append(layer)

        self.scam = SCAM(self.embed_dim, self.embed_dim)
        self.shallow_fuse = LFFM(self.embed_dim)
        self.deep_fuse = HFFM()
        self.final_fuse = FMM(self.embed_dim)

        # Final predction head
        self.convd2x = UpsampleConvLayer(self.embed_dim, self.embed_dim, kernel_size=4, stride=2)
        self.dense_2x = nn.Sequential(ResidualBlock(self.embed_dim))
        self.convd1x = UpsampleConvLayer(self.embed_dim, self.embed_dim, kernel_size=4, stride=2)
        self.dense_1x = nn.Sequential(ResidualBlock(self.embed_dim))

        self.change_probability = ConvLayer(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x1, x2):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        if self.ape:
            x1 = x1 + self.absolute_pos_embed
            x2 = x2 + self.absolute_pos_embed
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)

        scamout = []
        f1 = self.scam(to_the_shape(x1))
        f2 = self.scam(to_the_shape(x2))
        f = torch.abs(f1 - f2)
        scamout.append(f)

        outs1 = []
        outs2 = []
        for layer in self.layers:
            x1, x2, outs1, outs2, scamout = layer(x1, x2, outs1, outs2, scamout)

        c4 = torch.abs(outs1[3] - outs2[3])
        c3 = torch.abs(outs1[2] - outs2[2])
        c2 = torch.abs(outs1[1] - outs2[1])
        c1 = torch.abs(outs1[0] - outs2[0])

        out1 = self.shallow_fuse(scamout[0], c1, scamout[1])
        out2 = self.deep_fuse(c4, c3, c2)
        x = self.final_fuse(out1, out2)

        return x

    def forward(self, x1, x2):
        x = self.forward_features(x1, x2)

        x = self.convd2x(x)
        x = self.dense_2x(x)
        x = self.convd1x(x)
        x = self.dense_1x(x)

        x = self.change_probability(x)

        return x