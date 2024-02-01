if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from torch import nn
import timm
from timm.models._manipulate import checkpoint_seq
from utils import gumbel_topk


TIMM_DINOV2_ARCHS = {
    'small' : 384,
    'base'  : 768,
    'large' : 1024,
    'giant' : 1536,
}

## dyvit에서 가져옴
class Predictor(nn.Module):
    def __init__(self, num_patch, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patch = num_patch
        self.in_conv = nn.Sequential(
            # nn.LayerNorm(self.embed_dim * self.num_patch * 2),
            nn.Linear(self.embed_dim * self.num_patch, self.num_patch),
            # nn.Tanh()
        )

    def forward(self, x):
        # x1, x2 ---> B, NP, DIM
        B, NP, DIM = x.shape
        x = x.reshape(B, NP * DIM)
        x = self.in_conv(x) # x ---> B* NP
        # softmax를 한 번 거쳐서 out을 해야할지...?
        x = x.reshape(B, NP)
        return x


class Attention(timm.models.vision_transformer.Attention):
    def forward(self, x: torch.Tensor, acquire_attn: bool=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_drop = self.attn_drop(attn)
        x = attn_drop @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if acquire_attn:
            return x, attn
        else:
            return x


class Block(timm.models.vision_transformer.Block):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: float|None = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = timm.models.vision_transformer.Mlp,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = timm.models.vision_transformer.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = timm.models.vision_transformer.DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = timm.models.vision_transformer.LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = timm.models.vision_transformer.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward_attn_res(self, x: torch.Tensor, acquire_attn: bool=False) -> torch.Tensor:
        if acquire_attn:
            x, attn = self.attn(self.norm1(x), acquire_attn=True)
            x = self.ls1(x)
            return x + self.drop_path1(x), attn
        else:
            x, attn = self.attn(self.norm1(x), acquire_attn=True)
            x = self.ls1(x)
            return x + self.drop_path1(x)
    
    def forward_mlp_res(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    
    def forward(self, x: torch.Tensor, acquire_attn: bool=False) -> torch.Tensor:
        if acquire_attn:
            x, attn = self.forward_attn_res(x, acquire_attn=True)
            return self.forward_mlp_res(x), attn
        else:
            x = self.forward_attn_res(x, acquire_attn=False)
            return self.forward_mlp_res(x)


class EffDINOv2(timm.models.VisionTransformer):
    def __init__(
            self,
            model_name: str='small', 
            img_size: int=224,
            masked_block: int=9,
            masking_rate: float=0.4, 
            return_token: bool=False,
            norm_layer: bool=False,
            dino_v2_pretrained: bool=True,
    ):
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = 14
        self.embed_dim = TIMM_DINOV2_ARCHS[model_name]
        self.masked_block = masked_block
        self.num_patches = (img_size // 14)**2  # the number of patches except for the cls token.
        self.return_token = return_token
        self.norm_layer = norm_layer
        
        if not (0 <= masking_rate < 1):
            raise ValueError(f'Masking rate must be in [0, 1).')
        self.kept_patches = int((self.num_patches - int(self.num_patches * masking_rate)) ** 0.5)
        self.num_masks = int(self.num_patches - (self.kept_patches ** 2))
        self.masking_rate = self.num_masks / self.num_patches
        self.kept_patches_row = int((self.num_patches - self.num_masks) ** 0.5)
        
        super(EffDINOv2, self).__init__(
            img_size=img_size, 
            patch_size=self.patch_size, 
            num_classes=0, 
            init_values=1.0, 
            embed_dim=self.embed_dim, 
            block_fn=Block, 
        )

        
        if dino_v2_pretrained:
            pretrained_name = f'vit_{self.model_name}_patch14_dinov2.lvd142m'
            state_dict = timm.create_model(pretrained_name, pretrained=True, num_classes=0).state_dict()
            state_dict['pos_embed'] = self.pos_embed
            self.load_state_dict(state_dict)
        
        self.predictor = Predictor(num_patch=self.num_patches , embed_dim=self.embed_dim)
        
    ''' Forward Tree 
        forward 
            - forward_features 
                - forward_pre 
                - prune_dissimilar (optional)
                - forward_in 
                - forward_post 
            - forward_head 
    ''' 
    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks[:self.masked_block], x)
        else:
            x = self.blocks[:self.masked_block](x)
        return x
    
    def prune_dissimm(
        self, 
        x: torch.Tensor, 
        simm: torch.Tensor or None 
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        '''
        x1, x2: B, NP+1, DIM
        '''
        batch_size, num_patches, feat_dim = x.shape
        num_patches -= 1
        
        t, f = x[:, 0, None], x[:, 1:] # f1 --> B/2, NP, DIM
        
        num_keeps = num_patches - self.num_masks
        
        # Predictor
        pred_simm = self.predictor.forward(f)  #...............................| B, NP
        pred_simm = pred_simm.unsqueeze(-1)  #.................................| B, NP, 1
        
        if simm is not None: # simm을 계산한 Training phase의 경우
            mask_hard = gumbel_topk(simm, k=num_keeps, dim=1)#................| B, NP, 1
        else:
            mask_hard = gumbel_topk(pred_simm, k=num_keeps, dim=1)
        masked_f = f * mask_hard 
        
        indices = mask_hard.detach().bool()  # ................| B, NP, 1
        indices = indices.expand_as(masked_f)  # .............| B, NP, DIM
        masked_f = masked_f[indices].reshape(batch_size, num_keeps, feat_dim)
        
        out = torch.cat([t, masked_f], dim=1)

        return out, pred_simm # .......| B, NP, 1
    
    def calc_cosine(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        '''
        x1, x2: B, NP+1, DIM
        only training.
        '''
        if x1.shape != x2.shape:
            raise RuntimeError('The shape of the two tensors must equal.')
        
        t1, f1 = x1[:, 0, None], x1[:, 1:]
        t2, f2 = x2[:, 0, None], x2[:, 1:] # f1 --> B/2, NP, DIM
        
        simm = torch.nn.functional.cosine_similarity(f1, f2, dim=2)  #...| B/2, NP
        simm = torch.cat([simm, simm], dim=0) #...| B, NP
        
        return simm.unsqueeze(-1) # .......| B, NP, 1
    
    def predict_cosine(
        self, 
        x: torch.Tensor,
    ):
        t, f = x[:, 0, None], x[:, 1:] # f --> B, NP, DIM
        # Predictor
        pred_simm = self.predictor.forward(f)  #...............................| B, NP
        pred_simm = pred_simm.unsqueeze(-1)  #.................................| B, NP, 1
        
        # masked_f = f * pred_simm
        # x = torch.cat([t, masked_f], dim=1)
        
        return pred_simm
    
    def forward_in(self, x: torch.Tensor) -> torch.Tensor: 
        return self.blocks[self.masked_block](x, acquire_attn=True)
    
    def forward_post(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks[self.masked_block+1:](x))

    def forward_features(self, x: torch.Tensor, calc_cosine: bool=True) -> torch.Tensor: 
        x = self.forward_pre(x)
        if calc_cosine: # it will be activated at training 
            batch_size = x.size(0)
            if batch_size % 2 != 0:
                raise RuntimeError('The batch size must be even to prune by similarity between domains.')
            half_size = batch_size // 2
            x1, x2 = torch.split(x, (half_size, half_size), dim=0)
            simm = self.calc_cosine(x1, x2)
        else:
            simm = None
            
        # if prune:
        #     x, pred_simm = self.prune_dissimm(x, simm)
        pred_simm = self.predict_cosine(x)
        
        if simm is not None:
            t, f = x[:, 0, None], x[:, 1:]
            # weight = (simm + 1)/2
            f = f * simm
            x = torch.cat([t, f], dim=1)
        else:
            t, f = x[:, 0, None], x[:, 1:]
            # weight = (pred_simm + 1)/2
            f = f * pred_simm
            x = torch.cat([t, f], dim=1)
            
        x, attn = self.forward_in(x)
        return self.forward_post(x), attn, pred_simm, simm
        
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_layer:
            x = self.fc_norm(x)
        x = self.head_drop(x)
        
        t = x[:, 0]  # class token
        f = x[:, 1:]  # feature tokens
        
        if f.numel() == (f.size(0) * self.kept_patches_row * self.kept_patches_row * self.embed_dim):
            f = f.reshape(
                f.size(0), 
                self.kept_patches_row,
                self.kept_patches_row,
                self.embed_dim,
            ).permute(0, 3, 1, 2)
        else:
            num_patches_row = self.img_size // self.patch_size
            f = f.reshape(
                f.size(0),
                num_patches_row,
                num_patches_row,
                self.embed_dim,
            ).permute(0, 3, 1, 2)
        
        if self.return_token:
            return f, t
        else:
            return f, None
        
    def forward(self, x: torch.Tensor, calc_cosine: bool=True) -> torch.Tensor:
        feats, attn, pred_simm, simm = self.forward_features(x, calc_cosine=calc_cosine)
        f, t = self.forward_head(feats)
        return f, t, feats, attn, pred_simm, simm


if __name__ == '__main__':
    import timm
    DEVICE = 0
    
    sample = torch.randn(4, 3, 224, 224).cuda(DEVICE)
    net = EffDINOv2(return_token=True).cuda(DEVICE)
    
    f, t, feats, attn, pred_simm, simm = net(sample, prune=True)
    print(f'{f.shape=}')
    print(f'{t.shape=}')
    print(f'{attn.shape=}')
    print(f'{pred_simm.shape=}')
    print(f'{simm.shape=}')
    
    out_pre = net.forward_pre(sample)
    print(f'{out_pre.shape=}')
    
    out_pre_lhs, out_pre_rhs = torch.split(out_pre, (2, 2), dim=0)
    out_prn = net.prune_dissimilar(out_pre_lhs, out_pre_rhs)
    print(f'{out_prn[2].shape=}')
    out_prn = torch.cat((out_prn[0], out_prn[1]), dim=0)
    print(f'{out_prn.shape=}')
    
    out_in, out_attn = net.forward_in(out_prn)
    print(f'{out_in.shape=}')
    print(f'{out_attn.shape=}')
    
    out_post = net.forward_post(out_in)
    print(f'{out_post.shape=}')
    
    out_head_f, out_head_t = net.forward_head(out_post)
    print(f'{out_head_f.shape=}')
    print(f'{out_head_t.shape=}')
