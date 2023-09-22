import flax.linen as nn
from einops import rearrange
from typing import Callable
import jax.numpy as jnp
from functools import partial
from jax.experimental.host_callback import id_print
from dataclasses import field
from typing import List


class Residual(nn.Module):
    fn: Callable
    
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose(dim, [8], [4])


class Downsample(nn.Module):
    dim: int
    
    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, [8], [4], 2)(x)


def identity(x):
    return x


class Block(nn.Module):
    channels_out: int
    groups: int

    @nn.compact
    def __call__(self, x, scale_shift = None):
        # print("We have %d groups and dimension of %s. Output dim is %d" % (self.groups, x.shape, self.channels_out))
        N, W, C = x.shape
        x = nn.Conv(self.channels_out, [3], padding=1)(x)
        x = nn.GroupNorm(num_groups=self.groups)(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = nn.swish(x)
        x = nn.Conv(self.channels_out, [1], padding=0)(x)
        return x
        

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    channels_out: int
    groups: int = 8
    
    @nn.compact
    def __call__(self, x, time_emb = None):
        # print("Resnet with channels_out: %d, groups: %d" % (self.channels_out, self.groups))
        N, W, C = x.shape
        
        if time_emb is not None:
            assert N == time_emb.shape[0]
        
        block1 = Block(channels_out=self.channels_out, groups=self.groups)
        block2 = Block(channels_out=self.channels_out, groups=self.groups)
        mlp = (
            nn.Sequential([nn.swish, nn.Dense(self.channels_out)])
            if time_emb is not None
            else None
        )
        # print("X shape: %s" % (x.shape,))
        h = block1(x)
        assert h.shape == (N, W, self.channels_out)
        if time_emb is not None:
            time_emb = nn.Sequential([nn.swish, nn.Dense(self.channels_out)])(time_emb)
            assert time_emb.shape == (N, self.channels_out)
            h = h + time_emb[:, None, :] 
        # print("B1 X shape: %s" % (h.shape,))
        h = block2(h)
        # print("B2 X shape: %s" % (h.shape,))
        
        channels_x = x.shape[-1]
        res_conv = nn.Conv(self.channels_out, [1]) if channels_x != self.channels_out else identity
        h = (h + res_conv(x))/jnp.sqrt(2)
        # print("ENDCov: X shape: %s" % (h.shape,))
        assert h.shape == (N, W, self.channels_out)
        return h
    

class Attention(nn.Module):
    n_heads: int = 4
    dim_head: int = 32
    
    @nn.compact
    def __call__(self, x):
        b, w, c = x.shape
        hidden_dim = self.dim_head * self.n_heads
        scale = self.dim_head**-0.5
        
#         print(nn.Conv(hidden_dim * 3, [1, 1], use_bias=False)(x).shape)
        qkv = nn.Conv(hidden_dim * 3, [1], use_bias=False)(x).split(3, axis=-1)
        
        #rearrange so that we have different heads and pixels are vector per channel
        q, k, v = map(
            lambda t: rearrange(t, "b x (h c) -> b h x c", h=self.n_heads), qkv
        )
        
        q = q * scale
        #Think of the whole thing more that we have heads which are like
        #groups. Each group is a seperate feature map for each pixel.
        #Each feature map is a dim_heads dimensional vector, one value for
        #each channel. It would maybe make sense to rearrange it here
        #so that channel is the last variable (since this is the way
        #to think about it)
        
        #we now take the scalar product between the query and key
        #feature maps in each of the groups ("heads")
        #the groups therefore make us only have to take group^2 
        #scalar products instead of all of them
        #maybe each head can also just be seen as a parallel attention
        #layer.
        sim = jnp.einsum("b h i d, b h j d -> b h i j", q, k)
        
        #we take the softmax to get actual weights
        
        #THEY DO MINUS MAX HERE
        attn = nn.softmax(sim, axis=-1)
        
        #we now weight all the value vectors (think of the channels 
        #as vector entries in an dim_heads-sized vector) with the
        #above calculated attention. Therefore we get again a
        #computed value vector out. At position i d we now have
        #the dth entry of the computed value/feature-vector for
        #the ith pixel.
        #Therefore each head results in a dim_head-sized feature
        #vector per pixel
        out = jnp.einsum("b h i j, b h j k -> b h i k", attn, v)
        
        #we now again put the pixel in its x,y representation.
        #furthermore we concatenate all the feature vectors per
        #pixel from the different heads and put them at the beginning
        #again so that they look like channels.
        out = rearrange(out, "b h x d -> b x (h d)", x=w)
        
        out = nn.Conv(c, [1])(out)
        return out


class LinearAttention(nn.Module):
    n_heads: int = 4
    dim_head: int = 32
        
    @nn.compact
    def __call__(self, x):
        b, w, c = x.shape
        hidden_dim = self.dim_head * self.n_heads
        scale = self.dim_head**-0.5
        
        qkv = nn.Conv(hidden_dim * 3, [1], use_bias=False)(x).split(3, axis=-1)
        
        q, k, v = map(
            lambda t: rearrange(t, "b x (h c) -> b h x c", h=self.n_heads), qkv
        )
        
        q = nn.softmax(q, axis=-1)
        k = nn.softmax(k, axis=-2)
        q = q * scale
        context = jnp.einsum("b h n c, b h n f -> b h c f", k, v)
        out = jnp.einsum("b h k v, b h q k -> b h q v", context, q)
        out = rearrange(out, "b h x v -> b x (h v)", x=w)
        out = nn.Conv(c, [1])(out)
        return out


class SinusoidalPositionEmbeddings(nn.Module):
    dim: int
        
    @nn.compact
    def __call__(self, t):
        half_dim = self.dim // 2
        embeddings = jnp.log(10_000) / (half_dim - 1)
        embeddings = jnp.exp(-jnp.arange(half_dim) * embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=1)
        return embeddings


class PreNorm(nn.Module):
    fn: Callable
    
    @nn.compact
    def __call__(self, x):
        x = nn.GroupNorm(1)(x)
        x = self.fn(x)
        return x        
    

class Unet(nn.Module):
    dim: int
    init_dim: int = None
    out_dim: int = None
    resnet_block_groups: int = 8
    
    @nn.compact
    def __call__(self, x, t):
        c = x.shape[-1]

        init_dim = self.init_dim or self.dim // 3 * 2
        init_conv = nn.Conv(init_dim, [7], padding=3)

        dim_mults = [1,2]
        dims = [init_dim, *map(lambda m: self.dim * m, dim_mults)]
        
        block_klass = partial(ResnetBlock, groups=self.resnet_block_groups)
        time_dim = self.dim * 4
        time_mlp = nn.Sequential([
            SinusoidalPositionEmbeddings(self.dim),
            nn.Dense(time_dim),
            nn.gelu,
            nn.Dense(time_dim)
        ])
        
        downs = []
        ups = []
        num_resolutions = len(dims)-1
        
        for i in range(num_resolutions):
            is_last = i == num_resolutions-1
            dim_out = dims[i+1]
            downs.append(
                [
                    block_klass(channels_out=dim_out),
                    block_klass(channels_out=dim_out),
                    Residual(PreNorm(LinearAttention())),
                    Downsample(dim_out) if not is_last else identity
                ]
            )
            
        mid_dim = dims[-1]
        mid_block1 = block_klass(mid_dim)
        mid_attn = Residual(PreNorm(Attention()))
        mid_block2 = block_klass(mid_dim)

        
        for i in range(num_resolutions-1):
            is_last = i == num_resolutions - 1
            dim_out = dims[-(i+2)]
            
            ups.append(
                [
                    block_klass(channels_out=dim_out),
                    block_klass(channels_out=dim_out),
                    Residual(PreNorm(LinearAttention())),
                    Upsample(dim_out) if not is_last else identity
                ]
            )
            
        out_dim = self.out_dim or c
        final_conv = nn.Sequential([block_klass(self.dim), nn.Conv(out_dim, [1, 1])])

        x = init_conv(x)
        t = time_mlp(t)
        h = []
        for block1, block2, attn, downsample in downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            
        x = mid_block1(x, t)
        x = mid_attn(x)
        x = mid_block2(x, t)
        for block1, block2, attn, upsample in ups:
            hp = h.pop()
            x = jnp.concatenate([x, hp], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            
        x = final_conv(x)
        return x

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)


class SBlock(nn.Module):
    channels_out: int
    
    @nn.compact
    def __call__(self, x):
        N, W, C = x.shape
        channels_out = self.channels_out
        
        x = nn.Conv(channels_out, [3], padding=1)(x)
        assert x.shape == (N, W, channels_out)
        x = nn.swish(x)
        x = nn.Conv(channels_out, [1], padding=0)(x)
        assert x.shape == (N, W, channels_out)
        
        return x


class SResNet(nn.Module):
    channels_out: int
    
    @nn.compact
    def __call__(self, x, t):
        N, W, C = x.shape
        
        channels_out = self.channels_out
        assert channels_out == C
    
        x = SBlock(channels_out)(x)
        t = nn.Sequential([
            nn.Dense(channels_out),
            nn.gelu,
            nn.Dense(channels_out)
        ])(t)[:, None, :]
        
        assert x.shape == (N, W, channels_out)
        assert t.shape == (N, 1, channels_out)

        
        h1 = x + t
        
        h2 = SBlock(channels_out)(h1)
        
        return 1/jnp.sqrt(2) * (h2 + x)


class UnetSimple(nn.Module):

    @nn.compact
    def __call__(self, x, t):
        N, W, C = x.shape
        
        assert W % 4 == 0
        
        init_dim = 3
        
        t1 = nn.Sequential([
            SinusoidalPositionEmbeddings(init_dim),
            nn.Dense(init_dim),
            nn.gelu,
            nn.Dense(init_dim)
        ])(t)
        
        x1 = nn.Conv(init_dim, [3], padding=1)(x)
        
        h = SResNet(init_dim)(x1, t1)
        
        h = Downsample(init_dim*2)(h)

        # nn.Conv(init_dim*2, [4], [4], padding=0)(h)        
        # assert h.shape == (N, W//4, init_dim*2)
        
        h = SResNet(init_dim*2)(h, t1)
        
        # h = nn.ConvTranspose(init_dim, [4], [4], padding=0)(h)
        h = Upsample(init_dim)(h)        

        assert h.shape == (N, W, init_dim)
        
        h = SResNet(init_dim)(h, t1)
        
        h = nn.Conv(1, [1], padding=0)(h)
        
        return h

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)
        
        
class SimpleDense(nn.Module):
    
    @nn.compact
    def __call__(self, x, t):
        
        N, W, C = x.shape
        assert C == 1
        x = x.reshape((N, W))
        
        time_dim = 5
        t = nn.Sequential([
            SinusoidalPositionEmbeddings(time_dim),
            nn.Dense(time_dim),
            nn.gelu
        ])(t)
        
        x_dim = 5
        x = nn.Sequential([
            nn.Dense(x_dim),
            nn.gelu
        ])(x)
        
        
        h = jnp.concatenate([x,t], axis=1)
        
        assert h.shape == (N, time_dim + x_dim)
        
        h = nn.Sequential([
            nn.Dense(2*W),
            nn.gelu,
            nn.Dense(2*W),
            nn.gelu,
            nn.Dense(W)
        ])(h)
        
        return h.reshape((N, W, 1))

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)


class ApproximateScore(nn.Module):
    """A simple model with multiple fully connected layers and some fourier features for the time variable."""

    @nn.compact
    def __call__(self, x, t):
        N, W, C = x.shape
        x = x.reshape((N, W))

        in_size = x.shape[1]
        n_hidden = 256
        act = nn.relu
        t = jnp.stack([t - 0.5, jnp.cos(2*jnp.pi*t)],axis=1)
        print(x.shape)
        print(t.shape)
        print((N, 2))
        assert t.shape == (N, 2)
        x = jnp.concatenate([x, t],axis=1)
        assert x.shape == (N, W + 2)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape((N, W))

    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)
