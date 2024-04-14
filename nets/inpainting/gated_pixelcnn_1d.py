import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.spg.gated_pixelcnn_v2 import GatedMaskedConv2
from nets.spg.vqvae_modules import Res_CNR_Stack


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        # try:
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)
        # except AttributeError:
        #     print("Skipping initialization of ", classname)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module





class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedConv(nn.Module):
    def __init__(self, dim, kernel, stride, padding, residual, upsample=False, double=False):
        super().__init__()
        self.residual = residual
        if stride == 2 and dim > 128 :
            self.in_dim = dim * 2 if upsample else dim //2
        else:
            self.in_dim = dim

        if not upsample:
            conv = nn.Conv1d
        else:
            conv = nn.ConvTranspose1d

        if double:
            self.in_dim = dim * 2

        self.vert_stack = conv(
            self.in_dim, dim * 2,
            kernel, stride, padding
        )

        if stride == 2 or double:
            self.conv = conv(
            self.in_dim, dim,
            kernel, stride, padding
            )
        else:
            self.conv = nn.Identity()

        self.gate = GatedActivation()

        if self.residual:
            self.res = nn.Conv1d(dim, dim, 1)

    def forward(self, x):

        h_vert = self.vert_stack(x)
        h_vert = h_vert[:, :, :]
        out = self.gate(h_vert)

        if self.residual:
            out = self.res(out) + self.conv(x)

        return out


class GatedConv2(nn.Module):
    def __init__(self, dim, kernel, stride, padding, residual, upsample=False, double=False):
        super().__init__()
        self.residual = residual
        if stride == 2 and dim > 128 :
            self.in_dim = dim * 2 if upsample else dim //2
        else:
            self.in_dim = dim

        if not upsample:
            conv = nn.Conv2d
        else:
            conv = nn.ConvTranspose2d

        if double:
            self.in_dim = dim * 2

        self.vert_stack = conv(
            self.in_dim, dim * 2,
            kernel, stride, padding
        )

        if stride == 2 or double:
            self.conv = conv(
            self.in_dim, dim,
            kernel, stride, padding
            )
        else:
            self.conv = nn.Identity()

        self.gate = GatedActivation()

        if self.residual:
            self.res = nn.Conv1d(dim, dim, 1)

    def forward(self, x):

        h_vert = self.vert_stack(x)
        h_vert = h_vert[:, :, :]
        out = self.gate(h_vert)

        if self.residual:
            out = self.res(out) + self.conv(x)

        return out


class GatedMaskedConv1(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual, n_classes=0):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1)  # (ceil(n/2), n)
        padding_shp = (kernel // 2)
        self.vert_stack = nn.Conv1d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.gate = GatedActivation()

        if self.residual:
            self.res = nn.Conv1d(dim, dim, 1)

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row

    def forward(self, x, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x)
        h_vert = h_vert[:, :, :x.size(-1)]
        out = self.gate(h_vert + h[:, :, None])

        if self.residual:
            out = self.res(out) + x

        return out


class GatedPixelCNN(nn.Module):
    def __init__(self, groups, condi_dim, intermediate_dim, layers, num_code, n_classes, identity, maskgit):
        super().__init__()
        self.dim = intermediate_dim
        self.groups = groups
        self.ar_layers = layers
        gatedmaskedconv = GatedMaskedConv1 if self.groups==1 else GatedMaskedConv2
        gatedconv = GatedConv
        conv = nn.Conv1d if self.groups==1 else nn.Conv2d
        bn = nn.LayerNorm

        self.fusion = nn.Sequential(nn.Linear(condi_dim, 512),
                                    bn(512),
                                    nn.LeakyReLU(0.1),
                                    nn.Dropout(0.1))
        self.fusion2 = nn.Sequential(nn.Linear(1024, 512),
                                    bn(512),
                                    nn.LeakyReLU(0.1),
                                    nn.Dropout(0.1))
        if self.groups != 1:
            self.fusion3 = nn.Sequential(nn.Linear(1024, 512),
                                         bn(512),
                                         nn.LeakyReLU(0.1),
                                         nn.Dropout(0.1))

        # Create embedding layer to embed input
        self.state_embedding = nn.Embedding(num_code+1, intermediate_dim)

        # if self.groups == 1:
        #     self.g_pos = 0
        # else:
        #     self.g_pos = nn.Parameter(torch.FloatTensor(self.groups, intermediate_dim))
        #     nn.init.normal_(self.g_pos, std=0.001)

        # Building the PixelCNN layer by layer
        self.ar = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(self.ar_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.ar.append(
                gatedmaskedconv(mask_type, intermediate_dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            conv(intermediate_dim, 512, 1),
            nn.ReLU(True),
            conv(512, num_code, 1)
        )

        self.apply(weights_init)

    def forward(self, state, label, condition, epoch_ratio):
        """
        state.shape = (B, Cs, W//8)
        label.shape = (B, 1)
        condition.shape = (B, Ca, W)
        """

        condition = self.fusion(condition.transpose(1, 2)).transpose(1, 2)
        x = self.ar_forward(state, label, condition)

        return x

    def ar_forward(self, state, label, condition):
        x = state
        shp = x.size() + (-1,)
        x = self.state_embedding(x.reshape(-1)).reshape(shp)  # (B, H, W, C)
        if self.groups == 1:
            x = x.permute(0, 2, 1)
            for i, layer in enumerate(self.ar):
                if i == 1:
                    x = self.fusion2(torch.cat([x, condition], dim=1).transpose(1, 2)).transpose(1, 2)
                x = layer(x, label)
        else:
            # x = x + self.g_pos
            condition = condition.unsqueeze(3).repeat(1, 1, 1, self.groups)
            x = x.permute(0, 3, 1, 2)
            x_v, x_h = (x, x)
            for i, layer in enumerate(self.ar):
                if i == 1:
                    x_v = self.fusion2(torch.cat([x_v, condition], dim=1).transpose(1, 3)).transpose(1, 3)
                    x_h = self.fusion3(torch.cat([x_h, condition], dim=1).transpose(1, 3)).transpose(1, 3)
                x_v, x_h = layer(x_v, x_h, label)
                x = x_h

        return self.output_conv(x).permute(0, 2, 3, 1).contiguous()

    def predict(self, label=None, condition=None):
        Batch, Time, Group = condition.shape[0], condition.shape[2], self.groups
        condition = self.fusion(condition.transpose(1, 2)).transpose(1, 2)

        param = next(self.parameters())

        x = torch.zeros(
            (Batch, 1, Group),
            dtype=torch.int64, device=param.device
        )

        for i in range(Time):
            for j in range(Group):
                logits = self.ar_forward(x, label, condition[..., :(i+1)])
                probs = F.softmax(logits[:, i, j], -1)
                x.data[:, i, j].copy_(
                    # probs.argmax().squeeze().data
                    probs.multinomial(1).squeeze().data
                )
            if i != (Time - 1):
                blank = torch.zeros((Batch, 1, Group), dtype=torch.int64, device=param.device)
                x = torch.cat([x, blank], dim=1)
        return x


class GatedRefineNet(nn.Module):
    def __init__(self, mot_dim=256, dim=64, layers=10, mid_layers=10):
        super().__init__()
        self.dim = dim
        self.mid_layers = mid_layers
        d = dim // 4
        upsample = False

        self.motion_embedding = nn.Conv1d(mot_dim+2, dim//4, 1, 1, padding=0)

        self.enc = nn.ModuleList()
        for i in range(layers):
            residual = True
            if i == 0:
                kernel, stride, padding = 7, 1, 3
            elif i in [2, 4, 6]:
                kernel, stride, padding = 4, 2, 1
                if i in [4, 6]:
                    d = d * 2
            else:
                kernel, stride, padding = 3, 1, 1

            self.enc.append(
                GatedConv(d, kernel, stride, padding, residual)
            )

        self.mid = nn.ModuleList()
        for i in range(mid_layers):
            kernel, stride, padding, residual = 3, 1, 1, True
            double = True if i == mid_layers // 2 else False

            self.mid.append(
                GatedConv(d, kernel, stride, padding, residual, double=double)
            )

        self.dec = nn.ModuleList()
        for i in range(layers):
            residual = True
            if i == layers-1:
                kernel, stride, padding = 7, 1, 3
            elif i in [3, 5, 7]:
                kernel, stride, padding = 4, 2, 1
                upsample = True
                if i in [3, 5]:
                    d = d // 2
                if i == 5:
                    double = True
            else:
                kernel, stride, padding = 3, 1, 1
                if i in [4, 6, 8]:
                    double = True

            self.dec.append(
                GatedConv(d, kernel, stride, padding, residual, upsample=upsample, double=double)
            )
            upsample = False
            double = False

        # Add the output layer
        self.output_conv = nn.Conv1d(dim//4, mot_dim, 1, 1, padding=0)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, masked_motion, state, mask):
        """
        masked_motion.shape = (B, Cm, W)
        state.shape = (B, Cs, W//8)
        mask.shape = (B, 1, W)
        label.shape = (B, 1)
        audio.shape = (B, Ca, W)
        """

        ones = torch.ones_like(masked_motion[:, :, 0]).unsqueeze(dim=2)
        mm = torch.cat([masked_motion, ones, ones*mask[:, :, 0:1]], dim=2).transpose(1, 2)
        x = self.motion_embedding(mm)

        enc_feats = []

        for i, layer in enumerate(self.enc):
            x = layer(x)
            if i in [1, 3, 5]:
                enc_feats.append(x)

        for i, layer in enumerate(self.mid):
            if i == self.mid_layers // 2:
                x = torch.cat([x, state], dim=1)
            x = layer(x)

        f = 1
        for i, layer in enumerate(self.dec):
            if i in [4, 6, 8]:
                enc_feats[-f] = F.interpolate(enc_feats[-f], size=x.shape[-1], mode='linear', align_corners=True)
                x = torch.cat([x, enc_feats[-f]], dim=1)
                f = f + 1
            x = layer(x)

        return self.output_conv(x)


class Stage2(nn.Module):
    def __init__(self, mot_dim, aud_dim, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Stage2, self).__init__()
        self._num_hiddens = num_hiddens
        self._num_residual_layers = num_residual_layers
        self._num_residual_hiddens = num_residual_hiddens

        self.motion_embedding = nn.Conv1d(mot_dim+2, self._num_hiddens // 2, 1, 1, padding=0)
        self.audio_embedding = nn.Conv1d(aud_dim, self._num_hiddens // 2, 1, 1, padding=0)

        self._enc_1 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)
        self._enc_2 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)
        self._enc_3 = Res_CNR_Stack(self._num_hiddens, self._num_residual_layers, leaky=True, leaky_out=True)

        # Add the output layer
        self.output_conv = nn.Conv1d(self._num_hiddens, mot_dim, 1, 1, padding=0)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initializes weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight.data)
            module.bias.data.fill_(0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, masked_motion, audio, mask):
        """
        masked_motion.shape = (B, Cm, W)
        state.shape = (B, Cs, W//8)
        mask.shape = (B, 1, W)
        label.shape = (B, 1)
        audio.shape = (B, Ca, W)
        """

        ones = torch.ones_like(masked_motion[:, :, 0]).unsqueeze(dim=2)
        mm = torch.cat([masked_motion, ones, ones*mask[:, :, 0:1]], dim=2).transpose(1, 2)
        x = self.motion_embedding(mm)
        a = self.audio_embedding(audio.transpose(1, 2))

        x = torch.cat([x, a], dim=1)

        x = self._enc_1(x)
        x = self._enc_2(x)
        x = self._enc_3(x)

        return self.output_conv(x)


def main():
    net = GatedPixelCNN(4, 1024, 512, 10, 512, 4, True, True).to('cuda')
    state = torch.randint(0, 512, [4, 22, 4]).to('cuda')
    label = torch.randint(0, 4, [4]).to('cuda')
    condition = torch.randn([4, 1024, 22]).to('cuda')
    x = net.predict(label, condition)
    print('wait')

if __name__ == '__main__':
    main()

