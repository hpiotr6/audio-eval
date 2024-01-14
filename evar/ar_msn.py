from evar.ar_base import BaseAudioRepr, ToLogMelSpec
import torch


import torchvision
from lightly.models.modules.masked_autoencoder import MAEBackbone
from torch import nn


class AR_MSN(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)
        vit = torchvision.models.vit_b_32(pretrained=False)
        vit.conv_proj = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1), vit.conv_proj)
        self.backbone = MAEBackbone.from_vit(vit)

        # self.body = AudioNTT2022Encoder(n_mels=cfg.n_mels, d=cfg.feature_d)
        if cfg.weight_file is not None and cfg.weight_file != "":
            device = "gpu" if torch.cuda.is_available() else "cpu"
            state_dict = torch.load(cfg.weight_file, map_location=device)["state_dict"]

            filtered_state_dict = {
                key.replace("backbone.", ""): value
                for key, value in state_dict.items()
                if key.startswith("backbone")
            }
            self.backbone.load_state_dict(filtered_state_dict)

    # def precompute(self, device, data_loader):
    #     self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        pass

    #     x = self.to_feature(batch_audio)
    #     x = normalize_spectrogram(self.norm_stats, x)  # B,F,T
    #     x = self.augment_if_training(x)
    #     x = x.unsqueeze(1)  # -> B,1,F,T
    #     x = self.body(x)  # -> B,T,D=C*F
    #     x = x.transpose(1, 2)  # -> B,D,T
    #     return x

    def forward(self, batch_audio):
        batch_audio
        # x = self.encode_frames(batch_audio)
        # x = temporal_pooling(self, x)
        # return x
