"""Wrapper code for:

Data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language

## Reference
- [1] https://ai.facebook.com/research/data2vec-a-general-framework-for-self-supervised-learning-in-speech-vision-and-language/
- [2] https://huggingface.co/facebook/data2vec-audio-large-960h
"""

from evar.ar_base import BaseAudioRepr, temporal_pooling
import logging
try:
    from transformers import Data2VecAudioModel, Wav2Vec2Processor
except:
    logging.error('Install transformers.\n>>> pip install transformers')


class AR_Data2Vec(BaseAudioRepr):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.processor = Wav2Vec2Processor.from_pretrained(cfg.pretrained_model)
        self.backbone = Data2VecAudioModel.from_pretrained(cfg.pretrained_model)

    def encode_frames(self, batch_audio):
        device = batch_audio.device
        preprocessed = self.processor(batch_audio.cpu().numpy(), return_tensors="pt", sampling_rate=16000).input_values
        preprocessed = preprocessed[0].to(device) # [1, B, raw wave length] -> [B, raw wave length]
        features = self.backbone(preprocessed)['last_hidden_state'] # [B, T, D]
        return features.transpose(1, 2) # [B, D, T]

    def forward(self, batch_audio):
        return temporal_pooling(self, self.encode_frames(batch_audio))
