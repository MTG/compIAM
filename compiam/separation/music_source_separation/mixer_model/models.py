import torch
import torch.nn as nn

from compiam.separation.music_source_separation.mixer_model.modules import TFC_TDF


class ConvTDFNet(nn.Module):
    def __init__(
        self,
        hop_length,
        num_blocks,
        dim_t,
        n_fft,
        dim_c,
        dim_f,
        g,
        k,
        l,
        bn,
        bias,
        scale,
    ):
        super(ConvTDFNet, self).__init__()
        self.hop_length = hop_length
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.chunk_size = self.hop_length * (dim_t - 1)
        self.n_bins = self.n_fft // 2 + 1
        self.n = num_blocks // 2

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=dim_c, out_channels=g, kernel_size=(1, 1)),
            nn.BatchNorm2d(g),
            nn.ReLU(),
        )

        f = dim_f
        c = g
        self.encoding_blocks = nn.ModuleList()
        self.ds = nn.ModuleList()
        for i in range(self.n):
            self.encoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))
            self.ds.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=c,
                        out_channels=c + g,
                        kernel_size=scale,
                        stride=scale,
                    ),
                    nn.BatchNorm2d(c + g),
                    nn.ReLU(),
                )
            )
            f = f // 2
            c = c + g

        self.bottleneck_block = TFC_TDF(c, l, f, k, bn, bias=bias)

        self.decoding_blocks = nn.ModuleList()
        self.us = nn.ModuleList()
        for i in range(self.n):
            self.us.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=c,
                        out_channels=c - g,
                        kernel_size=scale,
                        stride=scale,
                    ),
                    nn.BatchNorm2d(c - g),
                    nn.ReLU(),
                )
            )
            f = f * 2
            c = c - g

            self.decoding_blocks.append(TFC_TDF(c, l, f, k, bn, bias=bias))
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=dim_c, kernel_size=(1, 1))
        )

    # calculates (non-mel) complex-valued-spectrograms
    # Output has the shape: torch.Size([b, c, dim_f, dim_t]) (here: c = 2, for mono, 4 for stereo audio, 2048 frequency dimension, 256 time dimension)
    @torch.jit.export
    def stft(self, audio_norm):
        audio_norm = audio_norm
        B, C, L = audio_norm.shape
        factor = int(L / self.chunk_size)
        # in inference: process slices of length chunk_size and add them back together
        spec_chunks = []
        for i in range(factor):
            audio_chunk = audio_norm[
                :, :, self.chunk_size * i : self.chunk_size * (i + 1)
            ].float()
            window = torch.hann_window(
                window_length=self.n_fft,
                periodic=True,
                dtype=torch.float,
                device=audio_norm.device,
            )
            audio_chunk = audio_chunk.reshape([-1, self.chunk_size])
            spec = torch.stft(
                audio_chunk.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                center=True,
                return_complex=True,
            )
            spec = torch.view_as_real(spec)
            spec = spec.permute([0, 3, 1, 2])
            spec_chunks.append(spec)
        spec = torch.cat(spec_chunks, dim=3)
        return spec[:, :, : self.dim_f]

    @torch.jit.export
    def istft(self, spec):
        B, _, _, t = spec.shape
        factor = int(t / self.dim_t)
        audio_chunks = []
        for i in range(factor):
            spec_chunk = spec[:, :, :, self.dim_t * i : self.dim_t * (i + 1)].float()
            freq_pad = torch.zeros(
                [1, self.dim_c, self.n_bins - self.dim_f, self.dim_t],
                device=spec.device,
            ).float()
            window = torch.hann_window(
                window_length=self.n_fft,
                periodic=True,
                dtype=torch.float,
                device=spec.device,
            )
            spec_chunk = torch.cat(
                [spec_chunk, freq_pad.repeat([spec_chunk.shape[0], 1, 1, 1])], -2
            )

            spec_chunk = spec_chunk.permute([0, 2, 3, 1]).contiguous()
            spec_chunk = torch.view_as_complex(spec_chunk)
            spec_chunk = torch.istft(
                spec_chunk,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window,
                center=True,
                return_complex=False,
            )
            audio_chunks.append(spec_chunk.reshape([B, 1, -1]))

        audio = torch.cat(audio_chunks, dim=2)
        return audio

    @torch.jit.export
    def forward(self, audio, iteration=0):
        spec = self.stft(audio)
        x = self.first_conv(spec)
        x = x.transpose(-1, -2)
        ds_outputs = []
        for i, (encoding_block, ds_block) in enumerate(
            zip(self.encoding_blocks, self.ds)
        ):
            x = encoding_block(x)
            ds_outputs.append(x)
            x = ds_block(x)
        x = self.bottleneck_block(x)

        for i, (decoding_block, us_block) in enumerate(
            zip(self.decoding_blocks, self.us)
        ):
            x = us_block(x)
            x = x * ds_outputs[-i - 1]
            x = decoding_block(x)

        x = x.transpose(-1, -2)
        x = self.final_conv(x)

        return self.istft(x)


class MDXModel(nn.Module):
    def __init__(
        self,
        sources=["vocal", "violin"],
        hop_length=558,
        dim_t=256,
        n_fft=6144,
        dim_c=2,
        dim_f=2048,
        num_blocks=11,
        g=32,
        k=3,
        l=3,
        bn=4,
        bias=False,
        scale=2,
    ):
        super(MDXModel, self).__init__()
        self.sources = sources
        self.hop_length = hop_length
        self.dim_t = dim_t
        self.n_fft = n_fft
        self.dim_c = dim_c
        self.dim_f = dim_f
        self.num_blocks = num_blocks
        self.g = g
        self.k = k
        self.l = l
        self.bn = bn
        self.bias = bias
        self.scale = scale
        self.chunk_size = self.hop_length * (self.dim_t - 1)

        # Separators
        self.model_dict = nn.ModuleList()
        for _ in self.sources:
            self.model_dict.append(
                ConvTDFNet(
                    hop_length=hop_length,
                    num_blocks=num_blocks,
                    dim_t=dim_t,
                    n_fft=n_fft,
                    dim_c=dim_c,
                    dim_f=dim_f,
                    g=g,
                    k=k,
                    l=l,
                    bn=bn,
                    bias=bias,
                    scale=scale,
                )
            )

        # Mixer module
        self.mixer = MixerMDX(num_sources=len(self.sources), chunk_size=self.chunk_size)

    def infer(self, audio):
        # audio has shape B, 1, N
        audio = audio.reshape(-1)
        predictions = []
        pad_length = self.chunk_size - (audio.shape[-1] % self.chunk_size)
        audio = torch.nn.functional.pad(audio, (0, pad_length))

        for i in range(0, audio.shape[-1], self.chunk_size):
            audio_chunk = audio[i : i + self.chunk_size].reshape(
                1, 1, -1
            )  # TODO Batching
            predictions.append(self.forward(audio_chunk))

        result = torch.cat(predictions, dim=-1)
        result = result[:, :, :-pad_length]

        vocal_separation = result[:, 0, :]
        violin_separation = result[:, 1, :]
        return vocal_separation, violin_separation

    def forward(self, audio):
        """Forward pass of the model separation"""
        predictions = []
        for model in self.model_dict:
            predictions.append(model(audio).squeeze(1))
        predictions.append(
            audio.squeeze(1)
        )  # the mixer module also receives the mix-audio, shape B, N
        prediction = torch.stack(predictions, dim=1)  # shape (B, (num_sources + 1), N)
        prediction = self.mixer(prediction)
        return prediction


class MixerMDX(nn.Module):
    def __init__(self, num_sources, chunk_size):
        super(MixerMDX, self).__init__()
        self.num_sources = num_sources
        self.chunk_size = chunk_size
        self.mixing_layer = nn.Linear(
            self.num_sources + 1, self.num_sources, bias=False
        )

    def forward(self, x):
        x = x.reshape(-1, self.num_sources + 1, self.chunk_size).transpose(-1, -2)
        x = self.mixing_layer(x)
        return x.transpose(-1, -2).reshape(-1, self.num_sources, self.chunk_size)
