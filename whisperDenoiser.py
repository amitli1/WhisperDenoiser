from lightning.pytorch           import Trainer
from typing                      import Optional, List, Tuple, Union
from whisper.tokenizer           import get_tokenizer
from lightning.pytorch           import Trainer
from torch.nn.functional         import pad
from pytorch_lightning.loggers   import TensorBoardLogger
from pytorch_lightning.loggers   import WandbLogger
from sklearn.model_selection     import train_test_split
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from tqdm                        import tqdm

import torchaudio.transforms     as at
import torch.nn                  as nn
import pandas                    as pd
import numpy                     as np
import lightning                 as pl
import torch.nn.functional       as f

import whisper
import evaluate
import torch
import torchaudio
import glob
import wandb


WHISPER_TYPE = "base"
LANGUAGE     = "ru"
BATCH_SIZE   = 4


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df_data, language):
        self.df = df_data
        self.woptions = whisper.DecodingOptions(language=language, without_timestamps=True)
        self.multilingual_tokenizer = get_tokenizer(multilingual=True, language=language, task=self.woptions.task)

    def get_sentance(self, fname):
        fname = fname[fname.rfind('/') + 1:]
        fname = fname.replace("wav", "mp3")
        sentence = self.df[self.df.file == fname]['sentence'].values[0]

        return sentence

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        fname = self.df.file.values[i]
        sentance = self.get_sentance(fname)

        noise_file_name = rf"Noised_Dataset/{fname.replace('mp3', 'wav')}"
        audio, sr = torchaudio.load(noise_file_name, normalize=True)
        audio = whisper.pad_or_trim(audio.flatten())
        noise_mels = whisper.log_mel_spectrogram(audio).squeeze()

        clean_file_name = rf"Clean_Dataset/{fname.replace('mp3', 'wav')}"
        audio, sr = torchaudio.load(clean_file_name, normalize=True)
        audio = whisper.pad_or_trim(audio.flatten())
        clean_mels = whisper.log_mel_spectrogram(audio).squeeze()

        # --- get tokens (will be used for ce loss)
        multilingual_tokens = [
                                  *self.multilingual_tokenizer.sot_sequence_including_notimestamps] + self.multilingual_tokenizer.encode(
            sentance)
        gt_tokens = multilingual_tokens[1:] + [self.multilingual_tokenizer.eot]

        return {"file_name": fname,
                "clean_mels": clean_mels.unsqueeze(dim=0),
                "noise_mels": noise_mels.unsqueeze(dim=0),
                "sentance": sentance,
                "multilingual_tokens": multilingual_tokens,
                "gt_tokens": gt_tokens}


class WhisperDataCollatorWithPadding:
    def __call__(self, features):
        file_name, clean_mels, noise_mels, sentance, multilingual_tokens, gt_tokens = [], [], [], [], [], []
        for f in features:
            file_name.append(f["file_name"])
            clean_mels.append(f["clean_mels"])
            noise_mels.append(f["noise_mels"])
            sentance.append(f["sentance"])
            multilingual_tokens.append(f["multilingual_tokens"])
            gt_tokens.append(f["gt_tokens"])

        clean_mels = torch.concat([mel[None, :] for mel in clean_mels])
        noise_mels = torch.concat([mel[None, :] for mel in noise_mels])

        gt_tokens_lengths = [len(lab) for lab in gt_tokens]
        multilingual_tokens_length = [len(e) for e in multilingual_tokens]
        max_label_len = max(gt_tokens_lengths + multilingual_tokens_length)

        gt_tokens = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in
                     zip(gt_tokens, gt_tokens_lengths)]
        multilingual_tokens = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in
                               zip(multilingual_tokens, multilingual_tokens_length)]

        batch = {
            "multilingual_tokens": multilingual_tokens,
            "gt_tokens": gt_tokens
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["file_name"] = file_name
        batch["sentance"] = sentance
        batch["clean_mels"] = clean_mels
        batch["noise_mels"] = noise_mels

        return batch


class ResnetBlock(nn.Module):
    """ A single Res-Block module """

    def __init__(self, dim: int, use_bias: bool):
        """
        Init
        :param dim: The dimension
        :param use_bias: Flag to use bias or not
        """
        super(ResnetBlock, self).__init__()

        # A res-block without the skip-connection, pad-conv-norm-relu-pad-conv-norm
        self.conv_block = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(dim, dim // 4, kernel_size=1, bias=use_bias)
            ),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(0.2, True),
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(dim // 4, dim // 4, kernel_size=3, bias=use_bias)
            ),
            nn.BatchNorm2d(dim // 4),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(
                nn.Conv2d(dim // 4, dim, kernel_size=1, bias=use_bias)
            ),
            nn.BatchNorm2d(dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Feed forward run
        :param input_tensor: The input tensor
        :return: The output tensor
        """
        # The skip connection is applied here
        return input_tensor + self.conv_block(input_tensor)


class RescaleBlock(nn.Module):
    """
    Rescale Block class
    """

    def __init__(self, n_layers: int, scale: Optional[float] = 0.5, n_mels: Optional[int] = 64,
                 use_bias: Optional[bool] = True):
        """
        Init
        :param n_layers: The number of layers
        :param scale: Scale factor
        :param n_mels: Base number of channels
        :param use_bias: Flag to use bias or not
        """
        super(RescaleBlock, self).__init__()

        self.scale = scale

        self.conv_layers = [None] * n_layers

        in_channel_power = scale > 1
        out_channel_power = scale < 1
        i_range = range(n_layers) if scale < 1 else range(n_layers - 1, -1, -1)

        for i in i_range:
            self.conv_layers[i] = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=n_mels * 2 ** (i + in_channel_power),
                        out_channels=n_mels * 2 ** (i + out_channel_power),
                        kernel_size=3,
                        stride=1,
                        bias=use_bias,
                    )
                ),
                nn.BatchNorm2d(n_mels * 2 ** (i + out_channel_power)),
                nn.LeakyReLU(0.2, True))

            self.add_module("conv_%d" % i, self.conv_layers[i])

        if scale > 1:
            self.conv_layers = self.conv_layers[::-1]

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, input_tensor: torch.Tensor,
                pyramid: Optional[torch.Tensor] = None,
                return_all_scales: Optional[bool] = False,
                skip: Optional[bool] = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """

        :param input_tensor: The input tensor
        :param pyramid: The pyramid tensor
        :param return_all_scales: Flag to return all scales
        :param skip: Flag to skip or not
        :return: Tuple with feature maps and all scales (if return_all_scales is True)
        """
        feature_map = input_tensor
        all_scales = []
        if return_all_scales:
            all_scales.append(feature_map)

        for i, conv_layer in enumerate(self.conv_layers):

            if self.scale > 1.0:
                feature_map = f.interpolate(
                    feature_map, scale_factor=self.scale, mode="nearest"
                )

            feature_map = conv_layer(feature_map)

            if skip:
                feature_map = feature_map + pyramid[-i - 2]

            if self.scale < 1.0:
                feature_map = self.max_pool(feature_map)

            if return_all_scales:
                all_scales.append(feature_map)

        return (feature_map, all_scales) if return_all_scales else (feature_map, None)


class Unet(nn.Module):
    """ Architecture of the Unet, uses res-blocks """

    def __init__(
            self,
            n_mels: Optional[int] = 64,
            n_blocks: Optional[int] = 6,
            n_downsampling: Optional[int] = 3,
            use_bias: Optional[bool] = True,
            skip_flag: Optional[bool] = True,
    ):
        """
        Init
        :param n_mels: The base number of channels
        :param n_blocks: The number of res blocks
        :param n_downsampling: The number of downsampling blocks
        :param use_bias: Use bias or not
        :param skip_flag: Use skip connections or not
        """
        super(Unet, self).__init__()

        # Determine whether to use skip connections
        self.skip = skip_flag

        # Entry block
        # First conv-block, no stride so image dims are kept and channels dim is expanded (pad-conv-norm-relu)
        self.entry_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.utils.spectral_norm(
                nn.Conv2d(1, n_mels, kernel_size=7, bias=use_bias)
            ),
            nn.BatchNorm2d(n_mels),
            nn.LeakyReLU(0.2, True),
        )

        # Downscaling
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        self.downscale_block = RescaleBlock(n_downsampling, 0.5, n_mels, True)

        # Bottleneck
        # A sequence of res-blocks
        bottleneck_block = []
        for _ in range(n_blocks):
            # noinspection PyUnboundLocalVariable
            bottleneck_block += [
                ResnetBlock(n_mels * 2 ** n_downsampling, use_bias=use_bias)
            ]
        self.bottleneck_block = nn.Sequential(*bottleneck_block)

        # Upscaling
        # A sequence of transposed-conv-blocks, Image dims expand by 2, channels dim shrinks by 2 at each block\
        self.upscale_block = RescaleBlock(n_downsampling, 2.0, n_mels, True)

        # Final block
        # No stride so image dims are kept and channels dim shrinks to 3 (output image channels)
        self.final_block = nn.Sequential(
            # nn.ReflectionPad2d(3), nn.Conv2d(n_mels, 1, kernel_size=7), nn.Tanh()
            # TODO: without Tanh, for not having output [-1,1]
            nn.ReflectionPad2d(3), nn.Conv2d(n_mels, 1, kernel_size=7)
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Feed forward run
        :param input_tensor: The input Tensor
        :param output_size: The output size
        :param random_affine: List of random affine numbers
        :return: The output tensor
        """
        # A condition for having the output at same size as the scaled input is having even output_size

        # Entry block
        feature_map = self.entry_block(input_tensor)

        # Downscale block
        feature_map, downscales = self.downscale_block(
            feature_map, return_all_scales=self.skip
        )

        # Bottleneck (res-blocks)
        feature_map = self.bottleneck_block(feature_map)

        # Upscale block
        feature_map, _ = self.upscale_block(
            feature_map, pyramid=downscales, skip=self.skip
        )

        # Final block
        output_tensor = self.final_block(feature_map)

        return output_tensor

    def save_model(self, model_path):
        cuda = True
        state = {'net': self.state_dict() if cuda else self.state_dict()}

        torch.save(state, model_path)


class UnetWhisperModel(pl.LightningModule):

    def __init__(self):
        super().__init__()

        lang = LANGUAGE
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.l1loss = torch.nn.L1Loss()
        self.metrics_wer = evaluate.load("wer")

        self.whisper_options = whisper.DecodingOptions(language=lang, without_timestamps=True)
        self.whisper_model = whisper.load_model(WHISPER_TYPE)
        self.tokenizer = whisper.tokenizer.get_tokenizer(True, language=LANGUAGE, task=self.whisper_options.task)

        for param in self.whisper_model.parameters():
            param.requires_grad = False

        config = {}
        config["n_mels"] = 80
        config["n_blocks"] = 6
        config["n_downsampling"] = 3
        config["use_bias"] = True
        config["skip_flag"] = True
        self.unet = Unet(config["n_mels"], config["n_blocks"], config["n_downsampling"], config["use_bias"],
                         config["skip_flag"])

    def forward(self, x):
        print("\n")
        print("--> forward")
        logits = self.unet(x)
        whisper_pred = self.whisper(logits)

        return logits, whisper_pred

    def calc_loss(self, batch):

        clean_mels = batch['clean_mels']
        noise_mels = batch['noise_mels']
        gt_sentance = batch['sentance']
        multilingual_tokens = batch['multilingual_tokens'].long()
        gt_tokens = batch['gt_tokens'].long()

        logits = self.unet(noise_mels)
        audio_features = self.whisper_model.encoder(logits.squeeze())
        out = self.whisper_model.decoder(multilingual_tokens, audio_features)

        # --- calc loss
        l1_loss = self.l1loss(logits, clean_mels)
        ce_loss = self.loss_ce(out.view(-1, out.size(-1)), gt_tokens.view(-1))
        loss = l1_loss * 0.5 + ce_loss * 0.5

        return loss, out, gt_tokens

    def calc_wer(self, whisper_out, gt_tokens):
        whisper_out[whisper_out == -100] = self.tokenizer.eot
        gt_tokens[gt_tokens == -100] = self.tokenizer.eot
        o_list, l_list = [], []

        for o, l in zip(whisper_out, gt_tokens):
            o = torch.argmax(o, dim=1)
            o_list.append(self.tokenizer.decode(o).lower())
            l_list.append(self.tokenizer.decode(l).lower().replace('<|endoftext|>', '').replace("<|en|>", "").replace(
                "<|transcribe|>", "").replace("<|notimestamps|>", ""))
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)

        return wer

    def training_step(self, batch, batch_idx):
        loss, whisper_out, gt_tokens = self.calc_loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def eval_step(self, batch):
        loss, whisper_out, gt_tokens = self.calc_loss(batch)
        wer = self.calc_wer(whisper_out, gt_tokens)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/wer", wer, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "wer": wer}

    def validation_step(self, batch, batch_idx):
        loss, whisper_out, gt_tokens = self.calc_loss(batch)
        wer = self.calc_wer(whisper_out, gt_tokens)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_wer", wer, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_wer": wer}

    def test_step(self, batch, batch_idx):
        loss, whisper_out, gt_tokens = self.calc_loss(batch)
        wer = self.calc_wer(whisper_out, gt_tokens)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_wer", wer, on_epoch=True, prog_bar=True, logger=True)
        return {"test_loss": loss, "test_wer": wer}

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(parameters, lr=0.0001)
        return {
            "optimizer": optimizer
        }




if __name__ == "__main__":

    print(f"Start main...")

    #
    # CUDA
    #
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nCUDA: {DEVICE}\n")

    #
    # Dataset
    #
    df                = pd.read_csv("noised_files_v2.csv")
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.33, random_state=42)

    train_dataset = AudioDataset(train_df, LANGUAGE)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   drop_last=True,
                                                   shuffle=True,
                                                   collate_fn=WhisperDataCollatorWithPadding())

    val_dataset = AudioDataset(val_df, LANGUAGE)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=BATCH_SIZE,
                                                 drop_last=True,
                                                 shuffle=True,
                                                 collate_fn=WhisperDataCollatorWithPadding())


    # model
    solver = UnetWhisperModel()

    # logger
    wandb.init()

    # train
    checkpoint_callback = ModelCheckpoint(
        monitor="val_wer",
        mode="min",
        dirpath="checkpoints/",
        filename="best_model-{epoch:02d}-{val_wer:.4f}",
        save_top_k=1,
    )

    # tb_logger    = TensorBoardLogger(save_dir="logs/")
    wandb_logger = WandbLogger(project="UnetWhisperModel")
    trainer = pl.Trainer(max_epochs=50,
                         logger=wandb_logger,
                         accumulate_grad_batches=4,
                         callbacks=[checkpoint_callback])

    trainer.fit(solver, train_dataloader, val_dataloader)
