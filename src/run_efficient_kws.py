import os

os.environ["OMP_NUM_THREADS"] = "2"
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class KWSLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # link the model arguments that the data module also needs, so they
        # only have to be specified once (under `model.init_args`).
        for arg in (
            "n_layers",
            "condensed_dimension",
            "sru_chunk_size_utt",
            "sru_chunk_size_kwd",
            "sru_ctx_window_kwd",
            "sru_ctx_window_utt",
            "sampling",
            "resample_every_epoch",
            "batch_size",
            "features_size",
            "learn_features",
            "load_embeddings",
            "pad_long_before_resize",
            "kws_whisper_ckpt",
        ):
            parser.link_arguments(
                f"model.init_args.{arg}",
                f"data.init_args.{arg}",
                apply_on="parse",
            )

        parser.link_arguments(
            "model.init_args.accumulate_grad_batches",
            "trainer.accumulate_grad_batches",
            apply_on="parse",
        )

        parser.add_lightning_class_args(ModelCheckpoint, "f1_checkpoint")
        parser.add_lightning_class_args(ModelCheckpoint, "f1_l4_checkpoint")
        parser.add_lightning_class_args(ModelCheckpoint, "checkpoint_final")

        parser.add_lightning_class_args(EarlyStopping, "early_stopping")


def cli_main():

    cli = KWSLightningCLI(save_config_callback=None, subclass_mode_model=True)


if __name__ == "__main__":

    cli_main()
