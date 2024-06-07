import os
os.environ["OMP_NUM_THREADS"] = "2"
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class KWSLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):  

        parser.link_arguments('model.init_args.sampling', 'data.init_args.sampling', apply_on='parse')
        parser.link_arguments('model.init_args.resample_every_epoch', 'data.init_args.resample_every_epoch', apply_on='parse')
        parser.link_arguments('model.init_args.kw_type', 'data.init_args.kw_type', apply_on='parse')
        parser.link_arguments('model.init_args.batch_size', 'data.init_args.batch_size', apply_on='parse')
        parser.link_arguments('model.init_args.accumulate_grad_batches', 'trainer.accumulate_grad_batches', apply_on='parse')
 
        parser.add_lightning_class_args(ModelCheckpoint, 'f1_checkpoint')  
        parser.add_lightning_class_args(ModelCheckpoint, 'checkpoint_final') 
        parser.add_lightning_class_args(ModelCheckpoint, 'f1_generalization_checkpoint')  

        parser.add_lightning_class_args(EarlyStopping, 'early_stopping')

    def before_instantiate_classes(self):

        if self.config.fit.model.init_args.adversarial_training:
            del self.config.fit.trainer.accumulate_grad_batches

        # when using adversarial training, changing the batch size in the following way allows correspondence between training step and optimizer step
        # and also makes possible the use of DANNCE to update the input examples and using accumulate_grad_batches at the same time
        if self.config.fit.model.init_args.adversarial_training:
            self.config.fit.data.init_args.batch_size = self.config.fit.model.init_args.batch_size * self.config.fit.model.init_args.accumulate_grad_batches


def cli_main():

    cli = KWSLightningCLI(
        save_config_callback = None,
        subclass_mode_model = True
    )

if __name__ == '__main__':
    
    cli_main()