import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
from pytorch_lightning.strategies import DeepSpeedStrategy

from mod_vpr_model import VPRModel
from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

if __name__ == '__main__':        
    datamodule = GSVCitiesDataModule(
        batch_size=16,
        img_per_place=2,
        min_img_per_place=2,
        shuffle_all=False, # shuffle all images or keep shuffling in-city only
        random_sample_from_each_place=True,
        image_size=(224, 224),
        num_workers=16,
        persistent_workers=True, # for CPU
        show_data_stats=True,
        val_set_names=[
            # 'pitts30k_val', 
            # 'pitts30k_test', 
            'pitts250k_val', 
            'pitts250k_test', 
            # 'nordland', 
            # 'sped', 
            'msls_val', 
        ], 
    )
    
    model = VPRModel(
        #---- Encoder
        backbone_arch='dinov2_vitb14',
        backbone_config={
            'model_name': 'dinov2_vitb14',
            'num_trainable_blocks': 4,
            'return_token': True,
            'norm_layer': True,
            'masking_rate': 0.5,
        },
        
        agg_arch='SALAD',
        agg_config={
            'num_channels': DINOV2_ARCHS['dinov2_vitb14'], # Effdinov2랑 사이즈를 맞춰야함..
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
        },
        lr = 6e-5,
        optimizer='adamW',
        weight_decay=9.5e-9, # 0.001 for sgd and 0 for adam,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },

        #----- Loss functions
        # example: ContrastiveLoss, TripletMarginLoss, MultiSimilarityLoss,
        # FastAPLoss, CircleLoss, SupConLoss,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner', # example: TripletMarginMiner, MultiSimilarityMiner, PairMarginMiner
        miner_margin=0.1,
        faiss_gpu=False
    )

    # model params saving using Pytorch Lightning
    # we save the best 3 models accoring to Recall@1 on pittsburg val
    checkpoint_cb = pl_callbacks.ModelCheckpoint(
        monitor='pitts250k_val/R1',
        filename=f'{model.encoder_arch}' + '_({epoch:02d})_R1[{pitts250k_val/R1:.4f}]_R5[{pitts250k_val/R5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode='max'
    )

    #------------------
    # we instanciate a trainer
    # trainer = pl.Trainer(
    #     accelerator='gpu',
    #     devices=4, strategy='ddp_find_unused_parameters_true',
    #     default_root_dir=f'./logs/', # Tensorflow can be used to viz 
    #     num_nodes=1,
    #     num_sanity_val_steps=0, # runs a validation step before stating training
    #     precision='16-mixed', # we use half precision to reduce  memory usage
    #     max_epochs=10,  # increased by 8 because the batch was halved. 
    #     check_val_every_n_epoch=1, # run validation every epoch
    #     callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
    #     reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
    #     log_every_n_steps=20,
    # )
    trainer = pl.Trainer(
        accelerator='gpu', 
        # strategy=DeepSpeedStrategy(),
        devices=[2],
        default_root_dir=f'./logs/', # Tensorflow can be used to viz 
        num_nodes=1,
        num_sanity_val_steps=0, # runs a validation step before stating training
        precision='16-mixed', # we use half precision to reduce  memory usage
        max_epochs=10,  # increased by 8 because the batch was halved. 
        check_val_every_n_epoch=1, # run validation every epoch
        callbacks=[checkpoint_cb],# we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1, # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )


# trainer = Trainer(accelerator=DDPPlugin(strategy=DDPStrategy(find_unused_parameters=True)))

    # we call the trainer, we give it the model and the datamodule
    trainer.fit(model=model, datamodule=datamodule)
