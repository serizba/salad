from typing import Iterator
import pytorch_lightning as pl
import torch
from torch.nn.parameter import Parameter
from torch.optim import lr_scheduler, optimizer

import utils
from models import helper


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
        #---- Backbone
        backbone_arch='effdinov2',
        backbone_config={},
        
        #---- Teacher
        teacher_arch='effdinov2',
        teacher_config={},
        
        #---- Aggregator
        agg_arch='ConvAP',
        agg_config={},
        
        #---- Train hyperparameters
        lr=0.03, 
        optimizer='sgd',
        weight_decay=1e-3,
        momentum=0.9,
        lr_sched='linear',
        lr_sched_args = {
            'start_factor': 1,
            'end_factor': 0.2,
            'total_iters': 4000,
        },
        
        #----- Loss
        loss_name='MultiSimilarityLoss', 
        miner_name='MultiSimilarityMiner', 
        miner_margin=0.1,
        faiss_gpu=False
    ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.backbone_config = backbone_config

        # Teacher
        self.teacher_arch = teacher_arch
        self.teacher_config = teacher_config
        
        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters() # write hyperparams into a file
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config)
        if teacher_arch is not None:
            self.teacher = helper.get_backbone(teacher_arch, teacher_config)
        else:
            self.teacher = None
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

        # For validation in Lightning v2.0.0
        self.val_outputs = []
        
    # the forward pass of the lightning model
    @utils.yield_as(tuple)
    def forward(self, x):
        if self.training:
            f, t, _, attn, pred_simm, simm = self.backbone(x)
            # simm = simm.unsqueeze(1) # B, 1, NP
            # simm = simm.reshape(simm.shape[0], simm.shape[1], -1)
        
        # f = f.reshape(f.shape[0], f.sahpe[1], -1)
        # weighted_f = f * re_simm # ............................| B, DIM, P, P
        
        else:
            f, t, _, attn, pred_simm, simm = self.backbone(x, calc_cosine=False)
        
        yield self.aggregator((f, t)), attn, pred_simm, simm
        
        # if self.training:
        #     # 여기는 다음에 될 것 같음요
        #     if self.teacher is not None:
        #         mask = mask.detach().unsqueeze(1).bool() # B/2, 1, NP..? 왜 이렇게 해놓은걸까나..?
        #         with torch.no_grad():
        #             tea_f, tea_t, _, tea_attn, _ = self.teacher(x, prune=False) # feature : B, NP, DIM일거고 token은 : B, 1, DIM일거란 말이지
        #             mask = torch.cat((mask, mask)) # B, 1, NP
        #             tea_f = tea_f.view(tea_f.shape[0], tea_f.shape[1], -1)  # feature : B, NP, DIM으로 수정
        #             masked_f = tea_f * mask
        #             indices = mask.expand_as(mask)
        #             masked_f = masked_f[indices].reshape(f.shape) # as same as student f
                    
        #             #att
        #             tea_attn = tea_attn.view(tea_attn.shape[0], tea_attn.shape[1], -1) # B, NH, 
        #             linear = nn.Linear(tea_attn.shape[-1], attn.shape[-1] * attn.shape[-2])
        #             tea_attn = lienar(tea_attn)
        #             tea_attn = tea_attn.view(attn.shape)
                    
        #             # TODO: mask (f, attn) by mask 
        #         yield self.aggregator((masked_f, tea_t)), tea_attn
        #     else:
        #         yield None, None
        # else:
        #     pass
        
    @utils.yield_as(list)
    def parameters(self, recurse: bool=True) -> Iterator[Parameter]:
        yield self.backbone.pos_embed
        yield from self.backbone.blocks[self.backbone.masked_block+1:].parameters(recurse=recurse)
        yield from self.backbone.norm.parameters(recurse=recurse)
        yield from self.backbone.fc_norm.parameters(recurse=recurse)
        yield from self.backbone.head_drop.parameters(recurse=recurse)
        yield from self.backbone.predictor.parameters(recurse=recurse) # predictor parameter 추가
        yield from self.aggregator.parameters(recurse=recurse)
    
    # configure the optimizer 
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay, 
                momentum=self.momentum
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')
        

        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'], gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )

        return [optimizer], [scheduler]
    
    # configure the optizer step, takes into account the warmup stage
    def optimizer_step(self,  epoch, batch_idx, optimizer, optimizer_closure):
        # warm up lr
        optimizer.step(closure=optimizer_closure)
        self.lr_schedulers().step()
        
    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels, pred_simm, simm, log_accuracy: bool=False):
        
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)
            #contrastive,,,?

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        if log_accuracy:
            # keep accuracy of every batch and later reset it at epoch start
            self.batch_acc.append(batch_acc)
            # log it
            self.log('b_acc', sum(self.batch_acc) /
                    len(self.batch_acc), prog_bar=True, logger=True)
        
        # MSE ver.
        # simm_loss = torch.nn.MSELoss()(pred_simm, simm)
        
        # split_pred = torch.chunk(pred_simm, chunks=2, dim=0)
        # simm_loss += torch.nn.MSELoss()(split_pred[0], split_pred[1])
        # total_loss = loss + simm_loss
        
        # L1 ver.
        simm_loss = torch.nn.L1Loss()(pred_simm, simm)
        
        split_pred = torch.chunk(pred_simm, chunks=2, dim=0)
        simm_loss += torch.nn.L1Loss()(split_pred[0], split_pred[1])
        total_loss = loss + simm_loss

        return total_loss, loss, simm_loss
    
    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape
        assert N == 2  # Our method forces each place to have exactly two images in a mini-batch. 
        
        # reshape places and labels
        # data 를 다시...
        # images = places.view(BS*N, ch, h, w)
        # labels = labels.view(-1)
        image_1, image_2 = torch.chunk(places, chunks=2, dim=1)
        image_1 = image_1.squeeze()
        image_2 = image_2.squeeze()
        images = torch.cat([image_1, image_2], dim=0)
        
        label_1, label_2 = torch.chunk(labels, chunks=2, dim=1)
        label_1 = label_1.squeeze()
        label_2 = label_2.squeeze()
        labels = torch.cat([label_1, label_2])

        # Feed forward the ba6tch to the model
        # Here we are calling the method forward that we defined above
        
        (output) = self(images) 
        descriptors_student, _, pred_simm, simm = output[0]
        if torch.isnan(descriptors_student).any():
            raise ValueError('NaNs in descriptors')
        # if descriptors_teacher is not None:
        #     if torch.isnan(descriptors_teacher).any():
        #         raise ValueError('NaNs in descriptors')

        # Call the loss_function we defined above
        loss, loss_cl, loss_simm = self.loss_function(descriptors_student, labels, pred_simm, simm, log_accuracy=True) 
        if self.teacher is not None:
            loss_teacher = self.loss_function(descriptors_teacher, labels, log_accuracy=False) 
            loss = loss + loss_teacher
        
        self.log('loss', loss.item(), logger=True, prog_bar=True)
        self.log('loss_cl', loss_cl.item(), logger=True, prog_bar=True)
        self.log('loss_simm', loss_simm.item(), logger=True, prog_bar=True)
        return {'loss': loss, 'loss_cl': loss_cl, 'loss_simm':loss_simm}
    
    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        output  = self(places)
        descriptors, _, _, _ = output[0]
        self.val_outputs[dataloader_idx].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()
    
    def on_validation_epoch_start(self):
        # reset the outputs list
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]
    
    def on_validation_epoch_end(self):
        """this return descriptors in their order
        depending on how the validation dataset is implemented 
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        val_step_outputs = self.val_outputs

        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets)==1: # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            elif 'nordland' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            elif 'sped' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                positives = val_dataset.ground_truth
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[ : num_references]
            q_list = feats[num_references : ]
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        print('\n\n')

        # reset the outputs list
        self.val_outputs = []