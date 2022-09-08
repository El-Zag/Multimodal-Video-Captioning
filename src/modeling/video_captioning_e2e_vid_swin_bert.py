import torch
from fairscale.nn.misc import checkpoint_wrapper
import random
import warnings
warnings.filterwarnings('error', message='.*NaN or Inf found in input tensor.*')
import numpy as np

class VideoTransformer(torch.nn.Module):
    def __init__(self, args, config, swin, transformer_encoder):
        super(VideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.trans_encoder = transformer_encoder
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = torch.nn.Linear(self.latent_feat_size, self.img_feature_dim) 
        #self.audio_norm = torch.nn.LayerNorm(self.img_feature_dim)
        self.sec_to_frame = [0, 78, 156, 235, 313, 392, 470, 548, 627, 705, 784]
        self.len_sec_to_frame = len(self.sec_to_frame)
        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length
        # learn soft attention mask
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)
        
        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, *args, **kwargs):
        
        #kwargs == input
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        vid_feats = self.swin(images)
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)

        vid_feats = self.fc(vid_feats) 
        #print(kwargs['audio_feat'])
        #print(vid_feats.shape)
        multi_modal_feats = torch.cat((vid_feats, kwargs['audio_feat']), dim=1)
        #print(multi_modal_feats.shape)
        #kwargs['audio_feat'] = self.audio_norm(kwargs['audio_feat']) # * 10
        #print("AUDIO", torch.mean(kwargs['audio_feat']).item())
        #for batch_vid in range(vid_feats.shape[0]):
            #for i in range(1, self.len_sec_to_frame) :
                #if torch.count_nonzero(kwargs['audio_feat'][batch_vid][i-1].cpu().detach()).item() == 0 :
                    #print("EMPTY TENSOR")
                #vid_feats[batch_vid][self.sec_to_frame[i-1]:self.sec_to_frame[i]] += kwargs['audio_feat'][batch_vid][i-1]
        #print("VID", torch.mean(vid_feats).item())
        #print("------------------------")
        
        # SHAPE OF THIS IS [6, 784, 512]
        #test = torch.zeros(size=(vid_feats.shape[0], 10, vid_feats.shape[2]), dtype=torch.float16).to('cuda')
        #vid_feats = torch.cat((vid_feats, test), 1)
        

        # prepare VL transformer inputs
        #kwargs['img_feats'] = vid_feats
        kwargs['img_feats'] = multi_modal_feats
        
        #kwargs['img_feats'] = kwargs['audio_feat']
        #print(self.learn_vid_att.weight.cpu().detach().numpy())
        #np.save("attention_mask", self.learn_vid_att.weight.cpu().detach().numpy())

        #print(self.learn_vid_att.weight.tolist())
        #torch.save(self.learn_vid_att.weight, 'attention_mask.pt')
        del kwargs['audio_feat']

        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        # learn soft attention mask
        if self.learn_mask_enabled:
            kwargs['attention_mask'] = kwargs['attention_mask'].float()
            vid_att_len = self.max_img_seq_length
            learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
            learn_att = self.sigmoid(learn_att)
            #print(f"vid_att_len {vid_att_len}\nlearn_att(1) {learn_att.shape}")
            diag_mask = torch.diag(torch.ones(vid_att_len)).cuda()
            video_attention = (1. - diag_mask)*learn_att
            learn_att = diag_mask + video_attention
            #print(f"video_attention {video_attention.shape}\nlearn_att {learn_att.shape}")
            if self.sparse_mask_soft2hard:
                learn_att = (learn_att>=0.5)*1.0
                learn_att = learn_att.cuda()
                learn_att.requires_grad = False
            kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
        outputs = self.trans_encoder(*args, **kwargs)
        #print(torch.isnan(outputs).any())
        if self.learn_mask_enabled:
            loss_sparsity = self.get_loss_sparsity(video_attention)  
            #print(torch.isnan(loss_sparsity).any())
            outputs = outputs + (loss_sparsity, )  
        return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    def diag_based_init_attn_mask(self, pretrain_attn_mask):
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        zeros_mask = torch.zeros_like(pretrained_learn_att)
        scale_factor = self.max_img_seq_length/pretrained_num_tokens
        
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 


    def bilinear_init_attn_mask(self, pretrain_attn_mask):
        print('init attn mask with bilinear interpolation')
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        scale_factor = int(self.max_img_seq_length/pretrained_num_tokens)
        sampler = torch.nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        with torch.no_grad():
            learn_att = sampler(pretrained_learn_att[None,None,:,:].double())[0,0,:,:].half()

    def random_init_attn_mask(self):
        ('random init attn mask')
        self.learn_vid_att = torch.nn.Embedding(self.max_img_seq_length*self.max_img_seq_length,1)


    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

 