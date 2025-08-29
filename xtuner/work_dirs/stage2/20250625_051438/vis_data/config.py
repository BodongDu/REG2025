SYSTEM = ''
accumulative_counts = 8
batch_size = 1
betas = (
    0.9,
    0.999,
)
custom_hooks = [
    dict(
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.DatasetInfoHook'),
    dict(
        evaluation_images='./BLCA/TCGA-GV-A40G-01Z-00-DX1.csv',
        evaluation_inputs=[
            'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.',
        ],
        every_n_iters=1000,
        prompt_template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
        system='',
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.engine.hooks.EvaluateChatHook'),
]
data_path = 'slidechat_train_vqa_stage2.json'
dataloader_num_workers = 64
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=500,
        max_keep_ckpts=2,
        type='mmengine.hooks.CheckpointHook'),
    logger=dict(
        interval=10,
        log_metric_by_epoch=False,
        type='mmengine.hooks.LoggerHook'),
    param_scheduler=dict(type='mmengine.hooks.ParamSchedulerHook'),
    sampler_seed=dict(type='mmengine.hooks.DistSamplerSeedHook'),
    timer=dict(type='mmengine.hooks.IterTimerHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evaluation_freq = 1000
evaluation_images = './BLCA/TCGA-GV-A40G-01Z-00-DX1.csv'
evaluation_inputs = [
    'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.',
]
image_path_list = None
launcher = 'none'
llava_dataset = dict(
    data_path='slidechat_train_vqa_stage2.json',
    dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
    image_folder='',
    image_path_list=None,
    max_length=19600,
    pad_image_to_square=False,
    per_image_length=None,
    template_map_fn=dict(
        template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
        type='xtuner.dataset.map_fns.template_map_fn_factory'),
    tokenizer=dict(
        padding_side='right',
        pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        trust_remote_code=True,
        type='transformers.AutoTokenizer.from_pretrained'),
    type='xtuner.dataset.LLaVADataset')
llm_name_or_path = 'Qwen/Qwen2.5-7B-Instruct'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
lr = 2e-05
max_epochs = 2
max_length = 19600
max_norm = 1
model = dict(
    freeze_llm=False,
    llm=dict(
        pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
        torch_dtype='torch.float16',
        trust_remote_code=True,
        type='transformers.AutoModelForCausalLM.from_pretrained'),
    pretrained_pth='stage1_pth',
    train_stage='2',
    type='xtuner.model.LLaVAModel')
optim_type = 'torch.optim.AdamW'
optim_wrapper = dict(
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ),
        lr=2e-05,
        type='torch.optim.AdamW',
        weight_decay=0),
    type='DeepSpeedOptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=2,
        eta_min=0.0,
        type='mmengine.optim.CosineAnnealingLR'),
]
per_image_length = None
pretrained_pth = 'stage1_pth'
prompt_template = 'xtuner.utils.PROMPT_TEMPLATE.qwen_chat'
randomness = dict(deterministic=False, seed=None)
resume = False
runner_type = 'FlexibleRunner'
sample_type = 'wsi'
save_steps = 500
save_total_limit = 2
strategy = dict(
    config=dict(
        bf16=dict(enabled=True),
        fp16=dict(enabled=False, initial_scale_power=16),
        gradient_accumulation_steps='auto',
        gradient_clipping='auto',
        train_micro_batch_size_per_gpu='auto',
        zero_allow_untested_optimizer=True,
        zero_force_ds_cpu_optimizer=False,
        zero_optimization=dict(overlap_comm=True, stage=2)),
    exclude_frozen_parameters=True,
    gradient_accumulation_steps=8,
    gradient_clipping=1,
    sequence_parallel_size=1,
    train_micro_batch_size_per_gpu=1,
    type='xtuner.engine.DeepSpeedStrategy')
tokenizer = dict(
    padding_side='right',
    pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
    trust_remote_code=True,
    type='transformers.AutoTokenizer.from_pretrained')
train_cfg = dict(max_epochs=2, type='xtuner.engine.runner.TrainLoop')
train_dataloader = dict(
    batch_size=1,
    collate_fn=dict(type='xtuner.dataset.collate_fns.default_collate_fn'),
    dataset=dict(
        data_path='slidechat_train_vqa_stage2.json',
        dataset_map_fn='xtuner.dataset.map_fns.llava_map_fn',
        image_folder='',
        image_path_list=None,
        max_length=19600,
        pad_image_to_square=False,
        per_image_length=None,
        template_map_fn=dict(
            template='xtuner.utils.PROMPT_TEMPLATE.qwen_chat',
            type='xtuner.dataset.map_fns.template_map_fn_factory'),
        tokenizer=dict(
            padding_side='right',
            pretrained_model_name_or_path='Qwen/Qwen2.5-7B-Instruct',
            trust_remote_code=True,
            type='transformers.AutoTokenizer.from_pretrained'),
        type='xtuner.dataset.LLaVADataset'),
    num_workers=64,
    pin_memory=True,
    sampler=dict(shuffle=True, type='mmengine.dataset.DefaultSampler'))
visualizer = None
warmup_ratio = 0.03
weight_decay = 0
work_dir = 'work_dirs/stage2'
