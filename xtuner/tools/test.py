# # # Copyright (c) OpenMMLab. All rights reserved.
# # import argparse
# # import os
# # import os.path as osp
# # from types import FunctionType
# # import deepspeed

# # from mmengine.config import Config, DictAction
# # from mmengine.registry import RUNNERS
# # from mmengine.runner import Runner
# # from sympy import im

# # from xtuner.configs import cfgs_name_path
# # from xtuner.model.utils import guess_load_checkpoint
# # from xtuner.registry import MAP_FUNC
# # from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
# #                           StopWordStoppingCriteria)
# # import torch
# # from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
# # from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
# #                           BitsAndBytesConfig, CLIPImageProcessor,
# #                           CLIPVisionModel, GenerationConfig)
# # from xtuner.utils import PROMPT_TEMPLATE
# # from PIL import Image
# # import pandas as pd
# # import numpy as np
# # from transformers import GenerationConfig, StoppingCriteriaList

# # import os

# # TORCH_DTYPE_MAP = dict(
# #     fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

# # def parse_args():
# #     parser = argparse.ArgumentParser(description='Test model')
# #     parser.add_argument('config', help='config file name or path.')
# #     parser.add_argument('--checkpoint', default=None, help='checkpoint file')
# #     parser.add_argument('--test_slide_csv', default=None, help='test_slide_csv')
# #     parser.add_argument('--test_output_csv', default=None, help='test_output_csv')
# #     parser.add_argument(
# #     '--torch-dtype',
# #     default='bf16',
# #     choices=TORCH_DTYPE_MAP.keys(),
# #     help='Override the default `torch.dtype` and load the model under '
# #     'a specific `dtype`.')
# #     parser.add_argument(
# #         '--work-dir',
# #         help='the directory to save the file containing evaluation metrics')
# #     parser.add_argument(
# #         '--cfg-options',
# #         nargs='+',
# #         action=DictAction,
# #         help='override some settings in the used config, the key-value pair '
# #         'in xxx=yyy format will be merged into config file. If the value to '
# #         'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
# #         'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
# #         'Note that the quotation marks are necessary and that no white space '
# #         'is allowed.')
# #     parser.add_argument(
# #         '--launcher',
# #         choices=['none', 'pytorch', 'slurm', 'mpi'],
# #         default='none',
# #         help='job launcher')
# #     parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
# #     args = parser.parse_args()
# #     if 'LOCAL_RANK' not in os.environ:
# #         os.environ['LOCAL_RANK'] = str(args.local_rank)
# #     return args

 
# # def register_function(cfg_dict):
# #     if isinstance(cfg_dict, dict):
# #         for key, value in dict.items(cfg_dict):
# #             if isinstance(value, FunctionType):
# #                 value_str = str(value)
# #                 if value_str not in MAP_FUNC:
# #                     MAP_FUNC.register_module(module=value, name=value_str)
# #                 cfg_dict[key] = value_str
# #             else:
# #                 register_function(value)
# #     elif isinstance(cfg_dict, (list, tuple)):
# #         for value in cfg_dict:
# #             register_function(value)

# # def main():
# #     torch.cuda.set_device(0)
# #     args = parse_args()

# #     # parse config
# #     if not osp.isfile(args.config):
# #         try:
# #             args.config = cfgs_name_path[args.config]
# #         except KeyError:
# #             raise FileNotFoundError(f'Cannot find {args.config}')

# #     # load config
# #     cfg = Config.fromfile(args.config)
# #     cfg.launcher = args.launcher
# #     if args.cfg_options is not None:
# #         cfg.merge_from_dict(args.cfg_options)

# #     # register FunctionType object in cfg to `MAP_FUNC` Registry and
# #     # change these FunctionType object to str
# #     register_function(cfg._cfg_dict)

# #     # work_dir is determined in this priority: CLI > segment in file > filename
# #     if args.work_dir is not None:
# #         # update configs according to CLI args if args.work_dir is not None
# #         cfg.work_dir = args.work_dir
# #     elif cfg.get('work_dir', None) is None:
# #         # use config filename as default work_dir if cfg.work_dir is None
# #         cfg.work_dir = osp.join('./work_dirs',
# #                                 osp.splitext(osp.basename(args.config))[0])

# #     # build the runner from config
# #     if 'runner_type' not in cfg:
# #         # build the default runner
# #         runner = Runner.from_cfg(cfg)
# #     else:
# #         # build customized runner from the registry
# #         # if 'runner_type' is set in the cfg
# #         runner = RUNNERS.build(cfg)
    
# #     model_kwargs = {
# #     'trust_remote_code': True,
# #     'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
# #     }

# #     state_dict = guess_load_checkpoint(args.checkpoint)
# #     # state_dict = torch.load(args.checkpoint, map_location='cpu')

# #     runner.model.load_state_dict(state_dict, strict=False)
# #     runner.model.eval()
# #     runner.logger.info(f'Load checkpoint from {args.checkpoint}')


# #     llm_name_or_path = 'Qwen/Qwen2.5-7B-Instruct'
# #     tokenizer = AutoTokenizer.from_pretrained(
# #                 llm_name_or_path,
# #                 trust_remote_code=True,
# #                 encode_special_tokens=True)

# #     llm = runner.model.llm
# #     llm.eval()

# #     LongNet_encoder = runner.model.LongNet_encoder.to(model_kwargs['torch_dtype'])
# #     LongNet_encoder.cuda()
# #     LongNet_encoder.eval()    

# #     projector = runner.model.projector.to(model_kwargs['torch_dtype'])
# #     projector.cuda()
# #     projector.eval()


# #     df_test_case = pd.read_csv(args.test_slide_csv)

# #     df_test_case['Output'] = df_test_case.apply(lambda x: '', axis=1)
# #     columns = ['ID','Slide','Tumor','Broad Category','Narrow Category','Question','A','B','C','D','Answer','Output']
# #     df_test_output = pd.DataFrame(columns=columns)
    
# #     for i in range(df_test_case.shape[0]):
        
# #         print('*'*30)
# #         print('id: ', i)
# #         case_name = df_test_case.loc[i, 'Slide']
# #         test_image_file = "TCGA_patch_feat/" + df_test_case.loc[i, 'Tumor'] + "/" + case_name + ".csv"
# #         if test_image_file.endswith('.csv'):
# #             image = pd.read_csv(test_image_file)
# #             image = image.iloc[:, :512]
# #             total_rows = image.shape[0]
# #             sample_num = 38400
# #             if total_rows >= sample_num:
# #                 indices = np.linspace(0, total_rows - 1, sample_num, dtype=int)
# #                 sampled_df = image.iloc[indices]
# #                 image = sampled_df.iloc[:sample_num]
# #             image = image.to_numpy().reshape(1, image.shape[0], 512)
# #             image = torch.from_numpy(image)
# #         else:
# #             image = Image.open(test_image_file).convert('RGB')
  
# #         image = image.cuda()
# #         prompt_template = PROMPT_TEMPLATE.qwen_chat
# #         SYSTEM = ''
# #         question = df_test_case.loc[i, 'Question']
# #         options = []
# #         for opt in ['A', 'B', 'C', 'D']:
# #             if pd.notna(df_test_case.loc[i, opt]):
# #                 options.append(f"{opt}. {df_test_case.loc[i, opt]}")
# #         options_str = '\n'.join(options)

# #         sample_input = f"{question}\n{options_str}"
# #         print('Input: ', sample_input)

# #         instruction = prompt_template.get('INSTRUCTION', '{input}')
# #         sample_input = DEFAULT_IMAGE_TOKEN + '\n' + sample_input
# #         inputs = (SYSTEM + instruction).format(input=sample_input, round=1, **runner.cfg)
# #         chunk_encode = []
# #         for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
# #             if idx == 0:
# #                 cur_encode = tokenizer.encode(chunk)
# #             else:
# #                 cur_encode = tokenizer.encode(
# #                     chunk, add_special_tokens=False)
# #             chunk_encode.append(cur_encode)
# #         assert len(chunk_encode) == 2
# #         input_ids = []
# #         for idx, cur_chunk_encode in enumerate(chunk_encode):
# #             input_ids.extend(cur_chunk_encode)
# #             if idx != len(chunk_encode) - 1:
# #                 input_ids.append(IMAGE_TOKEN_INDEX)
# #         input_ids = torch.tensor(input_ids).cuda()


# #         image = runner.model.LongNet_encoder(src_tokens=None, token_embeddings=image.permute(1, 0, 2).to(runner.model.llm.dtype))["encoder_out"]
# #         image = image.permute(1, 0, 2)

# #         pixel_values = runner.model.projector(image)
# #         mm_inputs = prepare_inputs_labels_for_multimodal(
# #             llm=runner.model.llm,
# #             input_ids=input_ids.unsqueeze(0),
# #             pixel_values=pixel_values)

# #         max_new_tokens=500
# #         gen_config = GenerationConfig(
# #             max_new_tokens=max_new_tokens,
# #             do_sample=False,
# #             eos_token_id=tokenizer.eos_token_id,
# #             pad_token_id=tokenizer.pad_token_id
# #             if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
# #         )
# #         stop_words=[]
# #         stop_words += prompt_template.get('STOP_WORDS', [])
# #         stop_criteria = StoppingCriteriaList()
# #         for word in stop_words:
# #             stop_criteria.append(
# #                 StopWordStoppingCriteria(tokenizer, word))

# #         generate_output = llm.generate(
# #             **mm_inputs,
# #             generation_config=gen_config,
# #             streamer=None,
# #             bos_token_id=tokenizer.bos_token_id,
# #             stopping_criteria=stop_criteria)

# #         generation_output = tokenizer.decode(generate_output[0])
# #         if generation_output.endswith('<|im_end|>'):
# #             generation_output = generation_output[:-10]
            
# #         print('Output: ', generation_output)

# #         add_row = {
# #             'ID': df_test_case.loc[i, 'ID'],
# #             'Slide': df_test_case.loc[i, 'Slide'],
# #             'Tumor': df_test_case.loc[i, 'Tumor'],
# #             'Broad Category': df_test_case.loc[i, 'Broad Category'],
# #             'Narrow Category': df_test_case.loc[i, 'Narrow Category'],
# #             'Question': question,
# #             'A': df_test_case.loc[i, 'A'],
# #             'B': df_test_case.loc[i, 'B'],
# #             'C': df_test_case.loc[i, 'C'],
# #             'D': df_test_case.loc[i, 'D'],
# #             'Answer': df_test_case.loc[i, 'Answer'],
# #             'Output': generation_output
# #         }
# #         df_test_output.loc[i] = add_row
# #         df_test_output.to_csv(args.test_output_csv)
        
# #     print('Test ok!')

# # if __name__ == '__main__':
# #     main()

# import argparse
# import os
# import os.path as osp
# import torch
# import pandas as pd
# import numpy as np
# from transformers import AutoTokenizer, GenerationConfig, StoppingCriteriaList
# from mmengine.config import Config, DictAction
# from mmengine.registry import RUNNERS
# from mmengine.runner import Runner

# from xtuner.configs import cfgs_name_path
# from xtuner.model.utils import guess_load_checkpoint, prepare_inputs_labels_for_multimodal
# from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
#                           StopWordStoppingCriteria, PROMPT_TEMPLATE)


# TORCH_DTYPE_MAP = dict(
#     fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

# def parse_args():
#     parser = argparse.ArgumentParser(description='Generate caption for WSI features')
#     parser.add_argument('config', help='Config file')
#     parser.add_argument('--checkpoint', help='Path to model checkpoint')
#     parser.add_argument('--test_slide_csv', help='Path to feature CSV (with patch_name)')
#     parser.add_argument('--test_output_csv', help='Output path for generated report')
#     parser.add_argument('--torch-dtype', default='bf16', choices=TORCH_DTYPE_MAP.keys())
#     parser.add_argument('--local_rank', type=int, default=0)
#     parser.add_argument('--cfg-options', nargs='+', action=DictAction)
#     return parser.parse_args()


# def main():
#     args = parse_args()
#     torch.cuda.set_device(args.local_rank)
#     os.environ['LOCAL_RANK'] = str(args.local_rank)

#     if not osp.isfile(args.config):
#         args.config = cfgs_name_path.get(args.config, args.config)
#     cfg = Config.fromfile(args.config)
#     cfg.work_dir = cfg.get('work_dir', './work_dirs/gen_caption')
#     cfg.launcher = 'none'
#     if args.cfg_options:
#         cfg.merge_from_dict(args.cfg_options)

#     runner = Runner.from_cfg(cfg)

#     # Load checkpoint
#     state_dict = guess_load_checkpoint(args.checkpoint)
#     runner.model.load_state_dict(state_dict, strict=False)
#     runner.model.eval()
#     runner.logger.info(f'✅ Loaded checkpoint: {args.checkpoint}')
    
#     model_kwargs = {
#     'trust_remote_code': True,
#     'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
#     }
#     # Model components
#     model = runner.model
#     tokenizer = AutoTokenizer.from_pretrained(
#         'Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True, encode_special_tokens=True)
#     model.LongNet_encoder = model.LongNet_encoder.to(model_kwargs['torch_dtype'])

#     model.LongNet_encoder.cuda().eval()
#     model.projector.cuda().eval()
#     model.llm.cuda().eval()

#     # === Step 1: Load features
#     df = pd.read_csv(args.test_slide_csv)
#     features = df.iloc[:, :512].to_numpy().astype(np.float32)
#     patch_names = df.iloc[:, 512].tolist() if df.shape[1] > 512 else None

#     N = features.shape[0]
#     features_tensor = torch.tensor(features).unsqueeze(0).cuda()  # [1, N, 512]
#     src_tokens = torch.arange(N).unsqueeze(0).cuda()              # [1, N]

#     # === Step 2: Encode & Project
#     with torch.no_grad():
#         enc_out = model.LongNet_encoder(
#             src_tokens=None,
#             token_embeddings=features_tensor.permute(1, 0, 2).to(model.llm.dtype)
#         )["encoder_out"]  # [N, 1, D]
#         enc_out = enc_out.permute(1, 0, 2)  # [1, N, D]
#         pixel_values = model.projector(enc_out)

#     # === Step 3: Prompt for caption
#     prompt_template = PROMPT_TEMPLATE.qwen_chat
#     instruction = prompt_template.get('INSTRUCTION', '{input}')
#     SYSTEM = ''
#     input_text = 'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.'
#     prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_text
#     final_input = (SYSTEM + instruction).format(input=prompt, round=1, **cfg)

#     # Tokenize input
#     chunk_encode = []
#     for idx, chunk in enumerate(final_input.split(DEFAULT_IMAGE_TOKEN)):
#         if idx == 0:
#             cur_encode = tokenizer.encode(chunk)
#         else:
#             cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
#         chunk_encode.append(cur_encode)

#     input_ids = []
#     for idx, cur_chunk_encode in enumerate(chunk_encode):
#         input_ids.extend(cur_chunk_encode)
#         if idx != len(chunk_encode) - 1:
#             input_ids.append(IMAGE_TOKEN_INDEX)
#     input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

#     # === Step 4: Generate
#     mm_inputs = prepare_inputs_labels_for_multimodal(
#         llm=model.llm, input_ids=input_ids, pixel_values=pixel_values)

#     stop_criteria = StoppingCriteriaList()
#     for word in prompt_template.get('STOP_WORDS', []):
#         stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))

#     gen_cfg = GenerationConfig(
#         max_new_tokens=512,
#         do_sample=False,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
#     )

#     output_ids = model.llm.generate(
#         **mm_inputs,
#         generation_config=gen_cfg,
#         stopping_criteria=stop_criteria,
#         streamer=None,
#         bos_token_id=tokenizer.bos_token_id
#     )

#     output_text = tokenizer.decode(output_ids[0])
#     output_text = output_text.split('<|im_end|>')[0].strip()

#     # === Step 5: Save
#     print('\n🧾 Generated Report:\n' + '-'*60)
#     print(output_text + '\n' + '-'*60)

#     output_df = pd.DataFrame({'report': [output_text]})
#     output_df.to_csv(args.test_output_csv, index=False)
#     print(f'✅ Report saved to {args.test_output_csv}')


# if __name__ == '__main__':
#     main()


import argparse
import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig, StoppingCriteriaList
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint, prepare_inputs_labels_for_multimodal
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria, PROMPT_TEMPLATE)

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='Batch generate caption from patch feature csv list')
    parser.add_argument('config', help='Config file')
    parser.add_argument('--checkpoint', help='Path to model checkpoint')
    parser.add_argument('--csv_list_file', help='Path to main csv listing all feature .csv paths')
    parser.add_argument('--output_json', help='Path to output json file')
    parser.add_argument('--torch-dtype', default='bf16', choices=TORCH_DTYPE_MAP.keys())
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()

def generate_report(model, tokenizer, features_tensor, model_dtype, cfg):
    with torch.no_grad():
        enc_out = model.LongNet_encoder(
            src_tokens=None,
            token_embeddings=features_tensor.permute(1, 0, 2).to(model_dtype)
        )["encoder_out"]
        enc_out = enc_out.permute(1, 0, 2)
        pixel_values = model.projector(enc_out)

    prompt_template = PROMPT_TEMPLATE.qwen_chat
    instruction = prompt_template.get('INSTRUCTION', '{input}')
    SYSTEM = ''
    input_text = 'Generate an overview summarizing the principal findings from the pathology examination of the whole slide image.'
    prompt = DEFAULT_IMAGE_TOKEN + '\n' + input_text 
    final_input = (SYSTEM + instruction).format(input=prompt, round=1, **cfg)

    #print(final_input)

    chunk_encode = []
    for idx, chunk in enumerate(final_input.split(DEFAULT_IMAGE_TOKEN)):
        cur_encode = tokenizer.encode(chunk, add_special_tokens=(idx == 0))
        chunk_encode.append(cur_encode)

    input_ids = []
    for idx, cur_chunk in enumerate(chunk_encode):
        input_ids.extend(cur_chunk)
        if idx != len(chunk_encode) - 1:
            input_ids.append(IMAGE_TOKEN_INDEX)
    input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()

    mm_inputs = prepare_inputs_labels_for_multimodal(
        llm=model.llm, input_ids=input_ids, pixel_values=pixel_values)

    stop_criteria = StoppingCriteriaList()
    for word in prompt_template.get('STOP_WORDS', []):
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))

    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # gen_cfg = GenerationConfig(
    #     max_new_tokens=512,
    #     do_sample=True,
    #     temperature=0.1,
    #     top_p=0.9,
    #     repetition_penalty=1.1,
    #     no_repeat_ngram_size=3,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    # )

    output_ids = model.llm.generate(
        **mm_inputs,
        generation_config=gen_cfg,
        stopping_criteria=stop_criteria,
        streamer=None,
        bos_token_id=tokenizer.bos_token_id
    )

    output_text = tokenizer.decode(output_ids[0])
    output_text = output_text.split('<|im_end|>')[0].strip()
    return output_text

def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    os.environ['LOCAL_RANK'] = str(args.local_rank)

    if not osp.isfile(args.config):
        args.config = cfgs_name_path.get(args.config, args.config)
    cfg = Config.fromfile(args.config)
    cfg.work_dir = cfg.get('work_dir', './work_dirs/gen_caption')
    cfg.launcher = 'none'
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    runner = Runner.from_cfg(cfg)

    # Load model and checkpoint
    state_dict = guess_load_checkpoint(args.checkpoint)
    runner.model.load_state_dict(state_dict, strict=False)
    runner.model.eval()
    runner.logger.info(f'✅ Loaded checkpoint: {args.checkpoint}')
    
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    model = runner.model
    tokenizer = AutoTokenizer.from_pretrained(
        'Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True, encode_special_tokens=True)

    model.LongNet_encoder = model.LongNet_encoder.to(model_kwargs['torch_dtype']).cuda().eval()
    model.projector = model.projector.cuda().eval()
    model.llm = model.llm.cuda().eval()

    # === Step 1: Read list of csv paths
    df_path_list = pd.read_csv(args.csv_list_file)
    if 'path' not in df_path_list.columns:
        raise ValueError("Main CSV must contain a column named 'path' with paths to feature CSV files.")

    result_list = []
    for idx, row in tqdm(df_path_list.iloc[:1000].iterrows(), total=min(1000, len(df_path_list)), desc='Generating reports'):

    #for idx, row in tqdm(df_path_list.iterrows(), total=len(df_path_list), desc='Generating reports'):
        feat_csv_path = row['path']
        if not osp.isfile(feat_csv_path):
            tqdm.write(f"❌ Skipping: {feat_csv_path} (not found)")
            continue

        try:
            df_feat = pd.read_csv(feat_csv_path)
            features = df_feat.iloc[:, :512].to_numpy().astype(np.float32)
            features_tensor = torch.tensor(features).unsqueeze(0).cuda()
            report = generate_report(model, tokenizer, features_tensor, model.llm.dtype, cfg) 
            slide_id = osp.basename(feat_csv_path).replace('_pry.csv', '.tiff')
            result_list.append({'id': slide_id, 'report': report})
            tqdm.write(f"✅ [{slide_id}] Done")

        except Exception as e:
            tqdm.write(f"❌ Error processing {feat_csv_path}: {e}")
            continue

    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2)
    print(f'\n✅ All reports saved to {args.output_json}')

if __name__ == '__main__':
    main()



