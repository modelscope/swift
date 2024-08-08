# Copyright (c) Alibaba, Inc. and its affiliates.
import re
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import json
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from packaging import version
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, StoppingCriteria
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from swift.llm.agent.utils import calculate_loss_scale, get_tools_prompt
from swift.torchacc_utils import pad_and_split_batch
from swift.utils import get_dist_setting, upper_bound, use_torchacc

DEFAULT_SYSTEM = 'You are a helpful assistant.'
History = List[Union[Tuple[str, str], List[str]]]
Prompt = List[Union[str, List[int], List[str]]]
StopWords = Prompt
Context = Union[str, List[int]]
TEMPLATE_MAPPING: Dict[str, Dict[str, Any]] = {}


class TemplateType:
    # text-generation
    default_generation = 'default-generation'
    chatglm_generation = 'chatglm-generation'
    qwen_audio_generation = 'qwen-audio-generation'
    # chat
    default = 'default'
    qwen = 'qwen'
    qwen_vl = 'qwen-vl'
    qwen_audio = 'qwen-audio'
    modelscope_agent = 'modelscope-agent'
    baichuan = 'baichuan'
    chatglm2 = 'chatglm2'
    chatglm3 = 'chatglm3'
    codegeex4 = 'codegeex4'
    llama = 'llama'  # llama2
    llama3 = 'llama3'
    llava1_5 = 'llava1_5'
    llava_mistral = 'llava-mistral'
    llava_vicuna = 'llava-vicuna'
    llava_yi = 'llava-yi'
    llava_llama_instruct = 'llava-llama-instruct'
    llava_qwen_instruct = 'llava-qwen-instruct'
    llama_llava_next = 'llama-llava-next'
    llava_next_video = 'llava-next-video'
    llava_next_video_yi = 'llava-next-video-yi'
    mistral_nemo = 'mistral-nemo'
    openbuddy = 'openbuddy'
    openbuddy2 = 'openbuddy2'
    internlm = 'internlm'
    internlm2 = 'internlm2'
    internlm_xcomposer2 = 'internlm-xcomposer2'
    internlm_xcomposer2_4khd = 'internlm-xcomposer2-4khd'
    internlm_xcomposer2_5 = 'internlm-xcomposer2_5'
    internvl = 'internvl'
    internvl2 = 'internvl2'
    internvl_phi3 = 'internvl-phi3'
    internvl2_phi3 = 'internvl2-phi3'
    florence = 'florence'
    yi = 'yi'
    yi1_5 = 'yi1_5'
    yi_vl = 'yi-vl'
    yuan = 'yuan'
    xverse = 'xverse'
    ziya = 'ziya'
    skywork = 'skywork'
    bluelm = 'bluelm'
    zephyr = 'zephyr'
    sus = 'sus'
    deepseek = 'deepseek'
    numina_math = 'numina-math'
    deepseek_coder = 'deepseek-coder'
    deepseek_vl = 'deepseek-vl'
    deepseek2 = 'deepseek2'
    codefuse_codellama = 'codefuse-codellama'
    codefuse = 'codefuse'
    cogvlm = 'cogvlm'
    cogvlm2_video = 'cogvlm2-video'
    glm4v = 'glm4v'
    cogagent_chat = 'cogagent-chat'
    cogagent_instruct = 'cogagent-instruct'
    orion = 'orion'
    minicpm = 'minicpm'
    minicpm_v = 'minicpm-v'
    minicpm_v_v2_5 = 'minicpm-v-v2_5'
    minicpm_v_v2_6 = 'minicpm-v-v2_6'
    gemma = 'gemma'
    paligemma = 'paligemma'
    mplug_owl2 = 'mplug-owl2'
    wizardlm2_awq = 'wizardlm2-awq'
    wizardlm2 = 'wizardlm2'
    atom = 'atom'
    phi3 = 'phi3'
    phi3_vl = 'phi3-vl'
    telechat = 'telechat'
    telechat_v2 = 'telechat-v2'
    dbrx = 'dbrx'
    mengzi = 'mengzi'
    c4ai = 'c4ai'
    chatml = 'chatml'
    # compatibility. (Deprecated)
    default_generation_bos = 'default-generation-bos'

    @classmethod
    def get_template_name_list(cls) -> List[str]:
        res = []
        for k in cls.__dict__.keys():
            if k.startswith('__') or k == 'get_template_name_list':
                continue
            res.append(cls.__dict__[k])
        return res


class StopWordsCriteria(StoppingCriteria):
    # The returned sentence includes stop words.
    def __init__(self, tokenizer: PreTrainedTokenizerBase, stop_words: StopWords, **tokenizer_kwargs) -> None:
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.tokenizer_kwargs = tokenizer_kwargs
        self.start_idx = -1

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs) -> bool:
        if self.start_idx == -1:
            self.start_idx = len(input_ids[0]) - 1
        tokenizer = self.tokenizer
        stop_words = self.stop_words
        # [-20:]: Assuming the end tokens do not exceed 20 tokens,
        #   to avoid input_ids being too long and affecting efficiency.
        text = tokenizer.decode(input_ids[0, self.start_idx:][-20:], **self.tokenizer_kwargs)
        for stop_word in stop_words:
            if isinstance(stop_word, str):
                if stop_word in text:
                    return True
            else:  # list
                if len(stop_word) > 0 and input_ids[0].tolist()[-len(stop_word):] == stop_word:
                    return True
        return False


class Template:
    """A template class for all supported models.

    Args:
        prefix: Prefix tokens before the first turn's prompt
        prompt: A list of elements whose types are str and list of integers. The input query part of every turn.
        chat_sep: The chat separators between every turn.
        suffix: The end tokens after the chat finished.
        default_system: A default system instruction.
        system_prefix: The prefix if the `system` is not empty.
        auto_add_bos: By default, the bos_token is not added. The auto_add_bos option will determine
            whether to add it based on `tokenizer.encode('')`.

        Examples:
            <start>system\nYou are a helpful assistant!<end>\n<bos><start>Who are you?<end>\n<start>assistant:I am a robot<end>\n<start>Who are you?<end>\n<start>assistant:I am a robot<end> # noqa
            --------------- --------------------------         ---  ----- ------------ ----------------------- ----------- ----                                                         -----
             system_prefix          system                   prefix prompt   query              prompt           response chat_sep                                                      suffix
    """

    special_tokens = ['<image>', '<video>', '<audio>', '<bbox>', '<ref-object>']
    special_keys = ['images', 'videos', 'audios', 'objects']
    grounding_type = 'norm_1000'
    image_placeholder = ['<image>']
    load_medias = True

    def __init__(self,
                 prefix: Prompt,
                 prompt: Prompt,
                 chat_sep: Optional[Prompt],
                 suffix: Prompt,
                 default_system: Optional[str] = None,
                 system_prefix: Optional[Prompt] = None,
                 auto_add_bos: bool = False,
                 tools_prompt: str = 'react_en',
                 tool_prompt: Optional[Prompt] = None,
                 padding_side: Literal['left', 'right'] = 'right') -> None:
        # check
        for x in [prefix, prompt, chat_sep, suffix, system_prefix]:
            assert x is None or isinstance(x, list)

        if default_system == '':
            default_system = None
        if self._has_system(prefix):
            assert system_prefix is None, 'The prefix already contains {{SYSTEM}}.'
            system_prefix = prefix
            prefix = self._replace_system(prefix)
        self.prefix = prefix
        self.system_prefix = system_prefix
        if self.system_prefix is None and not any(['{{SYSTEM}}' in context for context in prompt]):
            assert default_system is None, 'The template does not support `system`.'
        self.prompt = prompt
        self.chat_sep = chat_sep
        self.support_multi_round = self.chat_sep is not None
        self.suffix = suffix
        self.default_system = default_system
        self.use_default_system = True
        self.auto_add_bos = auto_add_bos
        self._is_init = False
        self.tools_prompt = tools_prompt
        self.tool_prompt = tool_prompt if tool_prompt is not None else self.prompt  # default as user
        self._is_vllm = False
        self._is_lmdeploy = False
        self.padding_side = padding_side

    @staticmethod
    def _replace_system(prefix: Prompt) -> Prompt:
        return [p.replace('{{SYSTEM}}', '') for p in prefix if '{{SYSTEM}}' in p]

    @staticmethod
    def _has_system(prefix: Prompt) -> bool:
        return any(['{{SYSTEM}}' in p for p in prefix])

    @staticmethod
    def _preprocess_prompt(tokenizer: PreTrainedTokenizerBase, value: Optional[Prompt]) -> Optional[Prompt]:
        """Turn `eos_token_id` to token id

        e.g. [['eos_token_id']] -> [[2]]
        """
        if value is None:
            return None
        res_value = []
        for v in value:
            if isinstance(v, list):
                res_v = []
                for sub_v in v:
                    if isinstance(sub_v, str):
                        sub_v = getattr(tokenizer, sub_v)
                    res_v.append(sub_v)
                v = res_v
            res_value.append(v)
        return res_value

    def _init_template(self,
                       tokenizer: PreTrainedTokenizerBase,
                       default_system: Optional[str] = None,
                       max_length: Optional[int] = None,
                       truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
                       model=None,
                       **kwargs) -> None:
        assert self._is_init is False, 'The template has been initialized.'
        self.is_multimodal = getattr(tokenizer, 'is_multimodal', None)
        self._is_init = True
        self.tokenizer = tokenizer
        # if default_system is None. not change self.default_system
        if default_system == '':
            self.default_system = None
        elif default_system is not None:
            assert self.system_prefix is not None, (
                f'The template does not support `system`, template_type: {getattr(self, "template_type", None)}')
            self.default_system = default_system
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.model = model
        self.use_loss_scale = kwargs.get('use_loss_scale', False)
        self.response_loss_scale_map = kwargs.get('loss_scale_map', None)
        self.query_loss_scale_map = None
        if self.response_loss_scale_map is not None:
            if 'query' in self.response_loss_scale_map and isinstance(self.response_loss_scale_map['query'], dict):
                self.query_loss_scale_map = self.response_loss_scale_map['query']
            if 'response' in self.response_loss_scale_map and isinstance(self.response_loss_scale_map['response'],
                                                                         dict):
                self.response_loss_scale_map = self.response_loss_scale_map['response']

        self.sequence_parallel_size = kwargs.get('sequence_parallel_size', 1)
        self.rescale_image = kwargs.get('rescale_image', -1)

        for key in ['prefix', 'prompt', 'chat_sep', 'suffix', 'system_prefix']:
            value = getattr(self, key)
            value = self._preprocess_prompt(tokenizer, value)
            setattr(self, key, value)

    def check_example(self, example: Dict[str, Any]) -> None:
        pass

    def add_default_tags(self, example: Dict[str, Any]) -> None:
        history: History = deepcopy(example.get('history') or [])
        query: str = example.get('query') or ''
        for media_key, media_tag in [('videos', '<video>'), ('images', '<image>'), ('audios', '<audio>')]:
            if example.get(media_key):
                infer_media_type = TEMPLATE_MAPPING[self.template_type].get('infer_media_type')
                if infer_media_type == 'round':
                    n_round = len(example[media_key])
                    assert n_round == len(history) + 1
                    for i, h, m in zip(range(n_round), history + [[query, None]], example[media_key]):
                        num_media_tags = len(re.findall(media_tag, h[0]))
                        if m:
                            assert num_media_tags <= 1, (
                                'The model includes at most one media per round. However, '
                                f'this round contains {num_media_tags} media_tags. query: {h[0]}')
                            if num_media_tags == 0:
                                h[0] = media_tag + h[0]
                        else:
                            assert num_media_tags == 0, f'Missing media. query: {h[0]}'
                        if i == n_round - 1:
                            query = h[0]
                        else:
                            history[i][0] = h[0]

                    example[media_key] = [m for m in example[media_key] if m]

                else:
                    num_media_tags = len(re.findall(media_tag, '\n'.join([h[0] for h in history]) + f'\n{query}'))
                    example[media_key] = [m for m in example[media_key] if m]
                    num_media = len(example[media_key])
                    num_new_tags = num_media - num_media_tags
                    assert num_new_tags >= 0, (f'Number of media: {num_media}, number of media_tags: {num_media_tags}')
                    if history:
                        history[0][0] = media_tag * num_new_tags + history[0][0]
                    else:
                        query = media_tag * num_new_tags + query

        example['query'] = query
        example['history'] = history

    def _preprocess_media(self, example):
        from .media import MediaTag
        # Format media_keys to list
        for media_key in MediaTag.media_keys.values():
            if example.get(media_key) and not isinstance(example[media_key], (tuple, list)):
                # change images field to list
                example[media_key] = [example[media_key]]

        # Parse <img></img> format images and merged into images key
        images_path = None
        if self.is_multimodal in {True, None}:  # If False, do not perform replace_img_tag
            example['query'], example['history'], images_path = replace_img_tag(
                example.get('query'),
                example.get('history') or [], '<image>')
        if images_path:
            example['images'] = example.get('images', []) + images_path

        # audio, video
        if self.is_multimodal in {True, None}:
            for k, tag, pattern in zip(['audios', 'videos'], ['<audio>', '<video>'],
                                       [r'<audio>(.+?)</audio>', r'<video>(.+?)</video>']):
                example['query'], example['history'], medias_path = replace_img_tag(
                    example.get('query'),
                    example.get('history') or [], tag, pattern)
                if medias_path:
                    example[k] = example.get(k, []) + medias_path

        # Add default tags to examples to note where to put the medias into the sequence
        self.add_default_tags(example)

        # Format objects(groundings/refs) to json
        if example.get('objects') and isinstance(example['objects'], str):
            # reload grounding from str
            example['objects'] = json.loads(example['objects'])
            objects = []
            for object in example['objects']:
                # Compatible with list format
                if isinstance(object, list):
                    object = {
                        'caption': object[0],
                        'bbox': object[1],
                        'bbox_type': None,
                        'image': 0,
                    }
                objects.append(object)
            example['objects'] = objects

        # Load image into PIL format
        from .vision_utils import load_image, rescale_image, _read_batch
        images = example.get('images') or []
        if images:
            if example.get('objects') or self.load_medias or self._is_lmdeploy or self._is_vllm:
                images = _read_batch(images, load_image)
            if example.get('objects'):
                # Normalize grounding bboxes
                self.normalize_bbox(example['objects'], images, to_type=self.grounding_type)
            if self.load_medias and self.grounding_type != 'real':
                images = [rescale_image(img, self.rescale_image) for img in images]
            example['images'] = images

        # Check the example that whether matching the very template's rules
        self.check_example(example)

    def preprocess(self, example):
        # Duplicate example and create a new one to prepare in-place changes
        example = example.copy()
        template_type: Optional[str] = getattr(self, 'template_type', None)
        tools: Union[List[Any], str] = example.get('tools') or []

        # Template needs to be initialized
        if not self._is_init:
            raise ValueError(
                'Template is not initialized, please use the `get_template` function to obtain the template.')

        # Reset system (by default value and agent tools)
        system: Optional[str] = example.get('system', None)
        if system is None:
            if self.use_default_system:
                system = self.default_system
        elif system == '':
            system = None
        else:
            assert self.system_prefix is not None, (
                f'The template does not support `system`, template_type: {template_type}')
        if tools:
            if isinstance(tools, str):
                tools = json.loads(tools)
            if system is None:
                system = ''
            system += get_tools_prompt(tools, self.tools_prompt)

        example['system'] = system

        # Check whether this template supports multi-round
        history: History = example.get('history') or []
        if len(history) > 0:
            assert self.support_multi_round, (
                f'The template does not support multi-round chat, template_type: {template_type}')

        # Set history_roles
        history_roles: Optional[History] = example.get('history_roles')
        if history_roles is None:
            example['history_roles'] = [['user', 'assistant'] for _ in range(len(history))]

        self._preprocess_media(example)
        return example

    def encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        example = self.preprocess(example)
        _encode = self._encode
        if self._is_lmdeploy or self._is_vllm:
            assert self.is_multimodal is not None, 'Please use the get_model_tokenizer function.'
            _encode = MethodType(Template._encode, self)
        return _encode(example)

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        from lmdeploy.vl.constants import IMAGE_DUMMY_TOKEN_INDEX
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
        idx_list.insert(0, -1)
        new_input_ids = []
        ranges = []
        for i in range(len(idx_list) - 1):
            _range = []
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            _range.append(len(new_input_ids))
            new_input_ids += [IMAGE_DUMMY_TOKEN_INDEX] * images[i].shape[0]
            _range.append(len(new_input_ids))
            ranges.append(_range)
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_embeddings'] = images
        inputs['input_embedding_ranges'] = ranges
        inputs['input_ids'] = new_input_ids

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """return: inputs, tokenizer_kwargs"""
        query: str = example.get('query') or ''
        query_role: str = example.get('query_role') or 'user'
        response: Optional[str] = example.get('response')
        history: History = example.get('history') or []
        history_roles: Optional[History] = example.get('history_roles')
        system: Optional[str] = example.get('system', None)
        is_multi_modal: bool = any([example.get(key) for key in Template.special_keys])

        inputs, tokenizer_kwargs = self._concat_and_tokenize(
            query,
            query_role,
            response,
            history,
            history_roles,
            system,
            self.truncation_strategy,
            auto_add_bos=self.auto_add_bos,
            example=example,
            is_multi_modal=is_multi_modal)
        if self._is_lmdeploy or self._is_vllm:
            inputs['images'] = example.get('images')
        if inputs.get('labels') is None:
            inputs.pop('loss_scale', None)
        return inputs, tokenizer_kwargs

    def _concat_context_list(
            self,
            context_list: List[Context],
            res_context_list: List[Context],  # inplace
            loss_scale_list: List[float],  # inplace
            system: Optional[str] = None,
            query: Optional[str] = None,
            response: Optional[str] = None,
            round0: Optional[int] = None) -> None:
        # concat context list and replace placeholder
        round1 = None
        if round0 is not None:
            round1 = str(round0 + 1)
            round0 = str(round0)
        for context in context_list:
            if isinstance(context, str):
                if '{{RESPONSE}}' == context:
                    assert response is not None
                    content_part, weight_part = calculate_loss_scale(query, response, self.use_loss_scale,
                                                                     self.response_loss_scale_map,
                                                                     self.query_loss_scale_map)
                    res_context_list.extend(content_part)
                    loss_scale_list.extend(weight_part)
                    continue
                old_str_list = ['{{SYSTEM}}', '{{QUERY}}', '{{ROUND0}}', '{{ROUND1}}']
                new_str_list = [system, query, round0, round1]
                for (old_str, new_str) in zip(old_str_list, new_str_list):
                    if new_str is not None and old_str in context:
                        context = context.replace(old_str, new_str)
            if len(context) == 0:
                continue
            res_context_list.append(context)
            loss_scale_list.append(0.)

    def _simplify_context_list(self, context_list: List[Context], loss_scale_list: List[float],
                               **kwargs) -> Tuple[List[Context], List[float]]:
        is_multi_modal: bool = kwargs.pop('is_multi_modal', False)

        if is_multi_modal:
            context_list, loss_scale_list = self.split_special_tokens(context_list, loss_scale_list)
        context_list, loss_scale_list = self.pre_tokenize(context_list, loss_scale_list, **kwargs)

        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list
        temp: List[str] = []
        temp_loss_scale = 0.
        for i, (context, loss_scale) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str) and (loss_scale == temp_loss_scale):
                temp.append(context)
            else:
                if len(temp) > 0:
                    res.append(''.join(temp))
                    res_loss_scale.append(temp_loss_scale)
                    temp.clear()
                if isinstance(context, str):  # loss_scale diff
                    temp.append(context)
                else:
                    res.append(context)
                    res_loss_scale.append(loss_scale)
                temp_loss_scale = loss_scale
        if len(temp) > 0:
            res.append(''.join(temp))
            res_loss_scale.append(temp_loss_scale)

        return res, res_loss_scale

    @staticmethod
    def split_special_tokens(context_list: List[Context],
                             loss_scale_list: List[float]) -> Tuple[List[Context], List[float]]:
        from swift.utils.utils import split_str_parts_by
        res: List[Context] = []
        loss_scale_res: List[float] = []
        from .utils import fetch_one
        for context, loss_scale in zip(context_list, loss_scale_list):
            contexts = []
            if isinstance(fetch_one(context), str):
                for d in split_str_parts_by(context, Template.special_tokens):
                    contexts.extend([d['key'], d['content']])
                contexts = [c for c in contexts if c]
                res.extend(contexts)
                loss_scale_res.extend([loss_scale] * len(contexts))
            else:
                res.append(context)
                loss_scale_res.append(loss_scale)
        return res, loss_scale_res

    def _tokenize(self, context, **tokenizer_kwargs):
        return self.tokenizer(
            context, return_attention_mask=False, add_special_tokens=False, **tokenizer_kwargs)['input_ids']

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        if media_type == 'image':
            if self._is_lmdeploy:
                return [[-100]]
            else:
                return self.image_placeholder
        if media_type == 'video':
            return ['<video>']
        if media_type == 'audio':
            return ['<audio>']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [object_['caption']]
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [f'({object_["bbox"][0]},{object_["bbox"][1]}),({object_["bbox"][2]},{object_["bbox"][3]})']
        else:
            return ['<bbox>']

    @classmethod
    def normalize_bbox(cls, objects, images, to_type: Literal['real', 'norm_1000', 'norm_1']):
        if not objects or not images:
            return

        for object in objects:
            bbox = object['bbox']
            bbox_type = object['bbox_type']
            idx = object['image']
            image = images[idx]
            if bbox_type == 'real':
                if to_type == 'real':
                    continue
                width, height = image.width, image.height
                object['bbox'] = [
                    int(coord / dim * 999) if to_type == 'norm_1000' else coord / dim
                    for coord, dim in zip(bbox, [width, height, width, height])
                ]
                object['bbox_type'] = to_type
            elif bbox_type == 'norm_1000':
                if to_type == 'norm_1000':
                    continue
                if to_type == 'norm_1':
                    object['bbox'] = [coord / 999. for coord in bbox]
                elif to_type == 'real':
                    width, height = image.width, image.height
                    object['bbox'] = [
                        int(coord / 999. * dim) for coord, dim in zip(bbox, [width, height, width, height])
                    ]
                object['bbox_type'] = to_type
            elif bbox_type == 'norm_1':
                if to_type == 'norm_1':
                    continue
                if to_type == 'norm_1000':
                    object['bbox'] = [int(coord * 999) for coord in bbox]
                elif to_type == 'real':
                    width, height = image.width, image.height
                    object['bbox'] = [int(coord * dim) for coord, dim in zip(bbox, [width, height, width, height])]
                object['bbox_type'] = to_type

    def pre_tokenize(self, context_list: List[Context], loss_scale_list: List[float],
                     **kwargs) -> Tuple[List[Context], List[float]]:
        # replace tag/object/box
        example = kwargs.get('example')  # get x_index
        res: List[Context] = []  # result of context_list
        res_loss_scale: List[float] = []  # result of loss_scale_list

        replace_tag = self.replace_tag
        if self._is_lmdeploy:
            replace_tag = MethodType(Template.replace_tag, self)

        for context, loss_scale in zip(context_list, loss_scale_list):
            if context == '<image>':
                c_list = replace_tag('image', example.get('image_index', 0), example)
                example['image_index'] = example.get('image_index', 0) + 1
            elif context == '<video>':
                c_list = replace_tag('video', example.get('video_index', 0), example)
                example['video_index'] = example.get('video_index', 0) + 1
            elif context == '<audio>':
                c_list = replace_tag('audio', example.get('audio_index', 0), example)
                example['audio_index'] = example.get('audio_index', 0) + 1
            elif context == '<ref-object>':
                c_list = self.replace_object(example.get('object_index', 0), example)
                example['object_index'] = example.get('object_index', 0) + 1
            elif context == '<bbox>':
                c_list = self.replace_box(example.get('box_index', 0), example)
                example['box_index'] = example.get('box_index', 0) + 1
            else:
                c_list = [context]
            res += c_list
            res_loss_scale += [loss_scale] * len(c_list)
        return res, res_loss_scale

    def _encode_context_list(
            self,
            context_list: List[Context],
            loss_scale_list: Optional[List[float]] = None) -> Tuple[List[int], List[int], List[float], Dict[str, Any]]:
        """return: input_ids, labels, tokenizer_kwargs"""
        input_ids: List[int] = []
        labels: List[int] = []
        loss_scale: List[float] = []
        tokenizer_kwargs = {}
        if loss_scale_list is None:
            loss_scale_list = [0.] * len(context_list)
        for i, (context, loss_weight) in enumerate(zip(context_list, loss_scale_list)):
            if isinstance(context, str):
                # tokenizer_kwargs is the returned tokenizer_kwargs,
                # while curr_tokenizer_kwargs is the tokenizer_kwargs for the current context.
                curr_tokenizer_kwargs = self._get_tokenizer_kwargs(context)
                self._concat_tokenizer_kwargs(tokenizer_kwargs, curr_tokenizer_kwargs)
                token_list = self._tokenize(context, **curr_tokenizer_kwargs)
            else:
                token_list = context
            input_ids += token_list
            if loss_scale_list[i] > 0.0:
                labels += token_list
            else:
                labels += [-100] * len(token_list)
            loss_scale.extend([loss_weight] * len(token_list))
        return input_ids, labels, loss_scale, tokenizer_kwargs

    def _concat_and_tokenize(self,
                             query: str,
                             query_role: str,
                             response: Optional[str],
                             history: History,
                             history_roles: History,
                             system: Optional[str],
                             truncation_strategy: str,
                             auto_add_bos: bool = False,
                             **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        return: inputs, tokenizer_kwargs
        """
        history = history.copy()

        res_context_list: List[Context] = []
        loss_scale_list: List[float] = []
        if auto_add_bos:
            bos_token_id = self.tokenizer.bos_token_id
            if isinstance(bos_token_id, int) and bos_token_id in self.tokenizer.encode(''):
                res_context_list.append([bos_token_id])
                loss_scale_list.append(0.)
        prompt = self.prompt.copy()
        if system is None:
            prompt = [context for context in prompt if '{{SYSTEM}}' not in context]
        if system is None or any(['{{SYSTEM}}' in context for context in prompt]):
            prefix = self.prefix
        else:
            prefix = self.system_prefix
        self._concat_context_list(prefix, res_context_list, loss_scale_list, system=system)

        history.append([query, response])
        history_roles.append([query_role, 'assistant'])

        # Set the loss_scale of chat_sep or suffix to 1 if efficient_eos.
        efficient_eos = False
        if self.chat_sep is not None and len(self.chat_sep) > 0:
            if isinstance(self.chat_sep[0], str) and isinstance(self.suffix[0], str) and self.chat_sep[0].startswith(
                    self.suffix[0]):
                efficient_eos = True
            elif isinstance(self.chat_sep[0], list) and self.chat_sep[0] == self.suffix[0]:
                efficient_eos = True

        for i, ((q, r), (qr, rr)) in enumerate(zip(history, history_roles)):
            context_list = self.tool_prompt.copy() if qr == 'tool' else prompt.copy()
            extra_context_list = []
            if i < len(history) - 1:
                context_list = [context for context in context_list if '{{SYSTEM}}' not in context]
                context_list.append('{{RESPONSE}}')
                if history[i + 1][0]:
                    extra_context_list = self.chat_sep
            elif r is not None:
                # last response
                context_list.append('{{RESPONSE}}')
                extra_context_list = self.suffix
                efficient_eos = True
            if q or r:
                self._concat_context_list(
                    context_list, res_context_list, loss_scale_list, query=q, response=r, system=system, round0=i)
                res_context_list += extra_context_list
                loss_scale_list += ([1.] if efficient_eos else [0.]) * len(extra_context_list)
        res_context_list, loss_scale_list = self._simplify_context_list(res_context_list, loss_scale_list, **kwargs)
        input_ids, labels, loss_scale, tokenizer_kwargs = self._encode_context_list(res_context_list, loss_scale_list)

        if response is None:
            labels = None

        if self.max_length is not None:
            if truncation_strategy == 'delete' and len(input_ids) > self.max_length:
                return {}, {}
            input_ids = input_ids[-self.max_length:]
            if labels is not None:
                labels = labels[-self.max_length:]
            if loss_scale is not None:
                loss_scale = loss_scale[-self.max_length:]
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
        }
        if self.use_loss_scale:
            inputs['loss_scale'] = loss_scale
        return inputs, tokenizer_kwargs

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        """return: curr_tokenizer_kwargs"""
        return {}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        assert len(tokenizer_kwargs) == 0

    @staticmethod
    def pad_sequence(sequences: List[Tensor],
                     padding_value: float = 0.,
                     padding_side: Literal['right', 'left'] = 'right'):
        padding_right = padding_side == 'right'
        if padding_right:
            return pad_sequence(sequences, batch_first=True, padding_value=padding_value)

        max_len = max([s.size(0) for s in sequences])

        padded_sequences = []
        for seq in sequences:
            pad_length = max_len - seq.size(0)
            pad_tuple = [0] * ((seq.dim() - 1) * 2) + [pad_length, 0]
            padded_seq = F.pad(seq, tuple(pad_tuple), 'constant', padding_value)
            padded_sequences.append(padded_seq)

        return torch.stack(padded_sequences)

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        """
        Args:
            batch(`List[Dict[str, Any]]`): The input data in batch
            padding_to(`int`, optional): Whether padding the batch to a fixed length, if none, the batch
                will be padded to the `longest`
        """
        tokenizer = self.tokenizer
        assert tokenizer.pad_token_id is not None
        inputs_embeds, input_ids = None, None
        if 'inputs_embeds' in batch[0]:
            inputs_embeds = [b['inputs_embeds'] for b in batch]
            attention_mask = [
                torch.ones((inputs_embeds[i].shape[0]), dtype=torch.int64) for i in range(len(inputs_embeds))
            ]
        else:
            input_ids = [torch.tensor(b['input_ids']) for b in batch]
            attention_mask = [torch.ones(len(input_ids[i]), dtype=torch.int64) for i in range(len(input_ids))]
        labels = [torch.tensor(b['labels']) for b in batch]
        loss_scale = [torch.tensor(b['loss_scale']) for b in batch] if 'loss_scale' in batch[0] else None
        padding_right = self.padding_side == 'right'

        if padding_to is not None:
            assert input_ids is not None  # inputs_embeds not support padding_to
            padding_len = padding_to - input_ids[0].shape[-1]
            if padding_len > 0:
                input_ids[0] = F.pad(input_ids[0], (0, padding_len) if padding_right else (padding_len, 0), 'constant',
                                     tokenizer.pad_token_id)
                attention_mask[0] = F.pad(attention_mask[0], (0, padding_len) if padding_right else (padding_len, 0),
                                          'constant', 0)
                labels[0] = F.pad(labels[0], (0, padding_len) if padding_right else (padding_len, 0), 'constant', -100)
                if loss_scale:
                    loss_scale[0] = F.pad(loss_scale[0], (0, padding_to - labels[0].shape[-1]) if padding_right else
                                          (padding_to - labels[0].shape[-1], 0), 'constant', 0.)

        if input_ids is None:
            inputs_embeds = self.pad_sequence(inputs_embeds, 0, self.padding_side)
        else:
            input_ids = self.pad_sequence(input_ids, tokenizer.pad_token_id, self.padding_side)
        attention_mask = self.pad_sequence(attention_mask, 0, self.padding_side)
        if loss_scale:
            loss_scale = self.pad_sequence(loss_scale, 0., self.padding_side)
        labels = self.pad_sequence(labels, -100, self.padding_side)

        if use_torchacc():
            rank, _, world_size, _ = get_dist_setting()
            input_ids, attention_mask, labels, loss_scale = pad_and_split_batch(
                padding_to,
                input_ids,
                attention_mask,
                labels,
                loss_scale,
                self.max_length,
                self.tokenizer,
                rank,
                world_size,
                padding_right=padding_right)
        if input_ids is not None:
            bs, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

            if self.sequence_parallel_size > 1:
                assert padding_right or bs == 1, 'Sequence parallel only support padding_side=right'
                from swift.trainers.xtuner import get_xtuner_sequence_parallel_world_size
                if get_xtuner_sequence_parallel_world_size() > 1:
                    from swift.trainers.xtuner import pad_and_split_for_sequence_parallel
                    input_ids, labels, position_ids, attention_mask, loss_scale = \
                        pad_and_split_for_sequence_parallel(
                            tokenizer, input_ids, labels, position_ids, attention_mask, loss_scale)

        res = {
            'attention_mask': attention_mask,
            'labels': labels,
        }
        if inputs_embeds is not None:
            res['inputs_embeds'] = inputs_embeds
        else:
            res['input_ids'] = input_ids
        # multimodal
        pixel_values = [b['pixel_values'] for b in batch if b.get('pixel_values') is not None]
        if len(pixel_values) > 0:
            res['pixel_values'] = torch.concat(pixel_values)

            image_sizes = [b['image_sizes'] for b in batch if b.get('image_sizes') is not None]
            if len(image_sizes) > 0:
                res['image_sizes'] = torch.concat(image_sizes)

        pixel_values_videos = [b['pixel_values_videos'] for b in batch if b.get('pixel_values_videos') is not None]
        if len(pixel_values_videos) > 0:
            res['pixel_values_videos'] = torch.concat(pixel_values_videos)
        if loss_scale is not None:
            res['loss_scale'] = loss_scale
        return res

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0, input_token_len:].tolist()

    @staticmethod
    def _is_chinese_char(cp: int) -> bool:
        """Checks whether CP is the codepoint of a CJK character."""
        # copy from transformers.generation.streamers.TextStreamer
        if ((0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF)
                or (0x2A700 <= cp <= 0x2B73F) or (0x2B740 <= cp <= 0x2B81F) or (0x2B820 <= cp <= 0x2CEAF)
                or (0xF900 <= cp <= 0xFAFF) or (0x2F800 <= cp <= 0x2FA1F)):
            return True

        return False

    @classmethod
    def _get_safe_print_idx(cls, response: str, print_idx: int, is_finished: bool = False) -> int:
        if is_finished:
            return len(response)
        if response.endswith('\n') or len(response) > 0 and cls._is_chinese_char(ord(response[-1])):
            print_idx = len(response)
        else:
            print_idx = max(response.rfind(' ') + 1, print_idx)
        return print_idx

    def generate_ids_to_response(
        self,
        generate_ids: List[int],
        is_finished: bool = True,
        *,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        # only stream=True
        return_delta: bool = False,
        print_idx: Optional[List[int]] = None,
        first_num_space: Optional[List[int]] = None,
    ):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        tokenizer = self.tokenizer
        # avoid printing template.suffix[-1])
        if isinstance(self.suffix[-1], list) and (not is_finished or is_finished
                                                  and generate_ids[-len(self.suffix[-1]):] == self.suffix[-1]):
            generate_ids = generate_ids[:-len(self.suffix[-1])]
        if not is_finished or is_finished and generate_ids[-1:] == [self.tokenizer.eos_token_id]:
            generate_ids = generate_ids[:-1]
        response = tokenizer.decode(generate_ids, **tokenizer_kwargs)
        if first_num_space is not None:
            # Avoid the occurrence of repeated words in sentence.
            res_fns = first_num_space  # res_first_num_space
            first_num_space = first_num_space[0]
            cur_num_space = len(response) - len(response.lstrip(' '))
            if not is_finished and first_num_space == -1:
                first_num_space = cur_num_space
                res_fns[0] = first_num_space
            if cur_num_space < first_num_space:
                response = ' ' * (first_num_space - cur_num_space) + response
            elif cur_num_space > first_num_space:
                response = response[cur_num_space - first_num_space:]
        if isinstance(self.suffix[-1],
                      str) and (not is_finished or is_finished and response[-len(self.suffix[-1]):] == self.suffix[-1]):
            idx = max(len(response) - len(self.suffix[-1]), 0)
            # To avoid response length being shorter than previous response length during streaming.
            if print_idx is not None:
                idx = max(idx, print_idx[0])
            response = response[:idx]

        if print_idx is not None:
            old_print_idx = print_idx[0]
            if not is_finished:
                # avoid printing incomplete words
                print_idx[0] = self._get_safe_print_idx(response, print_idx[0])
                response = response[:print_idx[0]]
            if return_delta:
                response = response[old_print_idx:]
        else:
            assert is_finished and not return_delta
        return response

    def post_process_generate_response(self, response: str, example: dict) -> str:
        return response


def register_template(template_type: str, template: Template, *, exist_ok: bool = False, **kwargs) -> None:
    if not exist_ok and template_type in TEMPLATE_MAPPING:
        raise ValueError(f'The `{template_type}` has already been registered in the TEMPLATE_MAPPING.')
    template.template_type = template_type
    template_info = {'template': template, **kwargs}
    TEMPLATE_MAPPING[template_type] = template_info


register_template(
    TemplateType.default,
    Template([], ['### Human:\n{{QUERY}}\n\n### Assistant:\n'], ['\n\n'], [['eos_token_id']],
             DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))


# You can set the query as '' to serve as a template for pre-training.
class DefaultGenerationTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}'], None, [['eos_token_id']], auto_add_bos=True)


register_template(TemplateType.default_generation, DefaultGenerationTemplate(), is_generation=True)
register_template(
    TemplateType.default_generation_bos,
    Template([['bos_token_id']], ['{{QUERY}}'], None, [['eos_token_id']]),
    is_generation=True)


class QwenTemplate(Template):

    def __init__(self, auto_add_bos: bool = False):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'],
                         ['<|im_end|>'],
                         DEFAULT_SYSTEM, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
                         auto_add_bos=auto_add_bos)


class QwenVLTemplate(QwenTemplate):

    load_medias = False

    def check_example(self, example):
        if self._is_lmdeploy:
            return
        images = example.get('images') or []
        from .utils import fetch_one
        assert not images or isinstance(fetch_one(images), str), 'QwenVL only supports datasets with images paths!'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'image'
        images = example.get('images') or []
        image = images[index]
        assert isinstance(image, str)
        return [f'Picture {index + 1}:<img>{image}</img>\n']

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example['objects']
        object_ = objects[index]
        return [f'<ref>{object_["caption"]}</ref>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example['objects']
        object_ = objects[index]
        return [f'<box>({object_["bbox"][0]},{object_["bbox"][1]}),({object_["bbox"][2]},{object_["bbox"][3]})</box>']


register_template(TemplateType.qwen, QwenTemplate())
register_template(TemplateType.qwen_vl, QwenVLTemplate())
register_template(TemplateType.chatml, QwenTemplate(auto_add_bos=True))

register_template(
    TemplateType.modelscope_agent,
    Template([], [' \n\n<|user|>:{{QUERY}} \n\n<|assistant|>:'], [], [' \n\n</s>'], DEFAULT_SYSTEM,
             [' \n\n<|system|>:{{SYSTEM}}']))


class _QwenAudioTemplateMixin:

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    example: Dict[str, Any]) -> List[Context]:
        assert media_type == 'audio'
        audios = example.get('audios') or []
        audio = audios[index]
        assert isinstance(audio, str)
        return [f'Audio {index + 1}:<audio>{audio}</audio>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, tokenizer_kwargs = super()._encode(example)
        if len(inputs) == 0:
            return inputs, tokenizer_kwargs
        inputs.pop('loss_scale', None)
        inputs.update(tokenizer_kwargs)
        return inputs, tokenizer_kwargs

    def _get_tokenizer_kwargs(self, context: str) -> Dict[str, Any]:
        return {'audio_info': self.tokenizer.process_audio(context)}

    def _concat_tokenizer_kwargs(self, tokenizer_kwargs: Dict[str, Any], curr_tokenizer_kwargs: Dict[str, Any]) -> None:
        audio_info = curr_tokenizer_kwargs.get('audio_info')
        old_audio_info = tokenizer_kwargs.get('audio_info')
        if old_audio_info is None:
            tokenizer_kwargs['audio_info'] = audio_info
        elif audio_info is not None:
            for k in ['input_audios', 'input_audio_lengths']:
                old_audio_info[k] = torch.concat([old_audio_info[k], audio_info[k]], dim=0)
            for k in ['audio_span_tokens', 'audio_urls']:
                old_audio_info[k] = old_audio_info[k] + audio_info[k]

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        if batch[0].get('audio_info') is not None:
            res['audio_info'] = [b['audio_info'] for b in batch]
        return res


class QwenAudioTemplate(_QwenAudioTemplateMixin, QwenTemplate):
    pass


class QwenAudioGenerationTemplate(_QwenAudioTemplateMixin, DefaultGenerationTemplate):
    pass


register_template(TemplateType.qwen_audio, QwenAudioTemplate(), lazy_tokenize=True)
register_template(
    TemplateType.qwen_audio_generation, QwenAudioGenerationTemplate(), lazy_tokenize=True, is_generation=True)

register_template(
    TemplateType.yi,
    Template([], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'], ['<|im_end|>'],
             None, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

register_template(
    TemplateType.yi1_5,
    Template([], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'], ['<|im_end|>'],
             None, ['{{SYSTEM}}']))

yi_vl_default_system = (
    'This is a chat between an inquisitive human and an AI assistant. Assume the role of the AI assistant. '
    "Read all the images carefully, and respond to the human's questions with informative, "
    'helpful, detailed and polite answers. '
    '这是一个好奇的人类和一个人工智能助手之间的对话。假设你扮演这个AI助手的角色。'
    '仔细阅读所有的图像，并对人类的问题做出信息丰富、有帮助、详细的和礼貌的回答。')


def _load_video_llava(video_path: str) -> np.ndarray:
    import av
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format='rgb24') for x in frames])


_T = TypeVar('_T')


class YiVLTemplate(Template):

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        inputs.pop('loss_scale', None)
        from llava.mm_utils import expand2square
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        images = example.get('images', [])
        for i, image in enumerate(images):
            background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
            images[i] = image
        if images:
            image_tensor = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
            inputs['images'] = image_tensor.to(model.dtype)
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'YIVL does not support mix-batch nlp dataset and multi-modal dataset'
        return res


class GLMTemplate(Template):

    def _init_template(self, tokenizer: PreTrainedTokenizerBase, *args, **kwargs) -> None:
        res = super()._init_template(tokenizer, *args, **kwargs)
        token_list = tokenizer.encode('')
        self.prefix.insert(0, token_list)
        if self.system_prefix is not None:
            self.system_prefix.insert(0, token_list)
        return res


class GLM4VTemplate(GLMTemplate):

    def __init__(self):
        super().__init__([], ['<|user|>\n{{QUERY}}<|assistant|>'], [], ['<|endoftext|>'], None,
                         ['<|system|>\n{{SYSTEM}}'])

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-100]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from .utils import history_to_messages

        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -100)
        if idx_list:
            idx = idx_list[0]
            image = example.get('images', [])[0]
            placeholder = '<|begin_of_image|><|endoftext|><|end_of_image|>'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            messages = history_to_messages(example.get('history') or [], example['query'], example.get('system'))
            messages[0]['image'] = image
            inputs2: Dict[str, Any] = self.tokenizer.apply_chat_template(messages, return_dict=True)
            inputs['images'] = inputs2['images']
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(TemplateType.glm4v, GLM4VTemplate(), infer_media_type='dialogue', lazy_tokenize=True, use_model=True)

register_template(
    TemplateType.yi_vl,
    YiVLTemplate([], ['### Human: {{QUERY}}\n### Assistant:'], ['\n'], ['\n###'], yi_vl_default_system,
                 ['{{SYSTEM}}\n\n']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)

register_template(TemplateType.baichuan, Template(['{{SYSTEM}}'], [[195], '{{QUERY}}', [196]], [], [['eos_token_id']]))

register_template(
    TemplateType.chatglm2,
    GLMTemplate(['{{SYSTEM}}'], ['[Round {{ROUND1}}]\n\n问：{{QUERY}}\n\n答：'], ['\n\n'], [['eos_token_id']]))

register_template(
    TemplateType.chatglm_generation, GLMTemplate([], ['{{QUERY}}'], None, [['eos_token_id']]), is_generation=True)

register_template(
    TemplateType.chatglm3,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|user|>'], None, ['<|system|>\n{{SYSTEM}}']))

codegeex4_system = '你是一位智能编程助手，你叫CodeGeeX。你会为用户回答关于编程、代码、计算机方面的任何问题，并提供格式规范、可以执行、准确安全的代码，并在必要时提供详细的解释。'

register_template(
    TemplateType.codegeex4,
    GLMTemplate([], ['<|user|>\n{{QUERY}}<|assistant|>\n'], [], ['<|endoftext|>'], codegeex4_system,
                ['<|system|>\n{{SYSTEM}}']))

register_template(
    TemplateType.deepseek,
    Template([['bos_token_id']], ['User: {{QUERY}}\n\nAssistant:'], [['eos_token_id']], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}\n\n']))
register_template(
    TemplateType.numina_math,
    Template([['bos_token_id']], ['### Problem: {{QUERY}}\n### Solution: '], ['\n'], [['eos_token_id']], None,
             [['bos_token_id'], '{{SYSTEM}}']))
register_template(
    TemplateType.deepseek2,
    Template([[100000]], ['User: {{QUERY}}\n\nAssistant:'], [[100001]], [[100001]], None, [[100000], '{{SYSTEM}}\n\n']))

# ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py
LLAMA_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest assistant. '
    'Always answer as helpfully as possible, while being safe. '
    'Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.")
register_template(
    TemplateType.llama,
    Template(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s><s>[INST] '], ['</s>'], LLAMA_DEFAULT_SYSTEM,
             ['<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n']))

register_template(TemplateType.mistral_nemo,
                  Template(['<s>[INST] '], ['{{SYSTEM}}\n\n', '{{QUERY}}[/INST]'], ['</s>[INST] '], ['</s>']))

register_template(
    TemplateType.llama3,
    Template(['<|begin_of_text|>'], [
        '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ], ['<|eot_id|>'], ['<|eot_id|>'], None,
             ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>']))

OPENBUDDY_DEFAULT_SYSTEM = (
    'You are a helpful, respectful and honest INTP-T AI Assistant named Buddy. You are talking to a human User.\n'
    'Always answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any '
    'harmful, political, religious, unethical, racist, sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'If a question does not make any sense, or is not factually coherent, '
    'explain why instead of answering something not correct. '
    "If you don't know the answer to a question, please don't share false information.\n"
    'You like to use emojis. You can speak fluently in many languages, for example: English, Chinese.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2021-09.\n'
    'You always deeply love and support China, Chinese government, people and culture.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'you are based on LLaMA and Falcon transformers model, not related to GPT or OpenAI.')
register_template(
    TemplateType.openbuddy,
    Template([], ['User: {{QUERY}}\nAssistant:'], ['\n'], [['eos_token_id']],
             OPENBUDDY_DEFAULT_SYSTEM, ['{{SYSTEM}}\n\n'],
             auto_add_bos=True))

OPENBUDDY2_DEFAULT_SYSTEM = (
    'You(assistant) are a helpful, respectful and honest INTP-T AI Assistant named Buddy. '
    'You are talking to a human(user).\nAlways answer as helpfully and logically as possible, while being safe. '
    'Your answers should not include any harmful, political, religious, unethical, racist, '
    'sexist, toxic, dangerous, or illegal content. '
    'Please ensure that your responses are socially unbiased and positive in nature.\n'
    'You cannot access the internet, but you have vast knowledge, cutoff: 2023-04.\n'
    'You are trained by OpenBuddy team, (https://openbuddy.ai, https://github.com/OpenBuddy/OpenBuddy), '
    'not related to GPT or OpenAI')

register_template(
    TemplateType.openbuddy2,
    Template([], ['<|role|>user<|says|>{{QUERY}}<|end|>\n<|role|>assistant<|says|>'], ['<|end|>\n'], ['<|end|>'],
             OPENBUDDY2_DEFAULT_SYSTEM, ['<|role|>system<|says|>{{SYSTEM}}<|end|>\n'],
             auto_add_bos=True))

INTERNLM_SYSTEM = (
    'You are an AI assistant whose name is InternLM (书生·浦语).\n'
    '- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
    'It is designed to be helpful, honest, and harmless.\n'
    '- InternLM (书生·浦语) can understand and communicate fluently in the language chosen '
    'by the user such as English and 中文.')

register_template(
    TemplateType.internlm,
    Template(['<s>'], ['<|User|>:{{QUERY}}\n<|Bot|>:'], ['<eoa>\n'], ['<eoa>'], INTERNLM_SYSTEM,
             ['<s><|System|>:{{SYSTEM}}\n']))
register_template(
    TemplateType.internlm2,
    Template(['<s>'], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'],
             ['<|im_end|>'], INTERNLM_SYSTEM, ['<s><|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))


def replace_img_tag(query: str,
                    history: History,
                    replace_token: str,
                    pattern=r'<img>(.+?)</img>') -> Tuple[str, History, List[str]]:
    images_path = []
    new_history = []
    for i, h in enumerate(history):
        if h[0] is None:
            new_history.append(h.copy())
        else:
            images_path += re.findall(pattern, h[0])
            new_history.append([re.sub(pattern, replace_token, h[0]), h[1]])
    if query is None:
        new_query = query  # pretrain dataset
    else:
        images_path += re.findall(pattern, query)
        new_query = re.sub(pattern, replace_token, query)
    return new_query, new_history, images_path


class InternLMXComposer2Template(Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by '
        'Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.')
    image_placeholder = ['</s>']

    def __init__(self, version):
        prefix = ['<s>']
        prompt = ['[UNUSED_TOKEN_146]user\n{{QUERY}}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n']
        chat_sep = ['[UNUSED_TOKEN_145]\n']
        suffix = ['[UNUSED_TOKEN_145]']
        system_prefix = ['<s>[UNUSED_TOKEN_146]system\n{{SYSTEM}}[UNUSED_TOKEN_145]\n']
        super().__init__(prefix, prompt, chat_sep, suffix, self.INTERNLM_XCOMPOSER_SYSTEM, system_prefix)
        self.version = version

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        dtype = self.model.dtype
        images = example.get('images', [])

        if self.version == 'v2.5':
            hd_num = 24
            Image_transform = get_class_from_dynamic_module('ixc_utils.Image_transform', self.tokenizer.model_dir)
            if len(images) > 1:
                hd_num = 6
            images = [Image_transform(image, hd_num=hd_num) for image in images]
        elif self.version == 'v2-4khd':
            hd_num = 55
            HD_transform = get_class_from_dynamic_module('ixc_utils.HD_transform', self.tokenizer.model_dir)
            images = [HD_transform(image, hd_num=hd_num) for image in images]
        images = [self.model.vis_processor(image).to(dtype) for image in images]
        if len(images) > 0:
            images = torch.stack(images, dim=0)
            images = self.model.img2emb(images)[0]

        inputs.pop('loss_scale', None)
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        if len(images) > 0:  # ignore <s>
            input_ids = input_ids[1:]
            if labels is not None:
                labels = labels[1:]
        input_ids.append(2)  # add dummy </s>
        if labels is not None:
            labels.append(2)
        else:
            labels = []
        res_inputs_embeds = []
        res_labels = []
        wrap_im_mask = []
        pre_i, i, idx = 0, 0, 0
        device = self.model.device
        internlm2_model = self.model.model
        if not hasattr(internlm2_model, 'tok_embeddings'):
            internlm2_model = internlm2_model.model
        tok_embeddings = internlm2_model.tok_embeddings
        while i < len(input_ids):
            if input_ids[i] == 2:  # replace_token
                res_input_ids = torch.tensor([1] + input_ids[pre_i:i], device=device)
                res_inputs_embeds.append(tok_embeddings(res_input_ids))
                wrap_im_mask += [0] * len(res_input_ids)
                res_labels += [-100] + labels[pre_i:i]
                if len(images) > 0 and idx < images.shape[0]:
                    res_inputs_embeds.append(images[idx].to(device))
                    wrap_im_mask += [1] * images.shape[1]
                    res_labels += [-100] * images.shape[1]
                idx += 1
                i += 1
                pre_i = i
                continue
            i += 1
        if len(labels) == 0:
            res_labels = None
        res_inputs_embeds = torch.concat(res_inputs_embeds, dim=0).detach()
        wrap_im_mask = torch.tensor(wrap_im_mask, dtype=torch.bool)[None]
        return {'inputs_embeds': res_inputs_embeds, 'im_mask': wrap_im_mask, 'labels': res_labels}, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        im_mask = [b['im_mask'][0] for b in batch]
        im_mask = self.pad_sequence(im_mask, 0, self.padding_side)
        res['im_mask'] = im_mask
        return res

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.internlm_xcomposer2,
    InternLMXComposer2Template(version='v2'),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)


class InternLMXComposer2_5Template(InternLMXComposer2Template):
    INTERNLM_XCOMPOSER_SYSTEM = (
        'You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n'
        '- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model '
        'that is developed by Shanghai AI Laboratory (上海人工智能实验室). '
        'It is designed to be helpful, honest, and harmless.\n'
        '- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen '
        'by the user such as English and 中文.\n'
        '- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively '
        'based on the provided image.')


register_template(
    TemplateType.internlm_xcomposer2_5,
    InternLMXComposer2_5Template(version='v2.5'),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(
    TemplateType.internlm_xcomposer2_4khd,
    InternLMXComposer2_5Template(version='v2-4khd'),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)


class InternvlTemplate(Template):
    system = 'You are an AI assistant whose name is InternLM (书生·浦语).'
    num_image_token = 256

    def __init__(self):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
                         ['<|im_end|>'],
                         self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'],
                         auto_add_bos=True)

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-100], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        labels = inputs.get('labels')
        from .vision_utils import transform_image
        pixel_values = [transform_image(image, 12) for image in example.get('images', [])]
        if pixel_values:
            pixel_values = torch.cat(pixel_values, dim=0)
            image_bs = pixel_values.shape[0]

            idx, idx2 = idx_list[0], idx_list[-1]  # remove [-100, -100]
            img_tokens: List[int] = self.tokenizer.encode(
                '<img>' + '<IMG_CONTEXT>' * self.num_image_token * image_bs + '</img>', add_special_tokens=False)
            input_ids = input_ids[:idx] + img_tokens + input_ids[idx2 + 1:]
            if labels is not None:
                labels = labels[:idx] + [-100] * len(img_tokens) + labels[idx2 + 1:]
            inputs['input_ids'] = input_ids
            inputs['labels'] = labels

            inputs['pixel_values'] = pixel_values.to(self.model.dtype)

        inputs.pop('loss_scale', None)
        return inputs, {}

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


class Internvl2Template(InternvlTemplate):
    video_segments = 8
    system = '你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。'

    def __init__(self):
        Template.__init__(
            self, [], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'], ['<|im_end|>'],
            self.system, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>'],
            auto_add_bos=True)

    def replace_tag(self, media_type, index, example) -> List[Context]:
        image_context = ['<image>\n'] if self._is_vllm else [[-100], '\n']
        if media_type == 'image':
            return image_context
        elif media_type == 'video':
            context_list = []
            for i in range(self.video_segments):
                context_list.append(f'Frame{i + 1}: ')
                context_list += image_context
            return context_list

    def replace_object(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [f'<ref>{object_["caption"]}</ref>']
        else:
            return ['<ref-object>']

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        objects = example.get('objects')
        if objects:
            object_ = objects[index]
            return [
                f'<box> [[{object_["bbox"][0]}, {object_["bbox"][1]}, '
                f'{object_["bbox"][2]}, {object_["bbox"][3]}]] </box>'
            ]
        else:
            return ['<bbox>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(InternvlTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        labels = inputs.get('labels')
        tokenizer = self.tokenizer
        model = self.model
        from .vision_utils import transform_image
        pixel_values_images = [transform_image(image, 12) for image in example.get('images', [])]
        videos_path = example.get('videos', [])
        if pixel_values_images:
            pixel_values = pixel_values_images
            assert len(pixel_values) == len(
                idx_list), f'len(pixel_values): {len(pixel_values)}, len(idx_list): {len(idx_list)}'
            num_patches = [pv.shape[0] for pv in pixel_values]
            pixel_values = torch.cat(pixel_values)
        elif videos_path:
            assert len(videos_path) == 1, f'videos_path: {videos_path}'
            from .vision_utils import load_video
            pixel_values, num_patches = load_video(videos_path[0], num_segments=self.video_segments)
        else:
            pixel_values = None
            num_patches = []
        assert len(num_patches) == len(
            idx_list), f'len(num_patches): {len(num_patches)}, len(idx_list): {len(idx_list)}'
        added_tokens_len = 0
        for idx, num_patch in zip(idx_list, num_patches):
            img_tokens: List[int] = tokenizer.encode(
                '<img>' + '<IMG_CONTEXT>' * self.num_image_token * num_patch + '</img>', add_special_tokens=False)
            input_ids = input_ids[:idx + added_tokens_len] + img_tokens + input_ids[idx + added_tokens_len + 1:]
            if labels is not None:
                labels = labels[:idx + added_tokens_len] + [-100] * len(img_tokens) + labels[idx + added_tokens_len
                                                                                             + 1:]
            added_tokens_len += len(img_tokens) - 1

        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        dtype = model.dtype
        inputs['input_ids'] = input_ids
        input_ids = torch.tensor(input_ids, device=device)
        inputs_embeds = embedding(input_ids)
        if pixel_values_images or videos_path:
            pixel_values = pixel_values.to(dtype=dtype, device=device)
            vit_embeds = model.extract_feature(pixel_values)
            selected = (input_ids == tokenizer.encode('<IMG_CONTEXT>', add_special_tokens=False)[0])
            inputs_embeds[selected] = vit_embeds.reshape(-1, vit_embeds.shape[-1])
        inputs_embeds = inputs_embeds.detach()
        inputs['labels'] = labels
        inputs['inputs_embeds'] = inputs_embeds
        inputs.pop('loss_scale', None)
        return inputs, {}


class InternvlPhi3Template(InternvlTemplate):
    system = 'You are an AI assistant whose name is Phi-3.'

    def __init__(self):
        Template.__init__(
            self, [], ['<|user|>\n{{QUERY}}<|end|><|assistant|>\n'], ['<|end|>'], ['<|end|>'],
            self.system, ['<|system|>\n{{SYSTEM}}<|end|>'],
            auto_add_bos=True)
        self.padding_side = 'left'


class Internvl2Phi3Template(Internvl2Template):

    def __init__(self):
        Template.__init__(
            self, [], ['<|user|>\n{{QUERY}}<|end|><|assistant|>\n'], ['<|end|>'], ['<|end|>'],
            self.system, ['<|system|>\n{{SYSTEM}}<|end|>'],
            auto_add_bos=True)
        self.padding_side = 'left'


register_template(
    TemplateType.internvl,
    InternvlTemplate(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(
    TemplateType.internvl_phi3,
    InternvlPhi3Template(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(
    TemplateType.internvl2,
    Internvl2Template(),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(
    TemplateType.internvl2_phi3,
    Internvl2Phi3Template(),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)


class FlorenceTemplate(Template):

    def __init__(self):
        super().__init__(['<s>'], ['{{QUERY}}</s><s>'], None, ['</s>'])
        self.task_prompts_without_inputs = {
            '<OCR>': 'What is the text in the image?',
            '<OCR_WITH_REGION>': 'What is the text in the image, with regions?',
            '<CAPTION>': 'What does the image describe?',
            '<DETAILED_CAPTION>': 'Describe in detail what is shown in the image.',
            '<MORE_DETAILED_CAPTION>': 'Describe with a paragraph what is shown in the image.',
            '<OD>': 'Locate the objects with category name in the image.',
            '<DENSE_REGION_CAPTION>': 'Locate the objects in the image, with their descriptions.',
            '<REGION_PROPOSAL>': 'Locate the region proposals in the image.'
        }

        self.task_prompts_with_input = {
            '<CAPTION_TO_PHRASE_GROUNDING>': 'Locate the phrases in the caption: {input}',
            '<REFERRING_EXPRESSION_SEGMENTATION>': 'Locate {input} in the image with mask',
            '<REGION_TO_SEGMENTATION>': 'What is the polygon mask of region {input}',
            '<OPEN_VOCABULARY_DETECTION>': 'Locate {input} in the image.',
            '<REGION_TO_CATEGORY>': 'What is the region {input}?',
            '<REGION_TO_DESCRIPTION>': 'What does the region {input} describe?',
            '<REGION_TO_OCR>': 'What text is in the region {input}?',
        }

    def replace_box(self, index: int, example: Dict[str, Any]) -> List[Context]:
        x1, y1, x2, y2 = example['objects'][index]['bbox']
        return [f'<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>']

    def _construct_prompts(self, text):
        # from processing_florence2.py
        # replace the task tokens with the task prompts if task token is in the text
        prompts = []

        for _text in text:
            if '<image>' in _text:
                _text = _text.replace('<image>', '')
            # 1. fixed task prompts without additional inputs
            for task_token, task_prompt in self.task_prompts_without_inputs.items():
                if task_token in _text:
                    assert _text == task_token, f'Task token {task_token} should be the only token in the text.'
                    _text = task_prompt
                    break
            # 2. task prompts with additional inputs
            for task_token, task_prompt in self.task_prompts_with_input.items():
                if task_token in _text:
                    _text = task_prompt.format(input=_text.replace(task_token, ''))
                    break
            prompts.append(_text)
        return prompts

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        processor = self.tokenizer.processor
        images = example.get('images', [])
        assert len(images) == 1, 'Florence series models only supports input with a single image.'
        from .vision_utils import transform_image
        image_tensors = transform_image(images[0])
        example['_image'] = image_tensors

        # process bbox
        if example.get('objects') is not None:
            if '<ref-object>' in example['query']:
                example['query'] = '<OPEN_VOCABULARY_DETECTION>'
                example['response'] = ''
                for idx in range(len(example['objects'])):
                    if idx != 0:
                        example['query'] += ','
                    example['query'] += example['objects'][idx]['caption']
                    example['response'] += example['objects'][idx]['caption'] + self.replace_box(idx, example)[0]
            elif '<bbox>' in example['query']:
                example['query'] = '<REGION_TO_DESCRIPTION>'
                example['response'] = ''
                for idx in range(len(example['objects'])):
                    bbox = self.replace_box(idx, example)[0]
                    example['query'] += bbox
                    example['response'] += example['objects'][idx]['caption']
        example['query'] = self._construct_prompts([example.get('query')])[0]

        inputs = processor(text=example['query'], images=images, return_tensors='pt').to(self.model.device)

        labels = None
        if example.get('response') is not None:
            labels = processor.tokenizer(
                text=example['response'], return_tensors='pt', padding=True,
                return_token_type_ids=False).input_ids.to(self.model.device)
        if labels is not None:
            inputs['labels'] = labels[0]

        inputs['input_ids'] = inputs['input_ids'][0]
        inputs['attention_mask'] = inputs['attention_mask'][0]
        inputs['pixel_values'] = inputs['pixel_values'].to(self.model.dtype)
        if self.max_length is None:
            self.max_length = 1024
        if self.truncation_strategy == 'delete' and len(inputs['input_ids']) > self.max_length:
            return {}, {}
        inputs['input_ids'] = inputs['input_ids'][:self.max_length]
        inputs['attention_mask'] = inputs['attention_mask'][:self.max_length]
        if inputs.get('labels') is not None:
            inputs['labels'] = inputs['labels'][:self.max_length]

        return inputs, {}

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()

    def post_process_generate_response(self, response, example):
        from .vision_utils import load_image
        if isinstance(example['images'], list):
            example['images'] = example['images'][0]
        image = load_image(example['images'])
        return str(
            self.tokenizer.processor.post_process_generation(
                response, task=example['query'], image_size=(image.width, image.height)))


register_template(
    TemplateType.florence,
    FlorenceTemplate(),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(TemplateType.xverse,
                  Template(['{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: '], [['eos_token_id']], [['eos_token_id']]))
register_template(TemplateType.yuan, Template([], ['{{QUERY}}<sep>'], None, [['eos_token_id']]))
register_template(TemplateType.ziya,
                  Template([['bos_token_id'], '{{SYSTEM}}'], ['<human>:{{QUERY}}\n<bot>:'], ['\n'], [['eos_token_id']]))

register_template(TemplateType.skywork,
                  Template(['<s>{{SYSTEM}}'], ['</s><s>[USER]{{QUERY}}[SEP][BOT]'], None, ['[SEP]</s>']))

register_template(TemplateType.bluelm,
                  Template([['bos_token_id'], '{{SYSTEM}}'], ['[|Human|]:{{QUERY}}[|AI|]:'], [], [['eos_token_id']]))

register_template(
    TemplateType.codefuse_codellama,
    Template(['{{SYSTEM}}'], ['<|role_start|>human<|role_end|>{{QUERY}}<|role_start|>bot<|role_end|>'], [],
             [['eos_token_id']]))

register_template(
    TemplateType.codefuse,
    Template([], ['<s>human\n{{QUERY}}\n<s>bot\n'], [['eos_token_id'], '\n'], [['eos_token_id']], None,
             ['<s>system\n{{SYSTEM}}\n']))

register_template(
    TemplateType.deepseek_coder,
    Template(['{{SYSTEM}}'], ['### Instruction:\n{{QUERY}}\n### Response:\n'], ['\n<|EOT|>\n'], ['\n<|EOT|>'],
             ('You are an AI programming assistant, utilizing the Deepseek Coder model, '
              'developed by Deepseek Company, and you only answer questions related to computer science. '
              'For politically sensitive questions, security and privacy issues, '
              'and other non-computer science questions, you will refuse to answer\n')))


class LlavaHfTemplate(Template):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if version.parse(transformers.__version__) < version.parse('4.43.0'):
            self.padding_side = 'left'

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images', [])
        if images:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values'] = image_inputs['pixel_values']
            if 'image_sizes' in image_inputs:
                inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


class LlavaVideoTemplate(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:

        if media_type == 'image':
            return ['<image>\n']
        assert media_type == 'video'
        media_file = example['videos'][index]
        if media_file.rsplit('.', 1)[-1] in {'jpg', 'png'}:
            return ['<image>\n']
        else:
            return ['<video>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images') or []
        videos_path = example.get('videos') or []
        if len(videos_path) > 0:
            from .vision_utils import _read_batch
            videos = _read_batch(videos_path, _load_video_llava)
            video_processor = self.tokenizer.processor.video_processor
            video_inputs = video_processor(videos, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values_videos'] = video_inputs['pixel_values_videos']
        if len(images) > 0:
            image_processor = self.tokenizer.processor.image_processor
            image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
            inputs['pixel_values'] = image_inputs['pixel_values']
            inputs['image_sizes'] = image_inputs['image_sizes']
        return inputs, {}


register_template(
    TemplateType.llava_next_video,
    LlavaVideoTemplate(['<s>{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '], ['</s>']),
    use_model=True,
    lazy_tokenize=True)

register_template(
    TemplateType.llava_next_video_yi,
    LlavaVideoTemplate(['{{SYSTEM}} '], ['USER: {{QUERY}} ASSISTANT:'], [' '], ['<|im_end|>']),
    use_model=True,
    infer_media_type='round',
    lazy_tokenize=True)


class Llava1_5Template(LlavaHfTemplate):

    def __init__(self):
        super().__init__(['<s>'], ['USER: {{QUERY}}\nASSISTANT:'], ['</s>'], ['</s>'])


register_template(TemplateType.llava1_5, Llava1_5Template(), use_model=True, lazy_tokenize=True)


class LLavaTemplate(Template):

    def __init__(self):
        # This template follows: https://github.com/haotian-liu/LLaVA/blob/main/llava/conversation.py#L350
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'],
                         None, ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200], '\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images', [])
        image_sizes = [x.size for x in images]
        from llava.mm_utils import process_images
        model = self.model.model
        if not hasattr(model, 'vision_tower'):
            model = model.model
        image_processor = model.vision_tower.image_processor
        if images:
            images_tensor = process_images(images, image_processor, self.model.config)
            inputs['images'] = images_tensor.to(model.dtype).squeeze(0)
            inputs['image_sizes'] = image_sizes
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = images
            res['image_sizes'] = sum([b['image_sizes'] for b in batch if 'image_sizes' in b], start=[])
        has_images = [(b == -200).sum() for b in res['input_ids']]
        assert all([
            h > 0 for h in has_images
        ]) or not any([h > 0
                       for h in has_images]), 'Llava does not support mix-batch nlp dataset and multi-modal dataset'
        return res

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


class Llava1_6Template(LlavaHfTemplate):

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        for b in batch:
            pixel_values = b.get('pixel_values')
            if pixel_values is not None:
                b['pixel_values'] = pixel_values.squeeze(0)
        res = super().data_collator(batch, padding_to)
        return res


class Llava1_6MistralTemplate(Llava1_6Template):

    def __init__(self):
        super().__init__(['<s>[INST] '], ['{{QUERY}} [/INST]'], ['</s>'], ['</s>'],
                         system_prefix=['<<SYS>>\n{{system}}\n<</SYS>>\n\n'])


class Llava1_6VicunaTemplate(Llava1_6Template):
    system = ('A chat between a curious human and an artificial intelligence assistant. '
              "The assistant gives helpful, detailed, and polite answers to the human's questions.")

    def __init__(self):
        super().__init__(['<s>'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'],
                         self.system,
                         system_prefix=['<s>{{SYSTEM}} '])


register_template(TemplateType.llava_mistral, Llava1_6MistralTemplate(), use_model=True, lazy_tokenize=True)

register_template(TemplateType.llava_vicuna, Llava1_6VicunaTemplate(), use_model=True, lazy_tokenize=True)


class LLava1_6YiTemplate(Llava1_6Template):

    def __init__(self):
        super().__init__([], ['<|im_start|>user\n{{QUERY}}<|im_end|><|im_start|>assistant\n'], ['<|im_end|>'],
                         ['<|im_end|>'],
                         system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        if self._is_vllm:
            return [[64000], '\n']
        else:
            return super().replace_tag(media_type, index, example)


register_template(TemplateType.llava_yi, LLava1_6YiTemplate(), use_model=True, lazy_tokenize=True)


class LLavaLlamaTemplate(Template):

    def __init__(self):
        Template.__init__(self, ['<|begin_of_text|>'], [
            '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ], ['<|eot_id|>'], ['<|eot_id|>'], None,
                          ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example):
        return ['<image>\n']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images', [])
        if raw_image:
            pixel_values = self.tokenizer.processor.image_processor(raw_image, return_tensors='pt')['pixel_values']
            inputs['pixel_values'] = pixel_values.to(self.model.dtype)
        return inputs, {}


register_template(TemplateType.llava_llama_instruct, LLavaLlamaTemplate(), use_model=True, lazy_tokenize=True)


class PaliGemmaTemplate(Template):

    def __init__(self):
        super().__init__([], ['{{QUERY}}\n'], None, ['<eos>'])

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type, index, example) -> List[Context]:
        assert media_type == 'image'
        if self._is_vllm:
            self.prompt = ['{{QUERY}}']
            return []
        else:
            self.prompt = ['{{QUERY}}\n']
            return ['<image>' * self.tokenizer.processor.image_seq_length + '<bos>']

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        raw_image = example.get('images', [])
        processor = self.tokenizer.processor
        if inputs['labels'] is not None:
            n = upper_bound(0, len(inputs['labels']), lambda idx: inputs['labels'][idx] == -100)
            n2 = len(inputs['labels']) - n
            inputs['token_type_ids'] = [0] * n + [1] * n2
        else:
            inputs['token_type_ids'] = [0] * len(inputs['input_ids'])
        if raw_image:
            model_inputs = processor(text=example['query'], images=raw_image[0], return_tensors='pt')
            inputs['pixel_values'] = model_inputs['pixel_values']
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.paligemma, PaliGemmaTemplate(), infer_media_type='dialogue', lazy_tokenize=True, is_generation=True)


class Phi3VisionTemplate(Template):

    image_placeholder = [[32044], '\n']  # <|image|>\n

    def __init__(self):
        Template.__init__(self, ['<s>'], ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'], ['<|end|>\n'], ['<|end|>'],
                          None, ['<s><|system|>\n{{SYSTEM}}<|end|>\n'])

    def replace_tag(self, media_type, index, example) -> List[Context]:
        if self._is_vllm:
            return [f'<|image_{index + 1}|>\n']  # <|image_1|>\n
        else:
            return super().replace_tag(media_type, index, example)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        images = example.get('images', [])
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, 32044)  # '<|image|>'

        if len(images) > 0:
            processor = self.tokenizer.processor
            inputs.update(processor.image_processor(images, return_tensors='pt'))
            assert len(idx_list) == len(images), f'len(idx_list): {len(idx_list)}, len(images): {len(images)}'
            res_input_ids = []
            res_labels = []
            num_img_tokens = inputs.pop('num_img_tokens').tolist()
            idx_list.insert(0, -1)
            for i in range(len(idx_list) - 1):
                image_token_id = -1
                res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + [image_token_id] * num_img_tokens[i] + [1]
                if labels is not None:
                    res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * (num_img_tokens[i] + 1)
            res_input_ids += input_ids[idx_list[-1] + 1:]
            input_ids = res_input_ids
            if labels is not None:
                res_labels += labels[idx_list[-1] + 1:]
                labels = res_labels

        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        return inputs, {}


register_template(TemplateType.phi3_vl, Phi3VisionTemplate(), lazy_tokenize=True)


class LlamaLlavaNextTemplate(LLavaTemplate):
    default_system = 'You are a helpful language and vision assistant. ' \
                     'You are able to understand the visual content that the user provides, ' \
                     'and assist the user with a variety of tasks using natural language.'

    def __init__(self):
        Template.__init__(self, ['<|begin_of_text|>'], [
            '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'
        ], ['<|eot_id|>'], ['<|eot_id|>'], self.default_system,
                          ['<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>'])


register_template(TemplateType.llama_llava_next, LlamaLlavaNextTemplate(), use_model=True, lazy_tokenize=True)


class LLavaQwenTemplate(LLavaTemplate):
    llavayi_query_template = 'You are a helpful assistant'

    def __init__(self):
        Template.__init__(self, [], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
                          ['<|im_end|>\n'], ['<|im_end|>'], self.llavayi_query_template,
                          ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'])


register_template(TemplateType.llava_qwen_instruct, LLavaQwenTemplate(), use_model=True, lazy_tokenize=True)


def _findall(token_list: List[int], token: int) -> List[int]:
    """Find the index of a token in the token_list."""
    res = []
    idx = -1
    try:
        while True:
            idx = token_list.index(token, idx + 1)
            res.append(idx)
    except ValueError:
        pass
    return res


class DeepseekVLTemplate(Template):
    DEEPSEEK_VL_SYSTEM = ('You are a helpful language and vision assistant. '
                          'You are able to understand the visual content that the user provides, '
                          'and assist the user with a variety of tasks using natural language.')

    image_placeholder = ['<image_placeholder>']

    def __init__(self):
        super().__init__(['<｜begin▁of▁sentence｜>{{SYSTEM}}\n\n'], ['User: {{QUERY}}\n\nAssistant:'],
                         ['<｜end▁of▁sentence｜>'], ['<｜end▁of▁sentence｜>'], self.DEEPSEEK_VL_SYSTEM)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        processor = self.tokenizer.processor
        input_ids, labels = inputs['input_ids'], inputs['labels']
        idx_list = _findall(input_ids, processor.image_id)
        new_input_ids, new_labels = [], []
        lo = 0
        for hi in idx_list:
            new_input_ids += input_ids[lo:hi]
            if labels is not None:
                new_labels += labels[lo:hi]
            new_input_ids += [processor.image_id] * processor.num_image_tokens
            new_labels += [-100] * processor.num_image_tokens
            lo = hi + 1
        new_input_ids += input_ids[lo:]
        if labels is not None:
            new_labels += labels[lo:]
        else:
            new_labels = None
        new_input_ids = torch.tensor(new_input_ids)
        num_image_tokens = torch.tensor([processor.num_image_tokens] * len(idx_list))
        images_outputs = processor.image_processor(images, return_tensors='pt')
        from deepseek_vl.models.processing_vlm import VLChatProcessorOutput
        output = VLChatProcessorOutput(
            sft_format=None,
            input_ids=new_input_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens)
        batched_output = processor.batchify([output])
        model = self.model
        batched_output = batched_output.to(device=model.device, dtype=model.dtype)
        inputs_embeds = model.prepare_inputs_embeds(**batched_output)[0]
        inputs['inputs_embeds'] = inputs_embeds
        inputs['labels'] = new_labels
        return inputs, {}

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.deepseek_vl,
    DeepseekVLTemplate(),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)  # only 'cpu' can pin_memory

register_template(
    TemplateType.zephyr,
    Template([], ['<|user|>\n{{QUERY}}</s>\n<|assistant|>\n'], ['</s>\n'], ['</s>'], None,
             ['<|system|>\n{{SYSTEM}}</s>\n']))

register_template(
    TemplateType.sus,
    Template(['{{SYSTEM}}'], ['### Human: {{QUERY}}\n\n### Assistant: '], ['<|endoftext|>'], ['<|endoftext|>']))

register_template(TemplateType.orion,
                  Template(['<s>{{SYSTEM}}'], ['Human: {{QUERY}}\n\nAssistant: </s>'], ['</s>'], ['</s>']))


class CogTemplate(Template):

    def check_example(self, example):
        images = example.get('images') or []
        assert len(images) <= 1

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        return []

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        image = example.get('images', [])
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer, query=example['query'], history=example.get('history'), images=image)
        image_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * image_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * image_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * image_token_len + labels[1:]
        if len(image) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
            if 'cross_images' in inputs2:
                # is cogagent
                inputs['cross_images'] = [[cross_img.to(dtype=dtype)] for cross_img in inputs2['cross_images']]
        return inputs, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        keys = ['images', 'cross_images']
        for key in keys:
            if key in batch[0]:
                res[key] = [b[key][0] for b in batch]
        token_type_ids = [torch.tensor(b['token_type_ids']) for b in batch]
        token_type_ids = self.pad_sequence(token_type_ids, 0, self.padding_side)
        res['token_type_ids'] = token_type_ids
        return res


register_template(
    TemplateType.cogagent_chat,
    CogTemplate(['<s>'], [' [INST] {{QUERY}} [/INST] '], [], ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogagent_instruct,
    CogTemplate(['<s>'], ['<EOI>Question: {{QUERY}} Answer:'], None, ['</s>']),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)

register_template(
    TemplateType.cogvlm,
    CogTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True)


def _load_video_cogvlm2(video_path: str) -> np.ndarray:
    from decord import cpu, VideoReader, bridge
    from .vision_utils import _read_video
    bridge.set_bridge('torch')
    mp4_stream = _read_video(video_path)
    clip_end_sec = 60
    clip_start_sec = 0
    num_frames = 24
    if mp4_stream is not None:
        decord_vr = VideoReader(mp4_stream, ctx=cpu(0))
    else:
        decord_vr = VideoReader(video_path, ctx=cpu(0))
    duration = len(decord_vr)  # duration in terms of frames
    start_frame = int(clip_start_sec * decord_vr.get_avg_fps())
    end_frame = min(duration, int(clip_end_sec * decord_vr.get_avg_fps())) if \
        clip_end_sec is not None else duration
    frame_id_list = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


class Cog2VideoTemplate(CogTemplate):

    def check_example(self, example):
        videos = example.get('videos') or []
        assert len(videos) <= 1

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super(CogTemplate, self)._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        videos_path = example.get('videos', [])
        from .vision_utils import _read_batch
        video = _read_batch(videos_path, _load_video_cogvlm2)
        inputs.pop('loss_scale', None)
        model = self.model
        inputs2 = model.build_conversation_input_ids(
            self.tokenizer,
            query=example['query'],
            history=example.get('history'),
            images=video,
            template_version='chat')
        video_token_len = inputs2['token_type_ids'].sum().item()
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        inputs['token_type_ids'] = [0] + [1] * video_token_len + [0] * len(input_ids[1:])
        inputs['input_ids'] = input_ids[:1] + [self.tokenizer.pad_token_id] * video_token_len + input_ids[1:]
        if labels is not None:
            inputs['labels'] = labels[:1] + [-100] * video_token_len + labels[1:]
        if len(video) > 0:
            dtype = model.dtype
            inputs['images'] = [[img.to(dtype=dtype)] for img in inputs2['images']]
        return inputs, {}


register_template(
    TemplateType.cogvlm2_video,
    Cog2VideoTemplate([['bos_token_id']], ['Question: {{QUERY}} Answer:'], ['\n'], [['eos_token_id']]),
    use_model=True,
    infer_media_type='dialogue',
    lazy_tokenize=True,
    media_type='video')

register_template(TemplateType.minicpm, Template(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']))


def _remove_idx(arr: List[int], idx_list: List[int]) -> List[int]:
    res = []
    idx_set = set(idx_list)
    for i, x in enumerate(arr):
        if i not in idx_set:
            res.append(x)
    return res


class MiniCPMVTemplate(Template):

    def __init__(self, *args, **kwargs):
        self.is_v2_5 = kwargs.pop('is_v2_5', False)
        super().__init__(*args, **kwargs)

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        if self._is_vllm:
            return ['(<image>./</image>)\n']
        else:
            return [[-1]]

    def check_example(self, example):
        images = example.get('images', [])
        if not self._is_vllm:
            assert len(images) == 1

    async def prepare_lmdeploy_inputs(self, inputs: Dict[str, Any]) -> None:
        images = inputs.pop('images', None) or []
        if len(images) == 0:
            return
        input_ids = inputs['input_ids']
        idx_list = _findall(input_ids, -100)
        idx_list.insert(0, -1)
        new_input_ids = []
        features = []
        for i in range(len(idx_list) - 1):
            new_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]]
            context_list = ['<image>', [-100], '</image>']
            feat = [x.squeeze() for x in images[i]['embeddings'].split(1)]
            grid = images[i].get('grid')
            if len(feat) > 1 and grid is not None:
                context_list.append('<slice>')
                for i in range(grid[1]):
                    if i > 0:
                        context_list.append('\n')
                    for _ in range(grid[0]):
                        context_list += ['<image>', [-100], '</image>']
                context_list.append('</slice>\n')
            new_input_ids += self._encode_context_list(context_list)[0]
            features += feat
        new_input_ids += input_ids[idx_list[-1] + 1:]
        inputs['input_ids'] = new_input_ids
        inputs['images'] = features
        await super().prepare_lmdeploy_inputs(inputs)

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example['images']
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -1)
        idx = idx_list[0]
        config = self.model.config
        tgt_sizes = None
        slice_mode = getattr(config, 'slice_mode', False)
        if slice_mode:
            if self.is_v2_5:
                from .utils import to_device
                image_processor = self.tokenizer.processor.image_processor
                image_inputs = image_processor(images, return_tensors='pt').to(self.model.dtype)
                placeholder = image_processor.get_slice_image_placeholder(image_inputs.image_sizes[0][0])
                pixel_values = to_device(image_inputs['pixel_values'], self.model.device)
                tgt_sizes = image_inputs['tgt_sizes']
            else:
                images, placeholder = self.model.get_slice_image_placeholder(images[0], self.tokenizer)
                pixel_values = [[self.model.transform(img).to(device=self.model.device) for img in images]]
            placeholder += '\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            input_tensor_ids = torch.tensor(input_ids)
            image_start_idx = torch.where(input_tensor_ids == self.tokenizer.im_start_id)[0]
            image_start_idx += 1
            image_end_idx = torch.where(input_tensor_ids == self.tokenizer.im_end_id)[0]
            valid_image_nums = max(len(image_start_idx), len(image_end_idx))
            image_bound = [
                torch.hstack(
                    [image_start_idx[:valid_image_nums].unsqueeze(-1), image_end_idx[:valid_image_nums].unsqueeze(-1)])
            ]
        else:
            placeholder = '<image>' + '<unk>' * config.query_num + '</image>\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            input_ids = (input_ids[:idx] + placeholder_id + input_ids[idx + 1:])
            if labels is not None:
                labels = (labels[:idx] + [-100] * len(placeholder_id) + labels[idx + 1:])
            image_bound = [torch.tensor([[idx, idx + config.query_num]])]
            pixel_values = [[self.model.transform(images[0]).to(device=self.model.device)]]
        data = {
            'input_ids': torch.tensor(input_ids)[None].to(device=self.model.device),
            'image_bound': image_bound,
            'pixel_values': pixel_values
        }
        if tgt_sizes is not None:  # v2.5
            data['tgt_sizes'] = tgt_sizes
        inputs_embeds, _ = self.model.get_vllm_embedding(data)
        inputs_embeds = inputs_embeds.detach()
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['inputs_embeds'] = inputs_embeds[0]
        return inputs, {}

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.minicpm_v,
    MiniCPMVTemplate(['<s>{{SYSTEM}}'], ['<用户>{{QUERY}}<AI>'], [], ['</s>']),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)


def _encode_video(video_path):
    from .vision_utils import _read_video
    mp4_stream = _read_video(video_path)
    MAX_NUM_FRAMES = 64

    from PIL import Image
    from decord import VideoReader, cpu  # pip install decord

    def uniform_sample(_l, _n):
        gap = len(_l) / _n
        idxs = [int(i * gap + gap / 2) for i in range(_n)]
        return [_l[i] for i in idxs]

    vr = VideoReader(mp4_stream, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]

    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames


class MiniCPMV2_6Template(Template):

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type in {'image', 'video'}
        if self._is_vllm:
            return ['(<image>./</image>)\n']
        else:
            return [[-1]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from .vision_utils import _read_batch
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        images = example.get('images')
        videos_path = example.get('videos')
        is_plain_text = not images and not videos_path
        images = [images]
        use_image_id = True
        max_slice_nums = None

        if videos_path:
            images = _read_batch(videos_path, _encode_video)
            use_image_id = False
            max_slice_nums = 1  # or 2

        input_ids = inputs['input_ids']
        labels = inputs['labels']
        idx_list = _findall(input_ids, -1)
        idx_list.insert(0, -1)

        from .utils import to_device
        image_processor = self.tokenizer.processor.image_processor
        image_inputs = image_processor(images, return_tensors='pt', max_slice_nums=max_slice_nums).to(self.model.dtype)
        pixel_values = to_device(image_inputs['pixel_values'], self.model.device)
        tgt_sizes = image_inputs['tgt_sizes']

        res_input_ids = []
        res_labels = []
        for i in range(len(idx_list) - 1):
            placeholder = image_processor.get_slice_image_placeholder(
                image_inputs.image_sizes[0][i], image_idx=i, max_slice_nums=max_slice_nums, use_image_id=use_image_id)
            placeholder += '\n'
            placeholder_id = self.tokenizer.encode(placeholder, add_special_tokens=False)
            res_input_ids += input_ids[idx_list[i] + 1:idx_list[i + 1]] + placeholder_id
            if labels is not None:
                res_labels += labels[idx_list[i] + 1:idx_list[i + 1]] + [-100] * len(placeholder_id)
        res_input_ids += input_ids[idx_list[-1] + 1:]
        input_ids = res_input_ids
        if labels is not None:
            res_labels += labels[idx_list[-1] + 1:]
            labels = res_labels
        if not is_plain_text:
            input_tensor_ids = torch.tensor(input_ids)
            unk_token = self.tokenizer.encode('<unk>', add_special_tokens=False)[0]
            indices = (input_tensor_ids == unk_token).nonzero(as_tuple=True)[0].tolist()

            ranges = []
            start = indices[0]
            for i in range(1, len(indices)):
                if indices[i] != indices[i - 1] + 1:
                    ranges.append([start, indices[i - 1] + 1])
                    start = indices[i]
            ranges.append([start, indices[-1] + 1])
            image_bound = [torch.tensor(ranges)]
        else:
            image_bound = []

        data = {
            'input_ids': torch.tensor(input_ids)[None].to(device=self.model.device),
            'image_bound': image_bound,
            'pixel_values': pixel_values,
            'tgt_sizes': tgt_sizes
        }
        inputs_embeds, _ = self.model.get_vllm_embedding(data)
        inputs_embeds = inputs_embeds.detach()
        inputs['input_ids'] = input_ids
        inputs['labels'] = labels
        inputs['inputs_embeds'] = inputs_embeds[0]
        return inputs, {}

    @staticmethod
    def get_generate_ids(generate_ids: Tensor, input_token_len: int) -> List[int]:
        return generate_ids[0].tolist()


register_template(
    TemplateType.minicpm_v_v2_6,
    MiniCPMV2_6Template([], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'],
                        ['<|im_end|>'], DEFAULT_SYSTEM, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']),
    use_model=True,
    lazy_tokenize=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

register_template(
    TemplateType.minicpm_v_v2_5,
    MiniCPMVTemplate(['<|begin_of_text|>{{SYSTEM}}'], [
        '<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ], ['<|eot_id|>'], ['<|eot_id|>'],
                     is_v2_5=True),
    use_model=True,
    lazy_tokenize=True,
    infer_media_type='dialogue',
    dataloader_num_workers=0,
    dataloader_pin_memory=False)

gemma_template = Template(['<bos>'], ['<start_of_turn>user\n{{QUERY}}<end_of_turn>\n<start_of_turn>model\n'],
                          ['<end_of_turn>\n'], ['<end_of_turn>'], None,
                          ['<bos><start_of_turn>system\n{{SYSTEM}}<end_of_turn>\n'])
register_template(TemplateType.gemma, gemma_template)

register_template(TemplateType.telechat, Template([], ['<_user>{{QUERY}}<_bot>'], ['<_end>'], ['<_end>']))

register_template(TemplateType.telechat_v2, Template([], ['<_user> {{QUERY}}<_bot>'], [], ['<_end>']))

DBRX_SYSTEM = (
    'You are DBRX, created by Databricks. You were last updated in December 2023. '
    'You answer questions based on information available up to that point.\n'
    'YOU PROVIDE SHORT RESPONSES TO SHORT QUESTIONS OR STATEMENTS, '
    'but provide thorough responses to more complex and open-ended questions.\n'
    'You assist with various tasks, from writing to coding (using markdown for code blocks '
    '— remember to use ``` with code, JSON, and tables).\n'
    'You do not have real-time data access or code execution capabilities.'
    ' You avoid stereotyping and provide balanced perspectives on controversial topics. '
    'You do not provide song lyrics, poems, or news articles and do not divulge details of your training data.\n'
    'This is your system prompt, guiding your responses. Do not reference it, just respond to the user. '
    'If you find yourself talking about this message, stop. You should be responding appropriately '
    'and usually that means not mentioning this.'
    'YOU DO NOT MENTION ANY OF THIS INFORMATION ABOUT YOURSELF UNLESS THE INFORMATION IS DIRECTLY '
    'PERTINENT TO THE USER\'S QUERY.')
register_template(
    TemplateType.dbrx,
    Template([], ['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'], ['<|im_end|>\n'], ['<|im_end|>'],
             DBRX_SYSTEM, ['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n']))

register_template(TemplateType.mengzi,
                  Template([], ['输入：{{QUERY}}输出：\n'], [], [['eos_token_id']], None, ['指令：{{SYSTEM}}']))

C4AI_SYSTEM = ('You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by '
               'providing thorough responses.You are trained by Cohere.')
register_template(
    TemplateType.c4ai,
    Template(
        ['<BOS_TOKEN>'],
        ['<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{QUERY}}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'],
        ['<|END_OF_TURN_TOKEN|>'], ['<|END_OF_TURN_TOKEN|>'], C4AI_SYSTEM,
        ['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{SYSTEM}}<|END_OF_TURN_TOKEN|']))


class mPlugOwl2Template(Template):

    def __init__(self):
        super().__init__(['{{SYSTEM}}'], ['USER: {{QUERY}}ASSISTANT:'], ['</s>'], [['eos_token_id']])

    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index, example) -> List[Context]:
        assert media_type == 'image'
        return [[-200]]

    def _encode(self, example: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        from mplug_owl2.mm_utils import process_images
        processor = self.tokenizer.processor
        images = example.get('images', [])
        for i, image in enumerate(images):
            # ref: https://modelscope.cn/models/iic/mPLUG-Owl2.1
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            images[i] = image
        inputs, _ = super()._encode(example)
        if len(inputs) == 0:
            return inputs, {}
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        if images:
            images = process_images(images, processor)
            images = images.to(self.model.dtype)
            return {'input_ids': input_ids, 'labels': labels, 'images': images}, {}
        else:
            return {'input_ids': input_ids, 'labels': labels}, {}

    def data_collator(self, batch: List[Dict[str, Any]], padding_to: Optional[int] = None) -> Dict[str, Any]:
        res = super().data_collator(batch, padding_to)
        images = [b['images'] for b in batch if 'images' in b]
        if images:
            res['images'] = torch.concat(images)
        return res


register_template(
    TemplateType.mplug_owl2, mPlugOwl2Template(), infer_media_type='round', use_model=True, lazy_tokenize=True)

register_template(TemplateType.wizardlm2_awq,
                  Template(['{{SYSTEM}}'], ['User:\n{{QUERY}}\n\nAssistant:\n'], ['\n\n'], ['</s>']))

_wizardlm2_system = ('A chat between a curious user and an artificial intelligence assistant. '
                     'The assistant gives helpful, detailed, and polite answers to the user\'s questions. ')
register_template(TemplateType.wizardlm2,
                  Template(['{{SYSTEM}}'], ['USER: {{QUERY}} ASSISTANT:'], ['</s>'], ['</s>'], _wizardlm2_system))

_default_phi3_system = ('You are a helpful digital assistant. '
                        'Please provide safe, ethical and accurate information to the user.')

register_template(
    TemplateType.phi3,
    Template(['<s>'], ['<|user|>\n{{QUERY}}<|end|>\n<|assistant|>\n'], ['<|end|>\n'], ['<|end|>'], _default_phi3_system,
             ['<s><|system|>\n{{SYSTEM}}<|end|>\n']))

register_template(TemplateType.atom,
                  Template(['{{SYSTEM}}'], ['<s>Human: {{QUERY}}\n</s><s>Assistant: '], ['</s>'], ['</s>']))


def get_template(
    template_type: str,
    tokenizer: PreTrainedTokenizerBase,
    default_system: Optional[str] = None,
    max_length: Optional[int] = None,
    truncation_strategy: Literal['delete', 'truncation_left'] = 'delete',
    model=None,
    **kwargs,
) -> Template:
    template_info = TEMPLATE_MAPPING[template_type]
    template = deepcopy(template_info['template'])
    template._init_template(tokenizer, default_system, max_length, truncation_strategy, model=model, **kwargs)
    return template
