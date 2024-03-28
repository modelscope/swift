# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from swift.utils.import_utils import _LazyModule
from .utils import *

if TYPE_CHECKING:
    # Recommend using `xxx_main`
    from .app_ui import gradio_chat_demo, gradio_generation_demo, app_ui_main
    from .deploy import deploy_main
    from .dpo import dpo_main
    from .infer import merge_lora, prepare_model_template, infer_main, merge_lora_main
    from .rome import rome_main
    from .sft import sft_main
    from .export import export_main
    from .eval import eval_main
else:
    _extra_objects = {
        k: v
        for k, v in globals().items() if not k.startswith('_')
    }
    _import_structure = {
        'app_ui':
        ['gradio_chat_demo', 'gradio_generation_demo', 'app_ui_main'],
        'deploy': ['deploy_main'],
        'dpo': ['dpo_main'],
        'infer': [
            'merge_lora', 'prepare_model_template', 'infer_main',
            'merge_lora_main'
        ],
        'rome': ['rome_main'],
        'sft': ['sft_main'],
        'export': ['export_main'],
        'eval': ['eval_main'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extra_objects,
    )
