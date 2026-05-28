# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Authors: Neuron Science Team, Amazon Annapurna Labs
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html

import ftfy
import regex as re
from transformers import AutoTokenizer

__all__ = ['HuggingfaceTokenizer']


def _clean_text(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=512, **kwargs):
        self.name = name
        self.seq_len = seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        _kwargs = {
            'return_tensors': 'pt',
            'padding': 'max_length',
            'truncation': True,
            'max_length': self.seq_len,
        }
        _kwargs.update(**kwargs)

        if isinstance(sequence, str):
            sequence = [sequence]
        sequence = [_clean_text(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        if return_mask:
            return ids.input_ids, ids.attention_mask
        return ids.input_ids
