"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from enum import Enum


class BackendType(Enum):
    unknown = 0
    onnx = 1
    tf = 2


class AcceleratorType(Enum):
    unknown = 0
    cpu = 1
    gpu = 2
    tpu = 3


class IndexerType(Enum):
    exact = 0
    approx = 1


class ModalityType(Enum):
    multimodal = 0
    text = 1
    image = 2
