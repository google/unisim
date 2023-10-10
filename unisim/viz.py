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

from .dataclass import Result
from tabulate import tabulate


class Visualization():

    @staticmethod
    def result(result: Result):
        rows = []
        for m in result.matches:

            # check content
            gs = round(m.global_similarity, 2) if m.global_similarity else ""
            ps = round(m.partial_similarity, 2) if m.partial_similarity else ""
            content = m.data[:30] if m.data else ""

            # build row
            row = [
                m.idx,
                m.is_global_match,
                gs,
                m.is_partial_match,
                ps,
                content
            ]
            rows.append(row)
        if result.query:
            print(f'Query {result.query_idx}: "{result.query}"')
        else:
            print(f"Query {result.query_idx}")
        print(tabulate(rows,
                       headers=['idx',
                                'is_global', 'global_sim',
                                'is_partial', 'partial_sim',
                                'content']))
