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
