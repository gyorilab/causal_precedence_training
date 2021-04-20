# -*- coding: utf-8 -*-

"""Get transitive relations from the Selventa BEL content."""

import click
import pandas as pd
import pybel
import pybel.constants as pc
import pystow
from more_click import verbose_option
from pybel.dsl import BaseConcept
from tqdm.autonotebook import tqdm

URL = 'https://github.com/cthoyt/selventa-knowledge/raw/master/selventa_knowledge/large_corpus.bel'

module = pystow.module('causal_precedence_training', 'selventa')


@click.command()
@verbose_option
@click.option('--force', is_flag=True)
def main(force: bool):
    df = get_dataframe(force=force)
    click.echo(df.head())


def get_dataframe(force: bool = False) -> pd.DataFrame:
    cache_path = module.join(name='large_corpus.tsv')
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, sep='\t')

    graph = get_graph(force=force)

    key_to_edge = {
        k: (u, v, d)
        for u, v, k, d in tqdm(graph.edges(keys=True, data=True))
    }

    len(graph.transitivities)

    rows = []
    for k1, k2 in tqdm(graph.transitivities):
        a, b, data = key_to_edge[k1]
        _b, c, _ = key_to_edge[k2]

        citation = data.get(pc.CITATION)
        if not citation or citation.namespace != 'pubmed':
            continue

        evidence = data.get(pc.EVIDENCE)
        if not evidence:
            continue

        if not (isinstance(a, BaseConcept) and isinstance(b, BaseConcept) and isinstance(c, BaseConcept)):
            continue

        rows.append((
            a.namespace,  # u1.identifier,
            a.name,
            b.namespace,  # v1.identifier,
            b.name,
            c.namespace,  # v2.identifier,
            c.name,
            citation.identifier,
            evidence,
        ))

    df = pd.DataFrame(rows, columns=[
        'a.prefix',  # 'a.identifier',
        'a.name',
        'b.prefix',  # 'b.identifier',
        'b.name',
        'c.prefix',  # 'c.identifier',
        'c.name',
        'pmid',
        'text',
    ])
    df.to_csv(cache_path, index=False, sep='\t')
    return df


def get_graph(force: bool = False) -> pybel.BELGraph:
    cache_path = module.join(name='large_corpus.bel.pickle')
    if cache_path.exists() and not force:
        return pybel.load(cache_path)
    path = module.ensure(url=URL, force=force)
    graph = pybel.from_bel_script(path, citation_clearing=False)
    pybel.dump(graph, path.as_posix())
    return graph


if __name__ == '__main__':
    main()
