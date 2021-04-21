# -*- coding: utf-8 -*-

"""Get transitive relations from the Selventa BEL content.

Run with ``python -m causal_precedence_training.sources.selventa``.
"""

import logging
import os
from typing import Tuple, Union

import bioregistry
import click
import pandas as pd
import pybel
import pybel.constants as pc
import pyobo
import pystow
from more_click import verbose_option
from pybel.dsl import BaseConcept
from pyobo.xrefdb.sources.famplex import _get_famplex_df
from tqdm.autonotebook import tqdm

from causal_precedence_training.resources import HERE

logger = logging.getLogger(__name__)

module = pystow.module('causal_precedence_training', 'selventa')

fplx_df = _get_famplex_df()
bel_fplx = dict(fplx_df.loc[fplx_df['target_ns'] == 'BEL', ['target_id', 'source_id']].values)


@click.command()
@verbose_option
@click.option('--force', is_flag=True)
def main(force: bool):
    for graph_name in 'large_corpus', 'small_corpus':
        click.secho(f'{graph_name} results:', fg='blue')
        df = get_normalized_dataframe(graph_name=graph_name, force=force)
        click.echo(df.head())


def get_normalized_dataframe(graph_name: str, force: bool = False) -> pd.DataFrame:
    df = get_dataframe(graph_name=graph_name, force=force)

    for letter in 'abc':
        df[[f'{letter}.prefix', f'{letter}.identifier', f'{letter}.name']] = [
            get_identifier(namespace, name)
            for namespace, name in df[[f'{letter}.prefix', f'{letter}.name']].values
        ]
    output_path = os.path.join(HERE, f'selventa_{graph_name}.tsv')
    df.to_csv(output_path, sep='\t', index=False)
    return df


MISSING_NAMESPACE = set()
MISSING = set()


def get_identifier(namespace: str, name: str) -> Union[Tuple[str, None, str], Tuple[str, str, str]]:
    if namespace in {'SFAM', 'SCOMP'}:
        return 'fplx', bel_fplx.get(name), name
    if namespace in {'SCHEM', 'CHEBI'}:
        prefix, identifier, name = pyobo.ground('chebi', name)
        return prefix or namespace, identifier, name

    norm_namespace = bioregistry.normalize_prefix(namespace)
    if norm_namespace is None:
        raise ValueError(f'could not normalize {namespace}')
    namespace = norm_namespace

    if namespace in MISSING_NAMESPACE:
        return namespace, None, name
    try:
        name_id_mapping = pyobo.get_name_id_mapping(namespace)
    except:
        logger.info('missing namespace: %s', namespace)
        MISSING_NAMESPACE.add(namespace)
        return namespace, None, name

    if name_id_mapping:
        identifier = name_id_mapping.get(name)
        if identifier:
            return namespace, identifier, name
        elif (namespace, name) not in MISSING:
            MISSING.add((namespace, name))
            logger.debug('missing lookup for %s ! %s', namespace, name)
            return namespace, None, name
    elif namespace not in MISSING_NAMESPACE:
        logger.info('empty namespace: %s', namespace)
        MISSING_NAMESPACE.add(namespace)

    return namespace, None, name


def get_dataframe(graph_name: str, force: bool = False) -> pd.DataFrame:
    cache_path = module.join(name=f'{graph_name}.tsv')
    if cache_path.exists() and not force:
        return pd.read_csv(cache_path, sep='\t')

    graph = get_graph(graph_name=graph_name, force=force)

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


def get_graph(graph_name: str, force: bool = False) -> pybel.BELGraph:
    """Get the Selventa large corpus as a BEL Graph."""
    url = f'https://github.com/cthoyt/selventa-knowledge/raw/master/selventa_knowledge/{graph_name}.bel'
    cache_path = module.join(name=f'{graph_name}.bel.pickle')
    if cache_path.exists() and not force:
        return pybel.load(cache_path)
    path = module.ensure(url=url, force=force)
    graph = pybel.from_bel_script(path, citation_clearing=False)
    pybel.dump(graph, path.as_posix())
    return graph


if __name__ == '__main__':
    main()
