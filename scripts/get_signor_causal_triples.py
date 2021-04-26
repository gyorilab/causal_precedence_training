import os
import pandas as pd
from copy import deepcopy
from itertools import product
from indra.sources import signor
from collections import defaultdict
from indra.statements import Activation, Inhibition

from causal_precedence_training import locations


def get_relevant_signor_statements():
    """Get Inhibition and Activation statements from SIGNOR
    
    Returns
    ----------
    signor_stmts_by_id : dict
        Dictionary mapping SIGNOR IDs to lists of associated statements
    """
    print('Collecting SIGNOR statements from the web.')
    sp = signor.process_from_web()
    signor_stmts_by_id = defaultdict(list)
    for stmt in sp.statements:
        if isinstance(stmt, Inhibition) or isinstance(stmt, Activation):
            signor_stmts_by_id[stmt.evidence[0].source_id].append(stmt)
    signor_stmts_by_id = dict(signor_stmts_by_id)
    return signor_stmts_by_id


def generate_signor_triples_dataframe(signor_stmts_by_id):
    print('Generating dataframe of SIGNOR triples.')
    SIGNOR_pathway_dfs = []
    for pathway_filename in os.listdir(locations.SIGNOR_PATHWAYS_DIRECTORY):
        filepath = os.path.join(locations.SIGNOR_PATHWAYS_DIRECTORY,
                                pathway_filename)
        # Load relevant columns of pathway tsv file
        pathway_df = pd.read_csv(filepath, sep='\t', keep_default_na=False,
                                 usecols=['SIGNOR_ID', 'ENTITYA', 'ENTITYB',
                                          'EFFECT'])
        # Filter to only activations and inhibitions
        pathway_df = pathway_df[
            pathway_df.EFFECT.isin(['up-regulates activity',
                                    'down-regulates activity'])]
        # Add column for INDRA statement associated to each edge in pathway
        pathway_df['statement'] = pathway_df.SIGNOR_ID.\
            apply(lambda x: signor_stmts_by_id[x][0])
        # Pathway TSV files are inconsistent regarding naming of pathways.
        # Some have column for the pathway name while others don't. Use the
        # associated filename ot label each pathway since it is always
        # present.
        pathway_df['pathway_filename'] = pathway_filename
        # Perform a self inner join from Object to Subject to collect causal
        # triples from within the pathway.
        pathway_df = pathway_df.\
            merge(pathway_df, left_on='ENTITYB', right_on='ENTITYA',
                  how='inner')
        # Do some renaming to clean up column names after join
        pathway_df.\
            rename({'ENTITYA_x': 'signor_entity1',
                    'ENTITYA_y': 'signor_entity2',
                    'ENTITYB_y': 'signor_entity3',
                    'statement_x': 'statement1',
                    'statement_y': 'statement2',
                    'pathway_filename_x': 'pathway_filename'}, axis=1,
                   inplace=True)
        # Drop unnecessary columns
        pathway_df.drop(['pathway_filename_y', 'ENTITYB_x',
                         'SIGNOR_ID_x', 'SIGNOR_ID_y'], axis=1, inplace=True)
        SIGNOR_pathway_dfs.append(pathway_df)
    # Concatenate the rows of the dataframes together into one large dataframe
    # Use shortname to make following filter more readable
    df = pd.concat(SIGNOR_pathway_dfs)
    # Filter self edges and loops
    df = df[(df.signor_entity1 != df.signor_entity3) &
            (df.signor_entity1 != df.signor_entity2) &
            (df.signor_entity2 != df.signor_entity3)]
    # Remove duplicate triples
    df = df.groupby(['signor_entity1', 'signor_entity2',
                     'signor_entity3'],
                    as_index=False).first()
    # Change order of columns
    df = df[['statement1', 'statement2', 'signor_entity1', 'signor_entity2',
             'signor_entity3', 'pathway_filename']]
    print('Expanding bound conditions.')
    # Expand bound conditions into multiple statements
    # Pull out agents A->B->C from chain
    df['A'] = df.statement1.apply(lambda x: x.subj)
    df['B'] = df.statement1.apply(lambda x: x.obj)
    df['C'] = df.statement2.apply(lambda x: x.obj)
    # Add columns for bound agents if they exist
    df['A_bound'] = df.A.apply(_get_bound_agent)
    df['B_bound'] = df.B.apply(_get_bound_agent)
    df['C_bound'] = df.C.apply(_get_bound_agent)
    # Drop bound conditions from agent columns
    df.loc[:, 'A'] = df.A.apply(_get_unbound_agent)
    df.loc[:, 'B'] = df.B.apply(_get_unbound_agent)
    df.loc[:, 'C'] = df.C.apply(_get_unbound_agent)

    # I can't think of a better way than just iterating over the dataframe
    new_rows = []
    for _, row in df.iterrows():
        for A, B, C in product([row['A'], row['A_bound']],
                               [row['B'], row['B_bound']],
                               [row['C'], row['C_bound']]):
            if pd.isna(A) or pd.isna(B) or pd.isna(C):
                continue
            new_stmt1 = deepcopy(row['statement1'])
            new_stmt2 = deepcopy(row['statement2'])
            new_stmt1.subj, new_stmt1.obj = A, B
            new_stmt2.subj, new_stmt2.obj = B, C
            new_rows.append([new_stmt1, new_stmt2, row['signor_entity1'],
                             row['signor_entity2'], row['signor_entity3'],
                             row['pathway_filename']])

    df = pd.DataFrame(new_rows,
                      columns=['statement1', 'statement2', 'signor_entity1',
                               'signor_entity2', 'signor_entity3',
                               'pathway_filename'])
    return df


def _get_bound_agent(agent):
    """Get first bound condition agent for an agent if one exists."""
    if agent.bound_conditions:
        return deepcopy(agent.bound_conditions[0].agent)
    else:
        return float('nan')


def _get_unbound_agent(agent):
    """Return agent with all bound conditoins removed"""
    result = deepcopy(agent)
    if result.bound_conditions:
        result.bound_conditions = []
    return result


def main():
    triples_df = generate_signor_triples_dataframe(
        get_relevant_signor_statements()
    )
    print('Writing result to file.')
    triples_df.to_pickle(os.path.join(locations.TRIPLES_DIRECTORY,
                                      'signor_causal_triples.pkl'))

if __name__ == '__main__':
    main()
