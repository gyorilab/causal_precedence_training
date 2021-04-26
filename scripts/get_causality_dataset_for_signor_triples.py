import os
import sys
import json
import shutil
import pandas as pd
from indra.statements.agent import default_ns_order

import causal_precedence_training.locations as loc
from causal_precedence_training.reach_output import \
    get_reach_causality_dataframe_for_triple, get_readings_for_reading_ids



def curie_from_db_refs(db_refs):
    """Get curie for highest priority namespace in db_refs dict 
    
    Parameters
    ----------
    db_refs : dict
        An INDRA style db_refs dict mapping namespaces to identifiers

    Returns
    -------
    curie : str
        A curie of the form f'{namespace}:{identifier}' associated to
        the db_refs entry with namespace highest in the priority list
        default_ns_order taken from `indra.statements.agent`. Chooses
        a random db_refs entry for namespaces that do not appear in the
        priority list. Returns None if given an empty db_refs dict.
    """
    for namespace in default_ns_order:
        if namespace in db_refs:
            return f'{namespace}:{db_refs[namespace]}'
    if db_refs:
        # If db_refs has no namespaces from priority list, just return a
        # random one
        ns, id_ = list(db_refs.items())[0]
        return f'{ns}:{id_}'
    return None


if __name__ == '__main__':
    all_results_path = os.path.join(loc.TRAINING_DATA_EXPORT_DIRECTORY,
                                    'signor_training_data')
    # Results for each individual triple are stored in the temp folder.
    # In case of an error, script can be restarted and the results
    # which had already been computed can be pulled from this folder
    temp_results_path = os.path.join(all_results_path, 'temp')
    if not os.path.exists(temp_results_path):
        os.makedirs(temp_results_path)
    completed = set(os.listdir(temp_results_path))
    try:
        signor_triples_df = pd.\
            read_pickle(os.path.join(loc.TRIPLES_DIRECTORY,
                                     'signor_causal_triples.pkl'))
    except FileNotFoundError:
        print('Signor Triples have not been generated. First run the script'
              ' get_signor_causal_triples.py')
        sys.exit(1)
    for index, row in signor_triples_df.iterrows():
        results_path = os.path.join(temp_results_path, f'triple_{index}')
        statement1, statement2 = row['statement1'], row['statement2']
        if f'triple_{index}' in completed:
            print('Results already computed for'
                        f' {statement1}, {statement2}')
            continue
        print(f'Working on {statement1}, {statement2}')
        agent1 = statement1.subj
        agent2 = statement1.obj
        agent3 = statement2.obj
        curie1 = curie_from_db_refs(agent1.db_refs)
        curie2 = curie_from_db_refs(agent2.db_refs)
        curie3 = curie_from_db_refs(agent3.db_refs)
        if curie1 is None:
            print(f'Ungrounded agent {agent1} with db_refs'
                           f'{ agent1.db_refs}')
        if curie2 is None:
            print(f'Ungrounded agent {agent2} with db_refs'
                           f'{ agent2.db_refs}')
        if curie3 is None:
            print(f'Ungrounded agent {agent3} with db_refs'
                           f'{ agent3.db_refs}')
        if any(curie is None for curie in (curie1, curie2, curie3)):
            # We use the existence of directory as sign that results have
            # already been computed. We need to create it even if no results
            # were found.
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            continue
        df = get_reach_causality_dataframe_for_triple(curie1, curie2, curie3)
        if df is None:
            # We use the existence of directory as sign that results have
            # already been computed. We need to create it even if no results
            # were found.
            if not os.path.exists(results_path):
                os.makedirs(results_path)
            print(f'No results found for {statement1}, {statement2}')
            continue
        print(f'Results found for {statement1}, {statement2}')
        df['agent1_name'] = agent1.name
        df['agent2_name'] = agent2.name
        df['agent3_name'] = agent3.name
        df['signor_stmt1'] = statement1.__str__()
        df['signor_stmt2'] = statement2.__str__()
        df = df[['signor_stmt1', 'signor_stmt2', 
                 'agent1_name', 'agent1', 'stmt_type1',
                 'agent2_name', 'agent2', 'stmt_type2',
                 'agent3_name', 'agent3', 'text1', 'text2',
                 'sentence_id1', 'sentence_id2', 'reading_id']]
        results_path = os.path.join(temp_results_path, f'triple_{index}')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        df.to_csv(os.path.join(results_path, 'dataset'), sep=',', index=False)

    all_dfs = []
    for directory in os.listdir(temp_results_path):
        triplet_directory = os.path.join(temp_results_path, directory)
        if os.listdir(triplet_directory):
            df = pd.read_csv(os.path.join(triplet_directory, 'dataset'), sep=',')
            all_dfs.append(df)

    results_df = pd.concat(all_dfs)
    reach_jsons = get_readings_for_reading_ids(results_df.reading_id.values)

    results_df['signor_stmt_type1'] = results_df.signor_stmt1.\
        apply(lambda x: x.split('(')[0])
    results_df['signor_stmt_type2'] = results_df.signor_stmt2.\
        apply(lambda x: x.split('(')[0])

    results_df = results_df.rename({'stmt_type1': 'database_stmt_type1',
                                    'stmt_type2': 'database_stmt_type2',
                                    'text1': 'sentence_text1',
                                    'text2': 'sentence_text2'},
                                   axis=1)

    results_df = results_df[['agent1_name', 'agent1', 'agent2_name', 'agent2',
                             'agent3_name', 'agent3', 'signor_stmt_type1',
                             'signor_stmt_type2', 'database_stmt_type1',
                             'database_stmt_type2', 'sentence_text1',
                             'sentence_text2', 'sentence_id1',
                             'sentence_id2', 'reading_id']]
    results_df.to_csv(os.path.join(all_results_path, 'signor_triples_dataset.csv'),
                      sep=',', index=False)
    with open(os.path.join(all_results_path,
                           'signor_triples_dataset_reach_output.json'), 'w') \
                           as f:
        json.dump(reach_jsons, f, indent=True)
    shutil.rmtree(temp_results_path)
