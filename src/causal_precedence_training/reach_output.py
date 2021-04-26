import pandas as pd
from ast import literal_eval
from collections import defaultdict


from .query_indra_db import get_raw_statement_jsons
from .query_indra_db import get_pa_statements_for_pair
from .query_indra_db import get_readings_for_reading_ids
from .query_indra_db import get_reach_support_for_pa_statements


def get_reach_support_for_triple(curie1, curie2, curie3):
    """Get reach support for triple A -> B -> C

    Support consists of pairs of statements linking A -> B and B -> C
    respectively, where A -> B and B -> C have reach support from the
    same reading of the same paper.

    Parameters
    ----------
    curie1 : str
       String of the form f'{namespace}:{identifier}' such as
       'HGNC:6091' or 'FPLX:PI3K'.
    curie2 : str
        See above
    curie3 : str
        See above

    Returns
    -------
    dict
        dict mapping reading_ids to dicts. The inner dicts have two entries
        with respective keys 'A->B' and 'B->C'. The value associated with
        'A->B' ('B->C') is a list of tuples of the form
        (raw_stmt_id, statement_type) for statements connecting 
        A -> B (B -> C).
    """
    # These are dictionaries mapping stmt_mk_hashes to statement types
    mk_hash_dict1 = get_pa_statements_for_pair(curie1, curie2)
    mk_hash_dict2 = get_pa_statements_for_pair(curie2, curie3)
    # If no statements found for either pair, return an empty dict
    if not mk_hash_dict1 or not mk_hash_dict2:
        return {}
    reach_support_AB = get_reach_support_for_pa_statements(mk_hash_dict1.keys())
    reach_support_BC = get_reach_support_for_pa_statements(mk_hash_dict2.keys())
    # Convert into dicts mapping reading_ids to tuples of raw statement ids
    # and statement types for A->B link and B->C link respectively
    reading_dict_AB = defaultdict(list)
    reading_dict_BC = defaultdict(list)
    for stmt_mk_hash, raw_stmt_id, reading_id in reach_support_AB:
        reading_dict_AB[reading_id].append((raw_stmt_id,
                                            mk_hash_dict1[stmt_mk_hash]))
    for stmt_mk_hash, raw_stmt_id, reading_id in reach_support_BC:
        reading_dict_BC[reading_id].append((raw_stmt_id,
                                            mk_hash_dict2[stmt_mk_hash]))
    # Keep only cases with A->B and B->C link in same REACH reading of same
    # paper.
    keep = reading_dict_AB.keys() & reading_dict_BC.keys()
    reading_dict_AB = {reading_id: stmts for reading_id,
                       stmts in reading_dict_AB.items()
                       if reading_id in keep}
    reading_dict_BC = {reading_id: stmts for reading_id,
                       stmts in reading_dict_BC.items()
                       if reading_id in keep}
    # Return in desired output format
    return {reading_id: {'A->B': reading_dict_AB[reading_id],
                         'B->C': reading_dict_BC[reading_id]}
            for reading_id in keep}


def match_up_stmts_to_sentence_positions(stmts_with_json, reach_json):
    """Match up raw statements to positions in reach sentence metadata

    Parameters
    ----------
    stmts_with_json : list
        A list of tuples of the form (raw_stmt_id, stmt_json) where
        the statement json is in dict form.
    reach_json : dict
        A reach output json in dict form

    Returns
    -------
    dict
        A dictionary mapping raw statement ids to nested tuples of the
        form (sentence_id, (start_pos, end_pos)) where sentence_id is the
        id of the evidence sentence within the reach json and start_pos and
        end_pos are the coordinates for the evidence sentence within article
        as given in the sentence metadata in the reach_json.
    """
    text_to_sentence_ids = {frame['verbose-text']: frame['sentence'] for
                            frame in reach_json['events']['frames']
                            if 'verbose-text' in frame}
    sentence_positions = {frame['frame-id']: (frame['start-pos']['offset'],
                                              frame['end-pos']['offset'])
                          for frame in reach_json['sentences']['frames']
                          if 'start-pos' in frame}
    stmts_to_sentence_positions = {}
    for stmt_id, stmt_json in stmts_with_json.items():
        evidence_text = stmt_json['evidence'][0]['text']
        sentence_id = text_to_sentence_ids[evidence_text]
        stmts_to_sentence_positions[stmt_id] = \
            (sentence_id, sentence_positions[sentence_id])
    return stmts_to_sentence_positions


def get_reach_causality_dataframe_for_triple(curie1, curie2, curie3,
                                             neighbor_cutoff=20):
    """Returns DataFrame of training examples for triple

    Parameters
    ----------
    curie1 : str
       String of the form f'{namespace}:{identifier}' such as
       'HGNC:6091' or 'FPLX:PI3K'.
    curie2 : str
        See above
    curie3 : str
        See above

    neighbor_cutoff : Optional[int]
        Sentence X is the considered a neighboring predecessor of sentence Y
        if the end position in the sentence coordinate of X is within this
        neighbor_cutoff of the start position in the sentence coordinate of
        Y.

    Returns
    -------
    pandas.DataFrame
    """
    rows = []
    reading_stmts_dict = get_reach_support_for_triple(curie1,
                                                      curie2,
                                                      curie3)
    if not reading_stmts_dict:
        return None
    reach_jsons = get_readings_for_reading_ids(reading_stmts_dict.keys())
    for reading_id, stmts in reading_stmts_dict.items():
        reach_json = reach_jsons[reading_id]
        AB_stmts_type_dict = {raw_stmt_id: type_ for raw_stmt_id, type_
                              in stmts['A->B']}
        BC_stmts_type_dict = {raw_stmt_id: type_ for raw_stmt_id, type_
                              in stmts['B->C']}
        AB_stmt_jsons = get_raw_statement_jsons(AB_stmts_type_dict.keys())
        BC_stmt_jsons = get_raw_statement_jsons(BC_stmts_type_dict.keys())
        AB = match_up_stmts_to_sentence_positions(AB_stmt_jsons, reach_json)
        BC = match_up_stmts_to_sentence_positions(BC_stmt_jsons, reach_json)
        for stmt_id1, (sentence_id1, (start_pos1, end_pos1)) in AB.items():
            for stmt_id2, (sentence_id2, (start_pos2, end_pos2)) in BC.items():
                if start_pos1 == start_pos2 or \
                   (end_pos1 > start_pos2 - neighbor_cutoff) and \
                   (end_pos1 < start_pos2):
                    text1 = AB_stmt_jsons[stmt_id1]['evidence'][0]['text']
                    text2 = BC_stmt_jsons[stmt_id2]['evidence'][0]['text']
                    rows.append([curie1,
                                 AB_stmts_type_dict[stmt_id1],
                                 curie2,
                                 BC_stmts_type_dict[stmt_id2],
                                 curie3,
                                 text1,
                                 text2,
                                 sentence_id1,
                                 sentence_id2,
                                 reading_id])

    df = pd.DataFrame(rows, columns=['agent1', 'stmt_type1',
                                     'agent2', 'stmt_type2',
                                     'agent3', 'text1', 'text2',
                                     'sentence_id1', 'sentence_id2',
                                     'reading_id'])
    return df
