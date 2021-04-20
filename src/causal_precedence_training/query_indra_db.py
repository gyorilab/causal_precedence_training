import json
from sqlalchemy import text
from functools import lru_cache
from contextlib import contextmanager

from indra_db.util import get_db
from indra_db.util.helpers import unpack


@contextmanager
def managed_db(db_label='primary', protected=False):
    """Get indra_db handle managed with contextmanager

    Cleans up even if an error occurs while handle is open
    """
    db = get_db(db_label, protected)
    try:
        yield db
    finally:
        db.session.rollback()
        db.session.close()


@lru_cache(maxsize=1024)
def get_pa_statements_for_pair(curie1, curie2):
    """Return dict with info for preassembled statements connecting two agents

    Parameters
    ----------
    curie1 : str
       String of the form f'{namespace}:{identifier}' such as
       'HGNC:6091' or 'FPLX:PI3K'.

    curie2 : str
        See above
    
    Returns
    -------
    dict
        Dictionary mapping stmt_mk_hashes for preassembled statements to
        statement types.
    """
    query = """--
    SELECT
        pa1.stmt_mk_hash, pa1.db_name, pa1.db_id,
        pa2.db_name, pa2.db_id, ps.type, ps.json
    FROM
        pa_agents pa1
    INNER JOIN
        pa_agents pa2
    ON
        pa1.stmt_mk_hash = pa2.stmt_mk_hash AND
        MD5(pa1.db_name || pa1.db_id) = MD5(:db_ns1 || :db_id1) AND
        MD5(pa2.db_name || pa2.db_id) = MD5(:db_ns2 || :db_id2) AND
        pa1.role = 'SUBJECT' AND pa2.role = 'OBJECT'
    INNER JOIN
        pa_statements ps
    ON
        pa2.stmt_mk_hash = ps.mk_hash
    """
    db_ns1, db_id1 = curie1.split(':', maxsplit=1)
    db_ns2, db_id2 = curie2.split(':', maxsplit=1)
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'db_ns1': db_ns1, 'db_id1': db_id1,
                                  'db_ns2': db_ns2, 'db_id2': db_id2})
    # Although absurdly unlikely, we filter MD5 hash collisions just
    # on principle. Also filter complexes with more than two members
    return {stmt_mk_hash: stmt_type for
            stmt_mk_hash, db_name1, id1, db_name2, id2,
            stmt_type, stmt_json in res
            if db_name1 == db_ns1 and id1 == db_id1 and
            db_name2 == db_ns2 and id2 == db_id2}


def get_reach_support_for_pa_statements(stmt_mk_hashes):
    """Return reading_ids and raw_stmt_ids of reach support for input

    Parameters
    ----------
    stmt_mk_hashes : list of int
        List of stmt_mk_hashes for preassembled statements

    Returns
    -------
    generator of tuple
        yields tuples of the form 
        (stmt_mk_hash, raw_stmt_id, reading_id)
        where stmt_mk_hash is the mk_hash of a preassembled statement in the
        input, raw_stmt_id is a raw statement id for a raw statement from REACH
        which supports the preassembled statement, and reading_id is that
        associated reading id in the reading table.
    """
    query = """--
    SELECT rl.pa_stmt_mk_hash, rs.id, rs.reading_id
    FROM
        raw_unique_links rl
    INNER JOIN
        raw_statements rs
    ON 
        rl.pa_stmt_mk_hash IN :stmt_mk_hashes AND
        rl.raw_stmt_id = rs.id
    INNER JOIN
        reading rd
    ON
        rs.reading_id = rd.id AND
        rd.reader = 'REACH'
    """
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'stmt_mk_hashes': tuple(set(stmt_mk_hashes))})
    return ((stmt_mk_hash, raw_stmt_id,
             reading_id) for stmt_mk_hash, raw_stmt_id, reading_id
            in res)


def get_readings_for_reading_ids(reading_ids):
    """Get json output associated to reading ids

    Parameters
    ----------
    reading_ids : list of ints
        reading ids for rows in readings table

    Returns
    -------
    dict
        dict mapping reading ids to jsons of reading output
    """
    query = 'SELECT id, bytes FROM reading WHERE id IN :reading_ids'
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'reading_ids': tuple(set(reading_ids))})
    return {reading_id: json.loads(unpack(bytes_))
            for reading_id, bytes_ in res}


def get_raw_statement_jsons(stmt_ids):
    """Get statement jsons associated to each in a list of raw statement ids

    Parameters
    ----------
    stmt_ids : list of int
        list of raw statement ids

    Returns
    --------
    dict
        dict mapping raw statement ids to statement jsons.
    """
    query = 'SELECT id, json FROM raw_statements WHERE id in :stmt_ids'
    with managed_db() as db:
        res = db.session.execute(text(query),
                                 {'stmt_ids': tuple(set(stmt_ids))})
    return {stmt_id: json.loads(json_.tobytes()) for stmt_id, json_ in res}
