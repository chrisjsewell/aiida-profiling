"""Useful queries for profiling PostgreSQL databases

These queries are mainly adapted from
https://gist.github.com/anvk/475c22cbca1edc5ce94546c871460fdd
"""
from functools import wraps

import pandas as pd


class PostGresAnalysis:
    """Context for analysing the database for a contained period.
    
    Recorded data only pertains to the context period.
    """

    def __init__(self, pg_stat=True, query_limit=100):
        self._pg_stat = pg_stat
        self._query_limit = query_limit
        self.pg_stats = None
        self.tables_diff = None
        self.indices_diff = None

    def __enter__(self):
        self._init_memory_tables = memory_tables_df()
        self._init_indices_stats = indices_stats_df()
        if self._pg_stat:
            query_reset_stats()
        return self

    def __exit__(self, *args):
        if self._pg_stat:
            self.query_stats = query_stats_df(limit=self._query_limit)
        self.tables_diff = memory_tables_df().subtract(self._init_memory_tables)
        self.indices_diff = indices_stats_df().subtract(self._init_indices_stats)
        self.indices_diff.drop("rows", axis=1, inplace=True)


def execute_raw(raw):
    from aiida.manage.manager import get_manager

    backend = get_manager()._load_backend(schema_check=False)
    return backend.execute_raw(raw)


# ------------------
# -- Memory Size  --
# ------------------


def memory_db_df():
    result = execute_raw(
        r"""
    SELECT 
        datname,
        pg_database_size(datname)
        from pg_database
        order by pg_database_size(datname);
    """
    )
    df = pd.DataFrame(result, columns=["database", "size_mb"])
    df["size_mb"] = df["size_mb"] * 1e-6
    return df


def memory_pg_classes_df():
    """Return size of `pg_class`'s

    `pg_class` catalogs tables and most everything else that has columns,
    or is otherwise similar to a table.
    See https://www.postgresql.org/docs/9.3/catalog-pg-class.html
    """
    result = execute_raw(
        r"""
    SELECT 
        sum(pg_relation_size(pg_class.oid))::bigint,
        nspname,
        CASE pg_class.relkind
            WHEN 'r' THEN 'table'
            WHEN 'i' THEN 'index'
            WHEN 'S' THEN 'sequence'
            WHEN 'v' THEN 'view'
            WHEN 't' THEN 'toast'
            ELSE pg_class.relkind::text
        END
    FROM pg_class
    LEFT OUTER JOIN pg_namespace ON (pg_namespace.oid = pg_class.relnamespace)
    GROUP BY pg_class.relkind, nspname
    ORDER BY sum(pg_relation_size(pg_class.oid)) DESC;
    """
    )
    df = pd.DataFrame(result, columns=["size_mb", "namespace", "relkind"])
    df.sort_index(axis=1, inplace=True)
    df["size_mb"] = df.size_mb * 1e-6
    return df


def memory_tables_df():
    """Return statistics on indices.
    
    See https://www.postgresql.org/docs/current/monitoring-stats.html
    """
    result = execute_raw(
        r"""
    select 
        relname, 
        pg_relation_size(relname::regclass) as table_size,
        pg_total_relation_size(relname::regclass) - pg_relation_size(relname::regclass) as index_size,
        pg_total_relation_size(relname::regclass) as total_size
    from pg_stat_user_tables
    """
    )
    df = pd.DataFrame(result, columns=["name", "table_mb", "indices_mb", "total_mb"])
    df.set_index("name", inplace=True)
    df = df * 1e-6
    df.sort_values("total_mb", ascending=False, inplace=True)
    return df


# -------------
# -- Indices --
# -------------


def indices_list_df():
    """Return list of indices by table and columns."""
    result = execute_raw(
        r"""
    select
        t.relname as table_name,
        i.relname as index_name,
        string_agg(a.attname, ',') as column_name
    from
        pg_class t,
        pg_class i,
        pg_index ix,
        pg_attribute a
    where
        t.oid = ix.indrelid
        and i.oid = ix.indexrelid
        and a.attrelid = t.oid
        and a.attnum = ANY(ix.indkey)
        and t.relkind = 'r'
        and t.relname not like 'pg_%'
    group by  
        t.relname,
        i.relname
    order by
        t.relname,
        i.relname;
    """
    )
    df = pd.DataFrame(result, columns=["table", "index", "columns"])
    df.set_index(["table", "columns"], inplace=True)
    return df


def indices_stats_df(sort_size=False, with_sql=False):
    """Return statistics on indices.
    
    See https://www.postgresql.org/docs/current/monitoring-stats.html
    """
    result = execute_raw(
        r"""
    SELECT
        pt.tablename AS TableName,
        t.indexname AS IndexName,
        pc.reltuples AS TotalRows,
        pg_relation_size(quote_ident(pt.tablename)::text) AS TableSize,
        pg_relation_size(quote_ident(t.indexrelname)::text) AS IndexSize,
        t.idx_scan AS TotalNumberOfScan,
        t.idx_tup_read AS TotalTupleRead,
        t.idx_tup_fetch AS TotalTupleFetched,
        pgi.indexdef AS IndexDef
    FROM pg_tables AS pt
    LEFT OUTER JOIN pg_class AS pc 
        ON pt.tablename=pc.relname
    LEFT OUTER JOIN
    ( 
        SELECT 
            pc.relname AS TableName,
            pc2.relname AS IndexName,
            psai.idx_scan,
            psai.idx_tup_read,
            psai.idx_tup_fetch,
            psai.indexrelname 
        FROM 
            pg_index AS pi
        JOIN pg_class AS pc 
            ON pc.oid = pi.indrelid
        JOIN pg_class AS pc2 
            ON pc2.oid = pi.indexrelid
        JOIN pg_stat_all_indexes AS psai 
            ON pi.indexrelid = psai.indexrelid 
    ) AS T
        ON pt.tablename = T.TableName
    LEFT OUTER JOIN pg_indexes as pgi
        ON T.indexname = pgi.indexname
    WHERE pt.schemaname='public'
    ORDER BY 1;
    """
    )
    columns = [
        "table",
        "index",
        "rows",
        "table_size_mb",
        "index_size_mb",
        # Number of index scans initiated on this index
        "scans",
        # Number of index entries returned by scans on this index
        "read",
        # Number of live rows fetched by index scans
        "fetched",
        "sql",
    ]
    df = pd.DataFrame(result, columns=columns)
    df.set_index(["table", "index"], inplace=True)
    df["table_size_mb"] = df.table_size_mb * 10e-6
    df["index_size_mb"] = df.index_size_mb * 10e-6
    if not with_sql:
        df.drop("sql", axis=1, inplace=True)
    if sort_size:
        df.sort_values("index_size_mb", ascending=False, inplace=True)
    else:
        df.sort_index(axis=0, inplace=True)
    return df


def indices_check_df(min_size_mb=0.1):
    """Check for tables that may requie an index."""
    result = execute_raw(
        r"""
    SELECT
        relname,
        seq_scan,
        idx_scan,
        pg_relation_size(relname::regclass) AS rel_size,
        n_live_tup
    FROM pg_stat_all_tables
    WHERE schemaname='public' AND pg_relation_size(relname::regclass)>{min_size};
    """.format(
            min_size=int(min_size_mb * 1e6)
        )
    )
    df = pd.DataFrame(
        result,
        columns=[
            "table",
            # Number of sequential scans initiated on this table
            "seq_scans",
            # Number of index scans initiated on this table
            "idx_scans",
            "size_mb",
            "live_rows",
        ],
    )
    df["idx_usage"] = 100 * df.idx_scans / (df.seq_scans + df.idx_scans)
    df["idx_required"] = (df.seq_scans - df.idx_scans) > 0
    df["size_mb"] = df["size_mb"] * 1e-6
    df.set_index("table", inplace=True)
    return df


# --------------------
# -- Data Integrity --
# --------------------


def cache_hit_ratio():
    """Ideally hit_ration should be > 90%"""
    result = execute_raw(
        r"""
    SELECT 
        sum(blks_hit)*100/sum(blks_hit+blks_read) as hit_ratio
        from pg_stat_database;
    """
    )
    return float(result[0][0])


def anomalies_df():
    """
    - c_commit_ratio should be > 95%
    - c_rollback_ratio should be < 5%
    - deadlocks should be close to 0
    - conflicts should be close to 0
    - temp_files and temp_bytes watch out for them
    """
    result = execute_raw(
        r"""
    SELECT 
        datname,
        (xact_commit*100)/nullif(xact_commit+xact_rollback,0) as c_commit_ratio,
        (xact_rollback*100)/nullif(xact_commit+xact_rollback, 0) as c_rollback_ratio,
        deadlocks,
        conflicts,
        temp_files,
        temp_bytes
    FROM pg_stat_database;
    """
    )
    df = pd.DataFrame(
        result,
        columns=[
            "database",
            "commit_ratio",
            "rollback_ratio",
            "deadlocks",
            "conflicts",
            "temp_files",
            "temp_size_mb",
        ],
    )
    df["temp_size_mb"] = df["temp_size_mb"] * 1e-6
    return df


def write_activity_df(limit=50):
    """
    hot_rate = rows HOT updated / total rows updated
    (Heap Only Tuple means with no separate index update required)
    
    Heap Only Tuple (HOT) means, creating a new update tuple if possible on the same page as the old tuple.
    Ideally hot_rate should be close to 100.
    You might be blocking HOT updates with indexes on updated columns. If those are expendable, you might get          better overall performance without them.
    """
    result = execute_raw(
        r"""
    SELECT
        s.relname,
        pg_relation_size(relid),
        coalesce(n_tup_ins,0) + 2 * coalesce(n_tup_upd,0) -
        coalesce(n_tup_hot_upd,0) + coalesce(n_tup_del,0) AS total_writes,
        (coalesce(n_tup_hot_upd,0)::float * 100 / (case when n_tup_upd > 0 then n_tup_upd else 1 end)::float) AS hot_rate
        /* This returns None
        (SELECT v[1] FROM regexp_matches(reloptions::text,E'fillfactor=(d+)') as r(v) limit 1) AS fillfactor
        */
    from pg_stat_all_tables
    s join pg_class c ON c.oid=relid
    order by total_writes desc
    limit {limit};
    """.format(
            limit=limit
        )
    )
    columns = [
        "table",
        "size_mb",
        "writes",
        "hot_rate",
        # "fill_factor"
    ]
    df = pd.DataFrame(result, columns=columns)
    df["size_mb"] = df["size_mb"] * 1e-6
    df.set_index("table", inplace=True)
    return df


# How many indexes are in cache
def cached_indices():
    result = execute_raw(
        r"""
    SELECT 
        sum(idx_blks_read) as idx_read,
        sum(idx_blks_hit) as idx_hit,
        (sum(idx_blks_hit) - sum(idx_blks_read)) / sum(idx_blks_hit) as ratio
    FROM pg_statio_user_indexes;
    """
    )
    return cached_indices


def dirty_pages():
    """maxwritten_clean and buffers_backend_fsyn should be 0"""
    result = execute_raw(
        r"""
    SELECT buffers_clean, maxwritten_clean, buffers_backend_fsync from pg_stat_bgwriter;
    """
    )
    return pd.Series(
        dict(
            zip(
                ("buffers_clean", "maxwritten_clean", "buffers_backend_fsync"),
                result[0],
            )
        )
    )


# -------------
# -- Queries --
# -------------


def requires_pg_stat(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        try:
            return func(*args, **kwds)
        except Exception as err:
            if 'relation "pg_stat_statements" does not exist' in str(err):
                raise RuntimeError(
                    "This function requires that the pg_stat_statements extension is initialised on your database"
                )
            raise

    return wrapper


@requires_pg_stat
def query_reset_stats():
    return execute_raw("select pg_stat_statements_reset();")


@requires_pg_stat
def query_stats_df(limit=100):
    """Return most CPU intensive queries
    
    See: https://www.postgresql.org/docs/9.4/pgstatstatements.html
    """
    result = execute_raw(
        r"""
    SELECT 
        query,
        round(total_time::numeric, 2) AS total_time,
        calls,
        rows,
        round((100 * total_time / sum(total_time::numeric) OVER ())::numeric, 2) AS percentage_cpu
    FROM pg_stat_statements
    ORDER BY total_time DESC
    LIMIT {limit};
    """.format(
            limit=limit
        )
    )
    # avg_time = total_time / calls
    df = pd.DataFrame(
        result, columns=["sql", "time_seconds", "calls", "rows", "cpu_percent"]
    )
    df["time_seconds"] = df["time_seconds"].astype(float) * 1e-6
    df["type"] = df.sql.apply(lambda s: s.split()[0].upper())
    return df


@requires_pg_stat
def query_write_df():
    """Return most writing (to shared_buffers) queries
    
    See: https://www.postgresql.org/docs/9.4/pgstatstatements.html
    """
    result = execute_raw(
        r"""
    SELECT
        query, 
        shared_blks_dirtied 
    from pg_stat_statements 
    where shared_blks_dirtied > 0
    order by 2 desc;
    """
    )
    return pd.DataFrame(result, columns=["sql", "blocks_written"])
