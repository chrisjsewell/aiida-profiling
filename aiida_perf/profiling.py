"""
This is code below is written to aide detailed profiling analysis,
focussed on working within the Jupyter Notebook.
It builds on aspects of code from the following packages:

- https://github.com/jiffyclub/snakeviz
- https://github.com/pwaller/pyprof2calltree
- https://github.com/amcfague/linesman

Example usage::

    with Profiler() as pr:
        run_code_to_profile()

    analyzer = ProfileAnalyzer(pr)
    analyzer.df  # show dataframe of profile
    ax = plot_sunburst(analyzer, depth=12, min_rel=0.2)

"""
import cProfile
from itertools import cycle
from math import ceil
from pstats import Stats
import re
import sys
from typing import List, Union

import networkx as nx
import numpy as np
import pandas as pd


if sys.version_info < (3, 8):

    class Profile(cProfile.Profile):
        def __enter__(self):
            self.enable()
            return self

        def __exit__(self, *args):
            self.disable()


else:
    Profile = cProfile.Profile


class Code(object):
    def __init__(self, filename, firstlineno, name):
        self.co_filename = filename
        self.co_firstlineno = firstlineno
        self.co_name = name

    def __repr__(self):
        return "Code({0},{1},{2})".format(
            self.co_filename, self.co_firstlineno, self.co_name
        )


class Entry(object):
    def __init__(self, code, callcount, reccallcount, inlinetime, totaltime, calls):
        self.code = code
        self.callcount = callcount
        self.reccallcount = reccallcount
        self.inlinetime = inlinetime
        self.totaltime = totaltime
        self.calls = calls

    def __repr__(self):
        return "Entry({0},cc={1},rc={2},it={3:.2e},tt={4:.2e},calls={5})".format(
            self.code,
            self.callcount,
            self.reccallcount,
            self.inlinetime,
            self.totaltime,
            len(self.calls),
        )


class Subentry(object):
    def __init__(self, code, callcount, reccallcount, inlinetime, totaltime):
        self.code = code
        self.callcount = callcount
        self.reccallcount = reccallcount
        self.inlinetime = inlinetime
        self.totaltime = totaltime

    def __repr__(self):
        return "SubEntry({0},cc={1},rc={2},it={3:.2e},tt={4:.2e})".format(
            self.code,
            self.callcount,
            self.reccallcount,
            self.inlinetime,
            self.totaltime,
        )


def pstats2entries(data):
    """Convert serialized pstats back to a list of raw entries.

    Converse operation of cProfile.Profile.snapshot_stats()

    Each profiler_entry is a tuple-like object with the
    following attributes:

        code          code object
        callcount     how many times this was called
        reccallcount  how many times called recursively
        totaltime     total time in this entry
        inlinetime    inline time in this entry (not in subcalls)
        calls         details of the calls

    The calls attribute is a list of profiler_subentry objects:

        code          called code object
        callcount     how many times this is called
        reccallcount  how many times this is called recursively
        totaltime     total time spent in this call
        inlinetime    inline time (not in further subcalls)
    """
    # Each entry's key is a tuple of (filename, line number, function name)
    entries = {}
    allcallers = {}

    # first pass over stats to build the list of entry instances
    for code_info, call_info in data.stats.items():
        # build a fake code object
        code = Code(*code_info)

        # build a fake entry object.  entry.calls will be filled during the
        # second pass over stats
        cc, nc, tt, ct, callers = call_info
        entry = Entry(
            code,
            callcount=cc,
            reccallcount=nc - cc,
            inlinetime=tt,
            totaltime=ct,
            calls=[],
        )

        # collect the new entry
        entries[code_info] = entry
        allcallers[code_info] = list(callers.items())

    # second pass of stats to plug callees into callers
    for entry in entries.values():
        entry_label = cProfile.label(entry.code)
        entry_callers = allcallers.get(entry_label, [])
        for entry_caller, call_info in entry_callers:
            cc, nc, tt, ct = call_info
            subentry = Subentry(
                entry.code,
                callcount=cc,
                reccallcount=nc - cc,
                inlinetime=tt,
                totaltime=ct,
            )
            # entry_caller has the same form as code_info
            entries[entry_caller].calls.append(subentry)

    return list(entries.values())


def entry_to_dict(stat, total_time=None, path_replace=None):
    code = stat.code
    if isinstance(code, str):
        code = Code(*cProfile.label(code))
    path = code.co_filename
    for pattern, repl in (path_replace or {}).items():
        path, _ = re.subn(pattern, repl, path)
    data = {
        "callcount": stat.callcount,
        "inlinetime": stat.inlinetime,
        "reccallcount": stat.reccallcount,
        "totaltime": stat.totaltime,
        "calls": len(stat.calls or []),
        "path": path,
        "lineno": code.co_firstlineno,
        "method": code.co_name,
    }
    if total_time is not None:
        data["percent"] = 100 * stat.totaltime / total_time
    return data


def entries_to_df(entries, path_replace=None):
    return pd.DataFrame(
        [entry_to_dict(e, path_replace=path_replace) for e in entries]
    ).sort_values("totaltime", ascending=False)


def _code_str(code):
    if isinstance(code, str):
        code = Code(*cProfile.label(code))
    return f"{code.co_filename},{code.co_firstlineno},{code.co_name}"


def create_graph(stats: List[Entry], path_replace=None):
    """
    Given an instance of :class:`pstats.Pstats`, this will use the generated
    call data to create a graph using the :mod:`networkx` library.  Node
    and edge information is stored in the graph itself, so that the stats
    object itself--which can't be pickled--does not need to be kept around.

    ``stats``:
        An instance of :class:`pstats.Pstats`, usually retrieved by calling
        :func:`~cProfile.Profile.getstats()` on a cProfile object.

    Returns a :class:`networkx.DiGraph` containing the callgraph.
    """
    g = nx.DiGraph(name="profile_stats")
    pks = {}

    for stat in stats:
        code_str = _code_str(stat.code)
        pks[code_str] = pks.get(code_str, len(pks))
        g.add_node(pks[code_str], **entry_to_dict(stat, path_replace=path_replace))

    for stat in stats:
        # Add all the calls as edges
        total = sum([c.totaltime for c in (stat.calls or [])])
        for call in stat.calls or []:
            call_attrs = {
                "callcount": call.callcount,
                "inlinetime": call.inlinetime,
                "reccallcount": call.reccallcount,
                "totaltime": call.totaltime,
                "fraction": (call.totaltime / total) if total else 1,
            }
            g.add_edge(
                pks[_code_str(stat.code)], pks[_code_str(call.code)], **call_attrs
            )

    return g


class ProfileAnalyzer:
    def __init__(self, obj: Union[str, Stats, cProfile.Profile], path_replace=None):

        if isinstance(obj, cProfile.Profile):
            self._stats = Stats(obj)
            # TODO this produces a different order of entries
            # self._entries = obj.getstats()
        elif isinstance(obj, Stats):
            self._stats = obj
        else:
            self._stats = Stats(obj)

        self._entries = pstats2entries(self._stats)

        self._path_replace = path_replace
        self._df = None
        self._graph = None

    def save(self, filename):
        self._stats.dump_stats(filename)

    @property
    def entries(self):
        return self._entries[:]

    def __getitem__(self, entry):
        return self._entries[entry]

    @property
    def graph(self):
        if self._graph is None:
            self._graph = create_graph(self._entries, self._path_replace)
        return self._graph

    @property
    def df(self):
        if self._df is None:
            self._df = entries_to_df(self._entries, self._path_replace)
        return self._df

    def filter_df(self, method=None, path=None):
        f = np.array([True for _ in range(self.df.shape[0])])
        if method is not None:
            f = np.logical_and(f, self.df.method.str.contains(method))
        if path is not None:
            f = np.logical_and(f, self.df.path.str.contains(path))
        return self.df[f]

    def get_root(self, single=True):
        nodes = [n for n, d in self.graph.in_degree() if d == 0]
        if not single:
            return nodes
        return self.df.loc[nodes].totaltime.idxmax()

    def get_total_time(self):
        nodes = self.get_root(single=False)
        return self.df.loc[nodes].totaltime.sum()

    def get_subgraph(
        self, root=None, depth=10, min_rel=1e-3, min_abs=1e-3, remove_cycles=False
    ) -> nx.DiGraph:
        """Return a subgraph, centred around a root node """
        root = root or self.get_root()
        g = self.graph.copy()
        g.remove_edges_from(
            [
                (u, v)
                for u, v, d in g.edges(data=True)
                if (d["fraction"] < min_rel) or (d["totaltime"] < min_abs)
            ]
        )
        g = nx.ego_graph(g, root, depth)
        if remove_cycles:
            g = break_cycles(g, copy=False)
        return g

    def guess_longest_path(self, root=None, edge_length="totaltime", ignore_path="~"):
        return guess_longest_path(
            self, root=root, edge_length=edge_length, ignore_path=ignore_path
        )
    
    def shortest_path(self, node, root=None, edge_length="totaltime"):
        root = root or self.get_root()
        return self.df.loc[nx.shortest_path(self.graph, root, node, weight=edge_length)]


def break_cycles(g, copy=True):
    if copy:
        g = g.copy()
    try:
        while True:
            cycle = nx.find_cycle(g)
            g.remove_edge(*cycle[-1])
    except nx.NetworkXNoCycle:
        pass
    return g


def code_str(d):
    return f'{d["path"]}:{d["lineno"]}({d["method"]})'


def guess_longest_path(p, root=None, edge_length="totaltime", ignore_path="~"):
    """Find the longest call chain, based on largest call times of children.
    
    By default we ignore built in paths (denoted ~)
    """
    node_chain = [root or p.get_root()]
    edge_chain = []
    while p.graph.adj[node_chain[-1]]:
        nodes = [
            (n, d)
            for n, d in p.graph.adj[node_chain[-1]].items()
            if ignore_path is None
            or not re.match(ignore_path, p.graph.nodes[n]["path"])
        ]
        if not nodes:
            break
        node = max(nodes, key=lambda n: n[1][edge_length])[0]
        if node in node_chain:
            break
        edge_chain.append((node_chain[-1], node))
        node_chain.append(node)

    return p.df.loc[node_chain]


def plot_sunburst(
    prof_analyze,
    root=None,
    highlight=(),
    depth=10,
    min_rel=0.1,
    min_abs=0.1,
    label_func=None,
    max_rows=12,
    color_edge="black",
    color_highlight="red",
    add_total_time=False,
):
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    root = root or prof_analyze.get_root()
    total_time = prof_analyze[root].totaltime
    label_func = label_func or code_str
    if isinstance(highlight, (int, str)):
        highlight = [highlight]
    g = prof_analyze.get_subgraph(
        root=root, depth=depth, min_rel=min_rel, min_abs=min_abs, remove_cycles=False
    )

    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_axis_off()

    color_cycle = cycle(get_cmap("tab20")(np.linspace(0, 1, 20)))
    legend_map = {}

    circle = np.pi * 2
    segments = [(root, 0, circle)]
    curr_depth = 1

    while segments and curr_depth <= depth:
        new_segs = []
        for root, theta, radius in segments:

            label = label_func(g.nodes[root])
            leg_label = None if label in legend_map else label
            if label not in legend_map:
                legend_map[label] = next(color_cycle)

            ax.bar(
                theta,
                curr_depth,
                radius,
                align="edge",
                zorder=-curr_depth,
                color=legend_map[label],
                edgecolor=color_highlight if root in highlight else color_edge,
                label=leg_label,
            )
            for child in g.successors(root):
                fraction = g.edges[(root, child)]["fraction"]
                new_segs.append((child, theta, radius * fraction))
                theta += radius * fraction

        segments = new_segs
        curr_depth += 1

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        ncol=ceil(len(legend_map) / (max_rows)),
    )
    if add_total_time:
        if total_time < 1e-2 or total_time > 1e3:
            ax.set_title(f"Time = {total_time:.2e}s")
        else:
            ax.set_title(f"Time = {total_time:.2f}s")
    return ax
