"""Categorise wrong predictions by comparing predicted and gold SQL structure (sqlglot).

The first matching category wins, roughly from most to least fundamental:
  syntax/execution error -> wrong table/column -> nesting error -> wrong join
  -> wrong aggregation/grouping -> wrong condition -> other (ordering, limit, select details)
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sensei_sql.core.executor import run_sql

CATEGORIES = (
    "syntax/execution error",
    "wrong table/column",
    "nesting error",
    "wrong join",
    "wrong aggregation/grouping",
    "wrong condition",
    "other (ordering/limit/select)",
)

AGG_FUNCS = (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)


@dataclass
class Structure:
    tables: frozenset
    select_cols: frozenset
    join_cols: frozenset
    where_cols: frozenset
    subqueries: int
    set_ops: int
    joins: int
    aggs: tuple
    group_by: frozenset
    has_having: bool
    where_ops: tuple
    literals: frozenset


def _norm(x: str) -> str:
    return x.lower().strip('"`[]')


def _cols(nodes, aliases) -> frozenset:
    return frozenset(
        _norm(c.name) for n in nodes if n is not None for c in n.find_all(exp.Column) if _norm(c.name) not in aliases
    ) - {"*"}


def structure(sql: str) -> Structure:
    tree = sqlglot.parse_one(sql, read="sqlite")
    selects = list(tree.find_all(exp.Select))
    aliases = {_norm(a.alias) for a in tree.find_all(exp.Alias)}
    joins = list(tree.find_all(exp.Join))
    conds = [s.args.get(k) for s in selects for k in ("where", "having")]
    cond_nodes = [n for c in conds if c is not None for n in c.walk()]
    return Structure(
        tables=frozenset(_norm(t.name) for t in tree.find_all(exp.Table)),
        select_cols=_cols([e for s in selects for e in s.expressions], aliases),
        join_cols=_cols([j.args.get("on") for j in joins], aliases),
        where_cols=_cols(conds, aliases),
        subqueries=sum(1 for _ in tree.find_all(exp.Subquery))
        + sum(1 for s in selects if s is not tree and not isinstance(s.parent, (exp.SetOperation, exp.Subquery))),
        set_ops=sum(1 for _ in tree.find_all(exp.SetOperation)),
        joins=len(joins),
        aggs=tuple(sorted(type(a).__name__ for a in tree.find_all(*AGG_FUNCS))),
        group_by=_cols([s.args.get("group") for s in selects], aliases),
        has_having=any(s.args.get("having") for s in selects),
        where_ops=tuple(sorted(type(n).__name__ for n in cond_nodes if isinstance(n, (exp.Predicate, exp.Not, exp.And, exp.Or)))),
        literals=frozenset(str(n.this).lower() for n in cond_nodes if isinstance(n, exp.Literal)),
    )


def categorise(db_path, gold_sql: str, pred_sql: str) -> str:
    if not pred_sql:
        return CATEGORIES[0]
    res = run_sql(db_path, pred_sql, timeout_s=10)
    if not res.ok:
        # hallucinated schema names are schema-linking errors, not syntax errors
        if res.error and ("no such column" in res.error or "no such table" in res.error):
            return "wrong table/column"
        return CATEGORIES[0]
    try:
        g, p = structure(gold_sql), structure(pred_sql)
    except SqlglotError:
        return CATEGORIES[0]
    if (g.subqueries, g.set_ops) != (p.subqueries, p.set_ops):
        return "nesting error"
    if g.tables != p.tables:
        # an extra/missing join table with the same selected columns is a join problem
        return "wrong join" if g.select_cols == p.select_cols else "wrong table/column"
    if g.select_cols != p.select_cols:
        return "wrong table/column"
    if (g.joins, g.join_cols) != (p.joins, p.join_cols):
        return "wrong join"
    if (g.aggs, g.group_by, g.has_having) != (p.aggs, p.group_by, p.has_having):
        return "wrong aggregation/grouping"
    if (g.where_cols, g.where_ops, g.literals) != (p.where_cols, p.where_ops, p.literals):
        return "wrong condition"
    return CATEGORIES[-1]
