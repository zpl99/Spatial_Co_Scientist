import logging
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import pandas as pd

from .idris_base import IdrisException

# TODO: Move the import into the __init__ method
try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import wordninja
except ImportError:
    wordninja = None


@dataclass
class IdrisTrainerResult:
    """The result of training a model.
    """
    create_table: str
    context: List[str]
    question_sql: List[Tuple[str, str]]


class IdrisTrainer:
    def __init__(
            self,
            alias: Optional[Dict[str, str]] = None
    ) -> None:
        """Create a new trainer. This module requires the DuckDB and the wordninja modules.

        :param alias: A dictionary of column aliases.
        """
        if duckdb is None:
            raise IdrisException("DuckDB module is required, please install it.")
        if wordninja is None:
            raise IdrisException("wordninja module is required, please install it.")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.alias = alias or {}

    def train(
            self,
            pdf: pd.DataFrame,
            table_name: str,
            alias_name: Optional[str] = None,
            limit: int = 10,
    ) -> IdrisTrainerResult:
        def ninja(text: str) -> str:
            return " ".join(wordninja.split(text.lower()))

        def get_ops(t: str) -> Tuple[str, str]:
            if t == "VARCHAR":
                if random.random() < 0.5:
                    return "is not", "!="
                else:
                    return "is", "="
            else:
                # TODO: Fix this to use random.choice
                return {
                    1: ("is less than", "<"),
                    2: ("is", "="),
                    3: ("is greater than", ">"),
                }[random.randint(1, 3)]

        if alias_name is None:
            alias_name = ninja(table_name)
        with duckdb.connect(":memory:") as conn:
            _ = conn.execute("create or replace view idris as select * from pdf")
            name_type = [
                (row[0], row[1])
                for row in conn.sql("DESCRIBE idris").fetchall()
            ]
            fields = ",\n".join([f"{n} {t}" for n, t in name_type])
            create_table = f"CREATE TABLE {table_name} (\n{fields}\n);"
            name_alias_type = [
                (n, self.alias.get(f"_col:{n}", ninja(n)), t)
                for n, t in name_type
            ]
            context = [f"Use column '{n}' in reference to {a}." for n, a, _ in name_alias_type]
            question_sql = []
            for field_name, field_alias, field_type in name_alias_type:
                self.logger.info(f"Processing {field_name} {field_alias} {field_type}...")
                # TODO - Use the top N common values
                rows = conn.execute(
                    f"""SELECT distinct({field_name}) as '{field_name}'
        FROM idris
        WHERE {field_name} IS NOT NULL
        ORDER BY RANDOM()
        LIMIT {limit}"""
                ).fetchall()
                for (v,) in rows:
                    op1, op2 = get_ops(field_type)
                    alias_key = f"{field_name}:{v}"
                    alias_def = v.lower() if field_type == "VARCHAR" else v
                    o = self.alias.get(alias_key, alias_def)
                    q = f"Show {alias_name} where {field_alias} {op1} {o}"
                    w = v if field_type in ("INTEGER", "FLOAT", "DOUBLE") else f"'{v}'"
                    s = f"SELECT * FROM {table_name} where {field_name}{op2}{w}"
                    question_sql.append((q, s))

        return IdrisTrainerResult(
            create_table=create_table,
            context=context,
            question_sql=question_sql,
        )
