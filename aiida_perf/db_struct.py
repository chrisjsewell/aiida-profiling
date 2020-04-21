import importlib
import pkgutil
import textwrap

from types import ModuleType
from typing import Dict, List, Type, Union

from graphviz import Digraph
import sqlalchemy as sqla
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import RelationshipProperty


def visualise_sqla(
    structure: Union[sqla.MetaData, DeclarativeMeta],
    border: int = 0,
    cell_border: int = 1,
    cell_spacing: int = 0,
    show_constraints: bool = True,
    show_orm_classes: bool = True,
    show_relationships: bool = True,
    sql_wrap: int = 40,
    graph_name: str = "SQL_Structure",
    **graph_kwargs,
) -> Digraph:
    """Visualise the structure of an SQLAlchemy database.

    See also:
    https://stackoverflow.com/questions/40250987/create-graphviz-dot-file-from-sqlalchemy-models
    """
    if show_relationships and not show_orm_classes:
        raise AssertionError(
            "show_relationships cannot be True if show_orm_classes is False"
        )
    if isinstance(structure, DeclarativeMeta):
        metadata = structure.metadata
    else:
        metadata = structure  # type: sqla.MetaData

    orm_map = {}
    if show_orm_classes and isinstance(structure, DeclarativeMeta):
        for orm in structure._decl_class_registry.values():
            if not hasattr(orm, "__table__"):
                continue
            # relationship_map = {}
            # if show_orm_relationships:
            #     for relationship in inspect(
            #         orm
            #     ).relationships:  # type: RelationshipProperty
            #         target_orm = relationship.mapper.class_
            #         relationship_map[relationship.key] = {
            #             "target_orm": target_orm,
            #         }
            # note a table may have multiple orm classes (i.e. single table inheritance)
            orm_map.setdefault(orm.__table__.fullname, []).append(orm)
            #     {"orm": orm, "relationships": inspect(orm).relationships}
            # )

    graph = Digraph(name=graph_name, **graph_kwargs)
    for table in metadata.sorted_tables:
        graph.node(
            table.fullname,
            table_to_node(
                table,
                orm_classes=orm_map.get(table.fullname, []),
                border=border,
                cell_border=cell_border,
                cell_spacing=cell_spacing,
                show_constraints=show_constraints,
                show_relationships=show_relationships,
                sql_wrap=sql_wrap,
            ),
            shape="plaintext",
        )
    for table in metadata.sorted_tables:  # type: sqla.Table
        for fk in sorted(
            table.foreign_keys, key=lambda f: f.target_fullname
        ):  # type: sqla.ForeignKey
            target = f"{fk.parent.table.fullname}:{fk.parent.name}"
            origin = f"{fk.column.table.fullname}:{fk.column.name}"
            label = []
            if fk.onupdate:
                label.append(f"ON_UPDATE={fk.onupdate}".upper())
            if fk.ondelete:
                label.append(f"ON_DELETE={fk.ondelete}".upper())
            label = "\n".join(label)
            if label:
                # turning the label into a node stops any overlap with tables
                label_node = f"label_{origin}_{target}".replace(":", "_")
                graph.node(label_node, label=label)
                graph.edge(origin, label_node, style="solid")
                graph.edge(label_node, target, style="solid")
            else:
                graph.edge(origin, target, style="solid")

        for orm in orm_map.get(table.fullname, []) if show_relationships else []:
            for rel in inspect(orm).relationships:  # type: RelationshipProperty
                target_orm = rel.mapper.class_
                origin_port = f"{table.fullname}:{rel_port(orm, rel)}"
                target_port = f"{target_orm.__table__.fullname}:{orm_port(target_orm)}"
                graph.edge(origin_port, target_port, style="dashed")

    return graph


def format_type(typ):
    """ Transforms the type into a nice string representation. """
    try:
        return typ.get_col_spec()
    except (AttributeError, NotImplementedError):
        pass
    if isinstance(typ, sqla.sql.sqltypes.Enum):
        return f"ENUM({','.join(typ.enums)})"
    try:
        return str(typ)
    except sqla.exc.CompileError:
        try:
            return typ.__class__.__name__
        except Exception:
            return "OTHER"


def table_to_node(
    table: sqla.Table,
    orm_classes: List[dict] = (),
    border: int = 0,
    cell_border: int = 1,
    cell_spacing: int = 0,
    show_constraints: bool = True,
    show_relationships: bool = True,
    sql_wrap: int = 40,
) -> str:
    # see http://www.graphviz.org/doc/info/shapes.html#html
    columns = table.c._data.values()
    check_constraints = []
    if show_constraints:
        for constraint in table.constraints:
            if isinstance(constraint, sqla.CheckConstraint):
                if constraint.name == "_unnamed_":
                    # used for enum constraints
                    continue
                name = f"{constraint.name} " if constraint.name else ""
                check_constraints.append(f"{name}CHECK ({constraint.sqltext.text})")
            elif isinstance(constraint, sqla.UniqueConstraint):
                name = f"{constraint.name} " if constraint.name else ""
                columns_str = ", ".join([c.name for c in constraint.columns])
                check_constraints.append(f"{name}UNIQUE ({columns_str})")
    # TODO show indexes

    components = []
    # opening
    components.append(
        f'<<TABLE BORDER="{border}" '
        f'CELLBORDER="{cell_border}" CELLSPACING="{cell_spacing}">'
    )
    # header
    components.append(f'<TR><TD COLSPAN="2"><B>{table.fullname}</B></TD></TR>')
    # columns
    for col in columns:
        components.append(
            f'<TR><TD ALIGN="LEFT" PORT="{col.name}">'
            f'{"<u>" + col.name + "</u>" if col.primary_key else col.name}</TD>'
            f'<TD ALIGN="LEFT">{format_type(col.type)}{"?" if col.nullable else ""}'
            "</TD></TR>"
        )
        # TODO show col.server_default?
    if check_constraints:
        components.append(
            '<TR><TD ALIGN="LEFT" COLSPAN="2"><B>Constraints</B></TD></TR>'
        )
    for csql in sorted(check_constraints):
        ccols = textwrap.wrap(csql, sql_wrap)
        for i, col in enumerate(ccols):
            # TODO graphviz issue with rendering only LR, see:
            # https://gitlab.com/graphviz/graphviz/issues/199
            sides = f"{'T' if i==0 else ''}{'B' if i==len(ccols)-1 else ''}LR"
            components.append(
                f'<TR><TD ALIGN="LEFT" COLSPAN="2" SIDES="{sides}">{col}</TD></TR>'
            )
    if orm_classes:
        components.append(
            '<TR><TD ALIGN="LEFT" COLSPAN="2"><B>ORM Classes</B></TD></TR>'
        )
    for orm_class in orm_classes:
        orm_name = orm_class.__name__
        relationships = list(inspect(orm_class).relationships)
        if show_relationships and relationships:
            rel = relationships[0]
            len_rels = len(relationships)
            components.append(
                "<TR>"
                f'<TD ROWSPAN="{len_rels}" ALIGN="LEFT" PORT="{orm_port(orm_class)}">'
                f"{orm_name}</TD>"
                f'<TD ALIGN="LEFT" PORT="{rel_port(orm_class, rel)}">{rel.key}</TD>'
                "</TR>"
            )
            for rel in relationships[1:]:
                components.append(
                    "<TR>"
                    f'<TD ALIGN="LEFT" PORT="{rel_port(orm_class, rel)}">{rel.key}</TD>'
                    "</TR>"
                )
        else:
            components.append(f'<TR><TD COLSPAN="2" ALIGN="LEFT">{orm_name}</TD></TR>')

    # closing
    components.append("</TABLE>>")
    return "\n".join(components)


def orm_port(orm_class: Type) -> str:
    return orm_class.__name__  # TODO also use module?


def rel_port(orm_class: Type, relationship: RelationshipProperty) -> str:
    return f"{orm_class.__name__}_{relationship.key}"  # TODO also use module?


def import_submodules(
    package: Union[str, ModuleType], recursive: bool = True
) -> Dict[str, ModuleType]:
    """Import all submodules of a module, recursively, including subpackages.

    This is useful for loading all ORM models of a `DeclarativeMeta`.
    """
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__):
        full_name = package.__name__ + "." + name
        results[full_name] = importlib.import_module(full_name)
        if recursive and is_pkg:
            results.update(import_submodules(full_name))
    return results
