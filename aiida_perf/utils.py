
def delete_group(group, dry_run=False, verbosity=0, ignore_missing=True):
    """Delete a group, and all the nodes in it."""
    from aiida import orm
    from aiida.common.exceptions import NotExistent
    from aiida.manage.database.delete.nodes import delete_nodes

    if isinstance(group, str):
        try:
            group = orm.Group.get(label=group)
        except NotExistent:
            if ignore_missing:
                return
            raise
    qb = (
        orm.QueryBuilder()
        .append(group.__class__, filters={'id': group.id}, tag='group')
        .append(orm.Node, with_group='group', project=["id"])
    )
    pks = [i for i, in qb.all()]
    delete_nodes(pks, force=True, dry_run=dry_run, verbosity=verbosity)
    if not dry_run:
        orm.Group.objects.delete(group.id)