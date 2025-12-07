import os


__all__ = ['make_table']

TABLES_DIR = './tables'


def make_table(
        filename, row_names, col_names, data, sep=',',
        row_names_format='{0}', col_names_format='{0}', cell_format='{0}'
):
    lines = []

    header = sep.join([''] + [col_names_format.format(name) for name in col_names])
    lines.append(header)

    for i, row_name in enumerate(row_names):
        row_data = [cell_format.format(data[i][j]) for j in range(len(col_names))]
        line = sep.join([row_names_format.format(row_name)] + row_data)
        lines.append(line)

    _save(filename, lines)


def _save(filename, lines):
    full_path = os.path.join(TABLES_DIR, filename)

    if not os.path.exists(full_path):
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

    with open(full_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f'Table saved to {full_path}')