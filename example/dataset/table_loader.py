import os.path as osp
import json

class BaseTableLoader:
    def __init__(self, dataroot: str, version: str):
        self.dataroot = dataroot
        self.version = version

    @property
    def table_root(self) -> str:
        """Returns the folder where the tables are stored for the relevant version."""
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name: str, drop_fields=None) -> dict:
        """Loads a table from a JSON file."""
        with open(osp.join(self.table_root, f'{table_name}.json')) as f:
            table = json.load(f)
        
        # Drop specified fields
        drop_fields = drop_fields or []
        if drop_fields:
            for record in table:
                for field in drop_fields:
                    record.pop(field, None)
        return table

