from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import pandas as pd


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    required: bool = False
    default: Any = None
    validator: Optional[Callable] = None


@dataclass
class DataSchema:
    name: str
    columns: List[ColumnSchema] = field(default_factory=list)

    def validate(self, df: pd.DataFrame) -> List[str]:
        errors = []
        col_names = set(df.columns)

        for col in self.columns:
            if col.required and col.name not in col_names:
                errors.append(f"缺少必填字段: {col.name}")
                continue

            if col.name not in col_names:
                continue

            try:
                if col.dtype == 'int':
                    pd.to_numeric(df[col.name], errors='raise')
                elif col.dtype == 'float':
                    pd.to_numeric(df[col.name], errors='raise')
                elif col.dtype == 'str':
                    df[col.name].astype(str)
                elif col.dtype == 'datetime':
                    pd.to_datetime(df[col.name], errors='raise')
            except (ValueError, TypeError) as e:
                errors.append(f"字段 {col.name} 类型错误，期望 {col.dtype}: {str(e)}")

            if col.validator and col.name in col_names:
                try:
                    if not col.validator(df[col.name]):
                        errors.append(f"字段 {col.name} 验证失败")
                except Exception as e:
                    errors.append(f"字段 {col.name} 验证异常: {str(e)}")

        return errors


USER_SCHEMA = DataSchema(
    name='users',
    columns=[
        ColumnSchema(name='user_id', dtype='str', required=True),
        ColumnSchema(name='age', dtype='int', required=False),
        ColumnSchema(name='gender', dtype='str', required=False),
        ColumnSchema(name='interests', dtype='str', required=False),
        ColumnSchema(name='location', dtype='str', required=False),
        ColumnSchema(name='register_time', dtype='datetime', required=False),
        ColumnSchema(name='extra', dtype='str', required=False),
    ],
)

ITEM_SCHEMA = DataSchema(
    name='items',
    columns=[
        ColumnSchema(name='item_id', dtype='str', required=True),
        ColumnSchema(name='title', dtype='str', required=False),
        ColumnSchema(name='category', dtype='str', required=False),
        ColumnSchema(name='tags', dtype='str', required=False),
        ColumnSchema(name='description', dtype='str', required=False),
        ColumnSchema(name='popularity', dtype='float', required=False),
        ColumnSchema(name='create_time', dtype='datetime', required=False),
        ColumnSchema(name='extra', dtype='str', required=False),
    ],
)

INTERACTION_SCHEMA = DataSchema(
    name='interactions',
    columns=[
        ColumnSchema(name='user_id', dtype='str', required=True),
        ColumnSchema(name='item_id', dtype='str', required=True),
        ColumnSchema(name='rating', dtype='float', required=False, default=1.0),
        ColumnSchema(name='timestamp', dtype='datetime', required=False),
        ColumnSchema(name='behavior_type', dtype='str', required=False, default='view'),
    ],
)


def get_schema(schema_name: str) -> DataSchema:
    schemas = {
        'users': USER_SCHEMA,
        'items': ITEM_SCHEMA,
        'interactions': INTERACTION_SCHEMA,
    }
    if schema_name not in schemas:
        raise ValueError(f"未知的 schema: {schema_name}")
    return schemas[schema_name]
