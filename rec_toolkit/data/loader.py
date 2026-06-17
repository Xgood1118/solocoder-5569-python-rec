import os
import pandas as pd
from typing import Optional
from .schema import get_schema, DataSchema


class DataLoader:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _get_file_path(self, data_type: str) -> str:
        return os.path.join(self.data_dir, f'{data_type}.csv')

    def load_csv(self, data_type: str, validate: bool = True) -> pd.DataFrame:
        file_path = self._get_file_path(data_type)
        if not os.path.exists(file_path):
            return pd.DataFrame()

        df = pd.read_csv(file_path)

        if validate:
            schema = get_schema(data_type)
            errors = schema.validate(df)
            if errors:
                raise ValueError(f"CSV 校验失败 ({data_type}):\n" + "\n".join(errors))

        return df

    def save_csv(self, data_type: str, df: pd.DataFrame, validate: bool = True):
        if validate:
            schema = get_schema(data_type)
            errors = schema.validate(df)
            if errors:
                raise ValueError(f"数据校验失败 ({data_type}):\n" + "\n".join(errors))

        file_path = self._get_file_path(data_type)
        df.to_csv(file_path, index=False)

    def batch_import(self, data_type: str, df: pd.DataFrame, validate: bool = True) -> int:
        if validate:
            schema = get_schema(data_type)
            errors = schema.validate(df)
            if errors:
                raise ValueError(f"批量导入校验失败 ({data_type}):\n" + "\n".join(errors))

        file_path = self._get_file_path(data_type)
        df.to_csv(file_path, index=False)
        return len(df)

    def incremental_append(self, data_type: str, df: pd.DataFrame, validate: bool = True) -> int:
        if validate:
            schema = get_schema(data_type)
            errors = schema.validate(df)
            if errors:
                raise ValueError(f"增量导入校验失败 ({data_type}):\n" + "\n".join(errors))

        file_path = self._get_file_path(data_type)
        if os.path.exists(file_path):
            existing = pd.read_csv(file_path)
            combined = pd.concat([existing, df], ignore_index=True)
            if data_type == 'users':
                combined = combined.drop_duplicates(subset=['user_id'], keep='last')
            elif data_type == 'items':
                combined = combined.drop_duplicates(subset=['item_id'], keep='last')
        else:
            combined = df

        combined.to_csv(file_path, index=False)
        return len(df)

    def upload_file(self, data_type: str, file_content: str, mode: str = 'batch',
                    validate: bool = True) -> int:
        from io import StringIO
        df = pd.read_csv(StringIO(file_content))

        if mode == 'batch':
            return self.batch_import(data_type, df, validate=validate)
        elif mode == 'incremental':
            return self.incremental_append(data_type, df, validate=validate)
        else:
            raise ValueError(f"未知的导入模式: {mode}")

    def validate_csv_content(self, data_type: str, file_content: str) -> dict:
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(file_content))
        except Exception as e:
            return {'valid': False, 'errors': [f'CSV 解析失败: {str(e)}']}

        schema = get_schema(data_type)
        errors = schema.validate(df)

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'row_count': len(df),
            'columns': list(df.columns),
        }
