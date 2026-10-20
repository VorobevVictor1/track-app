import tomllib
import tomli_w
from pathlib import Path
from dataclasses import dataclass, asdict, replace

@dataclass(frozen=True)
class AppConfig:
    sample_dir: str
    apply_mm_patch: bool
    
    def __post_init__(self):
        if not isinstance(self.sample_dir, str):
            raise TypeError(f'String object expected, got insted {type(self.sample_dir)}')
        elif not Path(self.sample_dir).resolve().is_dir():
            raise ValueError(f'Sample path must point to an existing sample dir,\ngot insted {Path(self.sample_dir).resolve()}')
        
        if not isinstance(self.apply_mm_patch, bool):
            raise TypeError(f'Bool object expected, got insted {type(self.apply_mm_patch)}')


class ConfigService:
    def __init__(self, path: Path, cfg: AppConfig) -> None:
        self.config_path = path
        self.settings = cfg
        

    @classmethod
    def from_file(cls, path: Path) -> "ConfigService":
        """
        Major way to create a config service

        Args:
            path (Path): Path to the config file

        Raises:
            FileNotFoundError: path is incorrect
            ValueError: config file is incorrect

        Returns:
            ConfigService: ready-to-use instance of service
        """
        if path.suffix != '.toml' or not path.is_file():
            raise FileNotFoundError(f'Unable to find correct toml file on path: {path}')
        else:
            with open(path, "rb") as f:
                try:
                    data = tomllib.load(f)
                    cfg = AppConfig(**data)
                except tomllib.TOMLDecodeError as e:
                    raise tomllib.TOMLDecodeError(f'Config file has incorrect value: {e}')
            
            return cls(path=path, cfg=cfg)
    
    def save(self, new_cfg: AppConfig) -> None:
        """Check whether input is correct toml and saves new config"""
        data_to_save = asdict(new_cfg)

        try:
            new_toml_str = tomli_w.dumps(data_to_save)
            tomllib.loads(new_toml_str)
        except tomllib.TOMLDecodeError as e:
            raise tomllib.TOMLDecodeError(f'Invalid toml structure from UI input: {e}')
        else:
            with open(self.config_path, 'wb') as f:
                tomli_w.dump(data_to_save, f)

            self.settings = new_cfg


    def update_settings(self, **kwargs) -> None:
        """Checks wheter UI input is correct config object"""
        try:
            new_settings = replace(self.settings, **kwargs)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid config UI input: {e}')
        
        self.save(new_settings)

config_instance: ConfigService | None = None