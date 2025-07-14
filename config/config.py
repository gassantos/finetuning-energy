
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    settings_files=['settings.toml', '.secrets.toml'],
    environments=True,
    env_switcher="APP_ENV",
    validators=[
        Validator("DEBUG", is_type_of=bool),
    ],
    load_dotenv=True,
)
