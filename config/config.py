
from dynaconf import Dynaconf, Validator

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=['settings.toml', '.secrets.toml'],
    environments=True,
    env_switcher=True,
    validators=[
        Validator("DEBUG", default=False, is_type_of=bool),
        Validator("ENVIRONMENT", default="development", is_type_of=str),
    ],
    load_dotenv=True
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.
