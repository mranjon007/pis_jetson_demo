import sys

import hydra
from loguru import logger
from omegaconf import DictConfig

from utils.demo.base import run_demo


@hydra.main(version_base=None, config_path="recipes")
def main(config: DictConfig) -> int:
    return run_demo(config)



if __name__ == "__main__":
    script_name = sys.argv[0]

    if "--config-name" not in sys.argv and "--config-path" not in sys.argv:
        logger.error(
            "Recipe (Hydra config file) mustbe specified with --config-name <recipe_name>"
        )
        logger.error(
            f"Usage: python3 {script_name} [--config-name <recipe_name>] [--config-path <recipe_yaml_path>] [KEY=VALUE, [KEY=VALUE ...]]"
        )
        logger.error(
            f"(Example: python3 {script_name} --config-path recipes/pms2.testvideo.yaml recipe.general.verbose=True)"
        )
        sys.exit(1)
    sys.exit(main())  # pylint: disable=E1120
