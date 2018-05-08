# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" Helpers for type hints.
"""

# pylint: disable=pointless-statement,unused-argument,missing-docstring,multiple-statements

from types import ModuleType
from typing import Any, Dict, List, Optional

from .config.args import ModuleArgs

from .common.module_results import ModuleResults
from .common.secmet import Record


class ConfigType:
    """ A type for the Config._Config object to ensure the right object types are
        being passed without all the warnings about non-existant members """

    def __getattr__(self, attr: str) -> Any: ...

    def __setattr__(self, attr: str, value: Any) -> None: ...

class AntismashModule(ModuleType):
    """ A type to prevent all the many "ModuleType has no attribute 'run_on_record'"
        errors that mypy will generate throughout the codebase
    """

    NAME: str
    SHORT_DESCRIPTION: str

    @staticmethod
    def run_on_record(record: Record, results: Optional[ModuleResults],
                      options: ConfigType) -> ModuleResults: ...

    @staticmethod
    def regenerate_previous_results(previous: Dict[str, Any], record: Record,
                                    options: ConfigType) -> Optional[ModuleResults]: ...

    @staticmethod
    def is_enabled(options: ConfigType) -> bool: ...

    @staticmethod
    def check_prereqs() -> List[str]: ...

    @staticmethod
    def check_options(options: ConfigType) -> List[str]: ...

    @staticmethod
    def get_arguments() -> ModuleArgs: ...

    # not implemented by every module, but by most
    @staticmethod
    def will_handle(products: List[str]) -> bool: ...
