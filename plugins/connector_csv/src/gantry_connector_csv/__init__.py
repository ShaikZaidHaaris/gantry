"""Gantry connector for flat, long-format CSV trajectory files.

The reference connector: no dependencies beyond core, no cleverness, and small
enough to read in one sitting. It exists so the connector contract gets
debugged against something trivial before it meets a real archive format.
"""

from .connector import VERSION, CsvConnector
from .writer import write_episodes

__all__ = ["VERSION", "CsvConnector", "write_episodes"]
