from __future__ import annotations
import argparse
import logging
from typing import Literal

from arc import LOG
from arc.config import Configs
from arc.pipeline import Results, run

__all__ = ['Arc']

class Arc():
    """
    High-level ARC runner.

    This class provides a lightweight interface for running ARC from Python and
    is used by the ``arc`` console script. It runs :func:`arc.pipeline.run`;
    the legacy :mod:`arc.Automated_Rating_Curve_Generator` is no longer called.
    """
    _mifn: str = ""
    _args: dict = {}

    def __init__(self, mifn: str = "", args: dict | None = None, quiet: bool = False, processes: int | Literal["auto"] = 1) -> None:
        """Initialize an `Arc` instance.

        Parameters
        ----------
        mifn : str, optional
            Path to an ARC model input file (MIF), YAML or tab-separated text.
        args : dict or None, optional
            Dictionary of key-value pairs corresponding to ARC input-file arguments. Will only be used if `mifn` is not provided.
        quiet : bool, optional
            If True, suppress progress bars and non-error log output.
        processes : int or {"auto"}, optional
            Kept for compatibility with the legacy runner. The pipeline runs in one process, so any other value is
            ignored with a warning.

        Returns
        -------
        None
        """
        self._mifn = mifn
        self._args = args or {}
        self._quiet = quiet
        self._processes = processes
        if quiet:
            self.set_log_level('error')

    def run(self) -> Results:
        """
        Run ARC.

        Returns
        -------
        Results
            What the run made. Outputs are also written to disk based on input-file arguments.
        """
        if self._mifn:
            LOG.info(f'Main Input File Given: {self._mifn}')
            configs = Configs.from_file(self._mifn)
        elif self._args:
            configs = Configs.from_mapping(self._args)
        else:
            raise ValueError('Arc needs a model input file (mifn) or a dictionary of input arguments (args).')

        if self._processes not in (1, "auto"):
            LOG.warning(f'ARC runs in one process; processes={self._processes!r} is ignored.')

        return run(configs, quiet=self._quiet)

    def set_log_level(self, log_level: str) -> 'Arc':
        """
        Set ARC's logging verbosity.

        Parameters
        ----------
        log_level : {"debug", "info", "warn", "error"}
            Desired log level.

        Returns
        -------
        Arc
            Self, for chaining.
        """
        handler = LOG.handlers[0]
        if log_level == 'debug':
            LOG.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
        elif log_level == 'info':
            LOG.setLevel(logging.INFO)
            handler.setLevel(logging.INFO)
        elif log_level == 'warn':
            LOG.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
        elif log_level == 'error':
            LOG.setLevel(logging.ERROR)
            handler.setLevel(logging.ERROR)
        else:
            LOG.setLevel(logging.WARNING)
            handler.setLevel(logging.WARNING)
            LOG.warning('Invalid log level. Defaulting to warning.')
            return

        LOG.info('Log Level set to ' + log_level)
        return self

def _main():
    """Command-line entry point for the ``arc`` console script."""
    parser = argparse.ArgumentParser(description='Run ARC')
    parser.add_argument('mifn', type=str, help='Model Input File Name')
    parser.add_argument('-l', '--log', type=str, help='Log Level',
                        default='warn', choices=['debug', 'info', 'warn', 'error'])
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress output progress bar and other non-error messages')
    parser.add_argument('-p', '--processes', type=str, default='1', help='Ignored: ARC runs in one process. Kept for compatibility.')
    args = parser.parse_args()
    processes: int | Literal["auto"]
    processes = "auto" if args.processes.strip().lower() == "auto" else int(args.processes)
    arc = Arc(args.mifn, args=None, quiet=args.quiet, processes=processes)
    arc.set_log_level(args.log)
    arc.run()

    LOG.info('Finished')



if __name__ == "__main__":
    _main()
