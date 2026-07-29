import argparse
import signal
import sys
from pathlib import Path
from threading import Event

from prime_rl.configs.rl import RLConfig
from prime_rl.external_cluster import run_external_cluster
from prime_rl.utils.config import cli
from prime_rl.utils.process import set_proc_title


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Parse provider identity while preserving Prime config overrides."""

    parser = argparse.ArgumentParser(
        prog="external-cluster",
        description=(
            "Run one rank of a provider-allocated PrimeRL cluster. Unknown options "
            "are applied as normal RLConfig overrides after the config file."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--addresses", nargs="+", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-state-dir", type=Path, required=True)
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    set_proc_title("ExternalCluster")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args, config_overrides = parse_args(raw_argv)
    config = cli(RLConfig, args=["@", str(args.config), *config_overrides])
    cancel_event = Event()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: cancel_event.set())
    try:
        run_external_cluster(
            config,
            rank=args.rank,
            addresses=args.addresses,
            run_id=args.run_id,
            local_state_dir=args.local_state_dir,
            launcher_argv=["external-cluster", *raw_argv],
            cancel_event=cancel_event,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main()
