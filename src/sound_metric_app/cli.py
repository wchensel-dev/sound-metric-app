"""Command-line entry point: analyze a Dewesoft file and print/store metrics."""

from __future__ import annotations

import argparse

from .dsp import MetricsProcessor
from .ingestion import list_channels, read_frame
from .storage import ResultsDatabase


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sma-analyze",
        description="Compute Peak dB, Peak dBA, Peak Impulse, Peak Leq(10ms) and "
        "LIAeq,100ms for a DewesoftX .dxd/.d7d file.",
    )
    parser.add_argument("file", help="Path to a .dxd / .d7d file")
    parser.add_argument("-c", "--channel", help="Channel name (auto-detected if omitted)")
    parser.add_argument("--list", action="store_true", help="List channels and exit")
    parser.add_argument("--store", metavar="DB", help="Also store results in this SQLite DB")
    args = parser.parse_args(argv)

    if args.list:
        for c in list_channels(args.file):
            print(f"  {c.name!r:24} unit={c.unit!r:8} fs={c.sample_rate:>10.1f} n={c.n_samples}")
        return 0

    frame = read_frame(args.file, channel=args.channel)
    result = MetricsProcessor().process(frame)

    print(f"File     : {result.source_file}")
    print(f"Channel  : {result.channel}  ({result.sample_rate:.0f} Hz, {result.n_samples} samples)")
    print(f"  Peak Pa               : {result.peak_pa:8.2f} Pa")
    print(f"  Peak dB               : {result.peak_db:8.2f} dB")
    print(f"  Peak dBA              : {result.peak_dba:8.2f} dB(A)")
    print(f"  Peak Impulse          : {result.impulse_pa_ms:8.2f} Pa*ms "
          f"({result.peak_impulse_db:.2f} dB*ms)")
    print(f"  Peak Leq(10ms) dBA    : {result.leq10ms_db:8.2f} dB(A)")
    print(f"  LIAeq,100ms  dBA      : {result.liaeq_100ms_db:8.2f} dB(A)")

    if args.store:
        with ResultsDatabase(args.store) as db:
            rid = db.add_result(result)
            print(f"Stored as row id {rid} in {args.store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
