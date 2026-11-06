"""Execute the fixed expanded Phase6 arithmetic mechanism control."""

import argparse
import json
from pathlib import Path

from qvelt.posttrain.phase6_smoke import run_phase6_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_phase6_smoke(args.directory, args.repository)
    # Detailed audits stay on disk; the stdout receipt remains bounded.
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "source_sha256",
                    "reward_increased",
                    "training_reward_first10",
                    "training_reward_last10",
                    "online_offline_weights_adam_rng_losses_exact",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
