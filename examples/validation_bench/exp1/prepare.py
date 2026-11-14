"""Download only the public, hash-pinned inputs for experiment one."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path


def fetch(path, url, expected):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != expected:
        raise ValueError(f"Public input hash mismatch: {url}")
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_bytes(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    pins = json.loads((Path(__file__).parent / "source-pins.json").read_text())
    configs = []
    for environment in pins["environments"]:
        config = environment["config"]
        root = output / "sources" / config["package"]
        for file in environment["files"]:
            fetch(root / file["path"], file["url"], file["sha256"])
        config["root"] = str(root)
        if environment.get("bucket"):
            bucket = environment["bucket"]
            pack = output / "portsim-train"
            fetch(pack / "tasks.jsonl.gz", bucket["url"], bucket["compressed_sha256"])
            body = gzip.decompress((pack / "tasks.jsonl.gz").read_bytes())
            if hashlib.sha256(body).hexdigest() != bucket["uncompressed_sha256"]:
                raise ValueError("PortSim train content does not match pinned manifest")
            (pack / "manifest.json").write_bytes(
                (root / "tasks/dock-v1-train/manifest.json").read_bytes()
            )
            config["environment"] = {"BERTH_TASKS_DIR": str(pack)}
        configs.append(config)
    output.mkdir(exist_ok=True, parents=True)
    (output / "config.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(output / "config.json")


if __name__ == "__main__":
    main()
