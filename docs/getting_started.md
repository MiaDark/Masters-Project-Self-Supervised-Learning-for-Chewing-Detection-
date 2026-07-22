# Getting started

## 1. Install

```bash
git clone <your-repo-url> chewing-ssl && cd chewing-ssl
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # package + lint/test/security tooling
```

For deep-learning work, install the `torch` build that matches your hardware
from [pytorch.org](https://pytorch.org/get-started/locally/) rather than the
generic pin.

## 2. Configure credentials

```bash
cp .env.example .env             # then edit it
aws configure                    # stores AWS keys outside the repo
```

## 3. Download data

```bash
python -m chewing_ssl.common.data.download --list      # see participant folders
python -m chewing_ssl.common.data.download --dry-run    # preview files + total size
python -m chewing_ssl.common.data.download              # download into data/raw/
```

See **[Data](data.md)** for what you are actually pulling, and
**[Architecture](architecture.md)** for where it goes next.

## 4. Run the checks

```bash
make lint && make coverage && make security
pre-commit install               # run the same checks at commit time
```