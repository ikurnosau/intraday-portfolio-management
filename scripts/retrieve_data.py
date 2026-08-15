import logging
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from config.train_config import load_train_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = load_train_config()
    retriever = config.data_config.retriever
    retriever.bars_with_quotes(
        symbol_or_symbols=config.data_config.symbol_or_symbols,
        start=config.data_config.start,
        end=config.data_config.end,
    )


if __name__ == "__main__":
    main()
