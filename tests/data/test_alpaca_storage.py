from io import BytesIO
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from data.object_store import B2ObjectStore
from data.raw.retrievers.alpaca_markets_retriever import AlpacaMarketsRetriever
from modeling.model_package import build_dataset_reference


class FakeClientError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {
            "ContentLength": len(self.objects[(Bucket, Key)]),
            "ETag": '"test-etag"',
            "VersionId": "test-version",
        }

    def upload_fileobj(self, file_object, bucket, key):
        self.objects[(bucket, key)] = file_object.read()

    def download_fileobj(
        self,
        bucket,
        key,
        file_object,
        Callback=None,
    ):
        payload = self.objects[(bucket, key)]
        file_object.write(payload)
        if Callback is not None:
            Callback(len(payload))


def test_b2_object_store_prefix_and_round_trip():
    client = FakeS3Client()
    store = B2ObjectStore(client, bucket="market-data", key_prefix="alpaca")

    assert not store.exists("bars/sample.pkl")

    store.upload_fileobj("bars/sample.pkl", BytesIO(b"payload"))

    output = BytesIO()
    store.download_fileobj("bars/sample.pkl", output)
    assert output.getvalue() == b"payload"
    assert store.exists("bars/sample.pkl")
    assert ("market-data", "alpaca/bars/sample.pkl") in client.objects
    assert store.uri_for_key("bars/sample.pkl") == (
        "s3://market-data/alpaca/bars/sample.pkl"
    )
    assert store.key_from_uri(
        "s3://market-data/alpaca/bars/sample.pkl"
    ) == "bars/sample.pkl"
    assert store.metadata("bars/sample.pkl").version_id == "test-version"


def test_retriever_pickles_through_object_store():
    store = B2ObjectStore(
        FakeS3Client(),
        bucket="market-data",
        key_prefix="alpaca",
    )
    retriever = AlpacaMarketsRetriever(object_store=store)
    payload = {"AAPL": {"close": [100.0, 101.0]}}

    retriever.save_data(payload, "bars", "sample.pkl")

    assert retriever.cache_exists("bars", "sample.pkl")
    assert retriever.load_data("bars", "sample.pkl") == payload


def test_dataset_reference_fingerprint_is_deterministic():
    store = B2ObjectStore(
        FakeS3Client(),
        bucket="market-data",
        key_prefix="alpaca",
    )
    store.upload_fileobj("bars/sample.pkl", BytesIO(b"payload"))

    first = build_dataset_reference(
        store,
        "bars/sample.pkl",
        {"symbols": ["AAPL"], "frequency": "1Min"},
    )
    second = build_dataset_reference(
        store,
        "bars/sample.pkl",
        {"frequency": "1Min", "symbols": ["AAPL"]},
    )

    assert first.fingerprint == second.fingerprint
    assert first.version_id == "test-version"


def test_quote_estimation_accepts_fixed_offset_endpoints():
    quote = SimpleNamespace(
        ask_price=101.0,
        ask_size=10,
        bid_price=100.0,
        bid_size=12,
    )

    class QuoteRetriever(AlpacaMarketsRetriever):
        def quotes(self, *args, **kwargs):
            return {"AAPL": [quote]}

    retriever = QuoteRetriever()

    result = retriever._quote_estimation(
        "AAPL",
        start=datetime.fromisoformat("2024-11-01T00:00:00-04:00"),
        end=datetime.fromisoformat("2026-08-01T00:00:00-05:00"),
    )

    assert result == {
        "ask_price": 101.0,
        "ask_size": 10,
        "bid_price": 100.0,
        "bid_size": 12,
    }


def test_exact_quotes_use_last_quote_before_each_bar_end():
    pages = {
        None: {
            "quotes": {
                "AAPL": [
                    {
                        "t": "2026-01-05T14:30:10Z",
                        "ap": 101.0,
                        "as": 10,
                        "bp": 100.0,
                        "bs": 11,
                    },
                    {
                        "t": "2026-01-05T14:30:59.367039499Z",
                        "ap": 102.0,
                        "as": 12,
                        "bp": 101.0,
                        "bs": 13,
                    },
                ]
            },
            "next_page_token": "page-2",
        },
        "page-2": {
            "quotes": {
                "AAPL": [
                    {
                        "t": "2026-01-05T14:31:00Z",
                        "ap": 103.0,
                        "as": 14,
                        "bp": 102.0,
                        "bs": 15,
                    }
                ]
            },
            "next_page_token": None,
        },
    }

    class RawQuoteClient:
        def __init__(self):
            self.requests = []

        def get(self, path, data):
            self.requests.append((path, data.copy()))
            if data["sort"] == "desc":
                return {"quotes": {"AAPL": []}}
            return pages[data["page_token"]]

    class QuoteRetriever(AlpacaMarketsRetriever):
        def _get_exact_quote_client(self):
            return raw_client

    raw_client = RawQuoteClient()
    retriever = QuoteRetriever(
        use_quote_estimation=False,
        exact_quote_workers=1,
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-05T09:30:00-05:00",
                    "2026-01-05T09:31:00-05:00",
                    "2026-01-05T09:32:00-05:00",
                ]
            ),
            "close": [100.5, 102.5, 102.75],
        }
    )

    result = retriever._add_exact_quotes("AAPL", bars)

    assert result[["ask_price", "ask_size", "bid_price", "bid_size"]].to_dict(
        "records"
    ) == [
        {
            "ask_price": 102.0,
            "ask_size": 12.0,
            "bid_price": 101.0,
            "bid_size": 13.0,
        },
        {
            "ask_price": 103.0,
            "ask_size": 14.0,
            "bid_price": 102.0,
            "bid_size": 15.0,
        },
        {
            "ask_price": 103.0,
            "ask_size": 14.0,
            "bid_price": 102.0,
            "bid_size": 15.0,
        },
    ]
    assert result["quote_timestamp"].tolist() == [
        pd.Timestamp("2026-01-05T09:30:59.367039499-05:00"),
        pd.Timestamp("2026-01-05T09:31:00-05:00"),
        pd.Timestamp("2026-01-05T09:31:00-05:00"),
    ]
    pagination_requests = [
        request
        for request in raw_client.requests
        if request[1]["sort"] == "asc"
    ]
    assert [request[1]["page_token"] for request in pagination_requests] == [
        None,
        "page-2",
    ]
    assert all(
        request[0] == "/stocks/quotes" and request[1]["limit"] == 10_000
        for request in pagination_requests
    )


def test_exact_quotes_seed_first_bar_from_prior_quote():
    class RawQuoteClient:
        def get(self, path, data):
            if data["sort"] == "desc":
                return {
                    "quotes": {
                        "AAPL": [
                            {
                                "t": "2026-01-05T14:29:50Z",
                                "ap": 100.0,
                                "as": 10,
                                "bp": 99.0,
                                "bs": 11,
                            }
                        ]
                    }
                }
            return {
                "quotes": {
                    "AAPL": [
                        {
                            "t": "2026-01-05T14:31:10Z",
                            "ap": 102.0,
                            "as": 12,
                            "bp": 101.0,
                            "bs": 13,
                        }
                    ]
                },
                "next_page_token": None,
            }

    class QuoteRetriever(AlpacaMarketsRetriever):
        def _get_exact_quote_client(self):
            return raw_client

    raw_client = RawQuoteClient()
    retriever = QuoteRetriever(
        use_quote_estimation=False,
        exact_quote_workers=1,
    )
    bars = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-01-05T09:30:00-05:00",
                    "2026-01-05T09:31:00-05:00",
                ]
            )
        }
    )

    result = retriever._add_exact_quotes("AAPL", bars)

    assert result["ask_price"].tolist() == [100.0, 102.0]
    assert result["quote_timestamp"].tolist() == [
        pd.Timestamp("2026-01-05T09:29:50-05:00"),
        pd.Timestamp("2026-01-05T09:31:10-05:00"),
    ]


def test_quote_modes_have_distinct_cache_file_names():
    estimated = AlpacaMarketsRetriever(use_quote_estimation=True)
    exact = AlpacaMarketsRetriever(use_quote_estimation=False)
    start = datetime.fromisoformat("2026-01-05T09:30:00-05:00")
    end = datetime.fromisoformat("2026-01-05T10:00:00-05:00")

    estimated_name = estimated.build_file_name("AAPL", start, end)
    exact_name = exact.build_file_name("AAPL", start, end)

    assert estimated_name != exact_name
    assert estimated_name.endswith("_AAPL.pkl")
    assert exact_name.endswith("_exact-quotes-asof-with-timestamp.pkl")
    assert (
        estimated._build_bars_file_name("AAPL", start, end)
        == exact._build_bars_file_name("AAPL", start, end)
    )


def test_has_bars_checks_requested_history_window():
    class HistoryClient:
        def __init__(self):
            self.requests = []

        def get_stock_bars(self, request):
            self.requests.append(request)
            data = {"AAPL": [SimpleNamespace()]} if request.symbol_or_symbols == "AAPL" else {}
            return SimpleNamespace(data=data)

    client = HistoryClient()
    retriever = AlpacaMarketsRetriever()
    retriever._client = client
    start = datetime.fromisoformat("2024-10-01T00:00:00-04:00")
    end = datetime.fromisoformat("2024-11-01T00:00:00-04:00")

    assert retriever.has_bars("AAPL", start, end)
    assert not retriever.has_bars("DRAM", start, end)
    expected_start = start.astimezone(timezone.utc).replace(tzinfo=None)
    expected_end = end.astimezone(timezone.utc).replace(tzinfo=None)
    assert all(
        request.start == expected_start and request.end == expected_end
        for request in client.requests
    )
