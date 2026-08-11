from io import BytesIO
from datetime import datetime, timezone
from types import SimpleNamespace

from data.object_store import B2ObjectStore
from data.raw.retrievers.alpaca_markets_retriever import AlpacaMarketsRetriever


class FakeClientError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise FakeClientError("404")
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

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
