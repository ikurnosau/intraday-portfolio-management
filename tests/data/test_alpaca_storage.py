from io import BytesIO

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

    def upload_fileobj(self, file_object, bucket, key):
        self.objects[(bucket, key)] = file_object.read()

    def download_fileobj(self, bucket, key, file_object):
        file_object.write(self.objects[(bucket, key)])


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
