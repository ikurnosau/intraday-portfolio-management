from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse

from config.settings import B2Settings


@dataclass(frozen=True)
class ObjectMetadata:
    uri: str
    size: int
    etag: str | None
    version_id: str | None


class B2ObjectStore:
    """Backblaze B2 object storage accessed through its S3-compatible API."""

    def __init__(self, client, bucket: str, key_prefix: str = "alpaca"):
        self.client = client
        self.bucket = bucket
        self.key_prefix = key_prefix.strip("/")

    @classmethod
    def from_settings(
        cls,
        settings: B2Settings,
        *,
        key_prefix: str | None = None,
    ) -> "B2ObjectStore":
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            region_name=settings.region,
            aws_access_key_id=settings.access_key_id,
            aws_secret_access_key=settings.secret_access_key,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        return cls(
            client=client,
            bucket=settings.bucket_name,
            key_prefix=settings.key_prefix if key_prefix is None else key_prefix,
        )

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.key_prefix}/{key}" if self.key_prefix else key

    def uri_for_key(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._full_key(key)}"

    def key_from_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or parsed.netloc != self.bucket:
            raise ValueError(f"URI does not belong to configured object store: {uri}")

        full_key = parsed.path.lstrip("/").rstrip("/")
        if self.key_prefix:
            prefix = f"{self.key_prefix}/"
            if not full_key.startswith(prefix):
                raise ValueError(
                    f"URI is outside configured key prefix '{self.key_prefix}': {uri}"
                )
            return full_key[len(prefix):]
        return full_key

    def metadata(self, key: str) -> ObjectMetadata:
        response = self.client.head_object(
            Bucket=self.bucket,
            Key=self._full_key(key),
        )
        etag = response.get("ETag")
        return ObjectMetadata(
            uri=self.uri_for_key(key),
            size=int(response["ContentLength"]),
            etag=etag.strip('"') if isinstance(etag, str) else None,
            version_id=response.get("VersionId"),
        )

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._full_key(key))
            return True
        except Exception as error:
            error_code = str(
                getattr(error, "response", {}).get("Error", {}).get("Code", "")
            )
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def upload_fileobj(self, key: str, file_object: BinaryIO) -> None:
        self.client.upload_fileobj(
            file_object,
            self.bucket,
            self._full_key(key),
        )

    def upload_file(self, key: str, path: str | Path) -> None:
        with Path(path).open("rb") as file_object:
            self.upload_fileobj(key, file_object)

    def download_fileobj(
        self,
        key: str,
        file_object: BinaryIO,
    ) -> None:
        full_key = self._full_key(key)
        from tqdm.auto import tqdm

        metadata = self.client.head_object(
            Bucket=self.bucket,
            Key=full_key,
        )
        with tqdm(
            total=metadata["ContentLength"],
            desc=f"Downloading {key.rsplit('/', 1)[-1]}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            self.client.download_fileobj(
                self.bucket,
                full_key,
                file_object,
                Callback=progress.update,
            )

    def download_file(self, key: str, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file_object:
            self.download_fileobj(key, file_object)
