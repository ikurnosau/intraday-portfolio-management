from typing import BinaryIO

from config.settings import B2Settings


class B2ObjectStore:
    """Backblaze B2 object storage accessed through its S3-compatible API."""

    def __init__(self, client, bucket: str, key_prefix: str = "alpaca"):
        self.client = client
        self.bucket = bucket
        self.key_prefix = key_prefix.strip("/")

    @classmethod
    def from_settings(cls, settings: B2Settings) -> "B2ObjectStore":
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
            key_prefix=settings.key_prefix,
        )

    def _full_key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.key_prefix}/{key}" if self.key_prefix else key

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
