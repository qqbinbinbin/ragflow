#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import os
import time
from common.decorator import singleton
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ClientSecretCredential, AzureAuthorityHosts
from azure.storage.filedatalake import FileSystemClient
from common import settings

_CLOUD_AUTHORITY_MAP = {
    "public": AzureAuthorityHosts.AZURE_PUBLIC_CLOUD,
    "china": AzureAuthorityHosts.AZURE_CHINA,
    "government": AzureAuthorityHosts.AZURE_GOVERNMENT,
    "germany": AzureAuthorityHosts.AZURE_GERMANY,
}


@singleton
class RAGFlowAzureSpnBlob:
    def __init__(self):
        self.conn = None
        self.account_url = os.getenv("ACCOUNT_URL", settings.AZURE["account_url"])
        self.client_id = os.getenv("CLIENT_ID", settings.AZURE["client_id"])
        self.secret = os.getenv("SECRET", settings.AZURE["secret"])
        self.tenant_id = os.getenv("TENANT_ID", settings.AZURE["tenant_id"])
        self.container_name = os.getenv("CONTAINER_NAME", settings.AZURE["container_name"])
        self.cloud = os.getenv("AZURE_CLOUD", settings.AZURE.get("cloud", "public")).lower()
        self.__open__()

    def __open__(self):
        try:
            if self.conn:
                self.__close__()
        except Exception:
            pass

        try:
            authority = _CLOUD_AUTHORITY_MAP.get(self.cloud, AzureAuthorityHosts.AZURE_PUBLIC_CLOUD)
            credentials = ClientSecretCredential(tenant_id=self.tenant_id, client_id=self.client_id, client_secret=self.secret, authority=authority)
            self.conn = FileSystemClient(account_url=self.account_url, file_system_name=self.container_name, credential=credentials)
        except Exception:
            logging.exception("Fail to connect %s" % self.account_url)

    def __close__(self):
        del self.conn
        self.conn = None

    def health(self):
        _bucket, fnm, binary = "txtxtxtxt1", "txtxtxtxt1", b"_t@@@1"
        f = self.conn.create_file(f"{_bucket}/{fnm}")
        f.append_data(binary, offset=0, length=len(binary))
        return f.flush_data(len(binary))

    def put(self, bucket, fnm, binary, tenant_id=None):
        blob = f"{bucket}/{fnm}"
        for _ in range(3):
            try:
                f = self.conn.create_file(f"{blob}")
                f.append_data(binary, offset=0, length=len(binary))
                return f.flush_data(len(binary))
            except Exception:
                logging.exception(f"Fail put {blob}")
                self.__open__()
                time.sleep(1)
                return None
        return None

    def rm(self, bucket, fnm, tenant_id=None):
        blob = f"{bucket}/{fnm}"
        try:
            self.conn.delete_file(f"{blob}")
        except Exception:
            logging.exception(f"Fail rm {blob}")

    def rm_strict(self, bucket, fnm, tenant_id=None):
        blob = f"{bucket}/{fnm}"
        try:
            self.conn.delete_file(blob)
        except ResourceNotFoundError:
            pass
        if self.conn.get_file_client(blob).exists():
            raise OSError("strict object deletion was incomplete")

    def _list_prefix_strict(self, prefix):
        list_path = prefix.rstrip("/")
        try:
            return [
                path.name
                for path in self.conn.get_paths(path=list_path, recursive=True)
                if not path.is_directory and path.name.startswith(prefix)
            ]
        except ResourceNotFoundError:
            return []

    def rm_prefix_strict(self, bucket, prefix, tenant_id=None):
        blob_prefix = f"{bucket}/{prefix}"
        object_names = self._list_prefix_strict(blob_prefix)
        for object_name in object_names:
            try:
                self.conn.delete_file(object_name)
            except ResourceNotFoundError:
                pass
        if self._list_prefix_strict(blob_prefix):
            raise OSError("strict object prefix deletion was incomplete")
        return len(object_names)

    def get(self, bucket, fnm, tenant_id=None):
        blob = f"{bucket}/{fnm}"
        for _ in range(1):
            try:
                client = self.conn.get_file_client(f"{blob}")
                r = client.download_file()
                return r.read()
            except Exception:
                logging.exception(f"fail get {blob}")
                self.__open__()
                time.sleep(1)
        return None

    def obj_exist(self, bucket, fnm, tenant_id=None):
        blob = f"{bucket}/{fnm}"
        try:
            client = self.conn.get_blob_client(f"{blob}")
            return client.exists()
        except Exception:
            logging.exception(f"Fail put {blob}")
        return False

    def obj_exist_strict(self, bucket, fnm, tenant_id=None):
        blob = f"{bucket}/{fnm}"
        return self.conn.get_file_client(blob).exists()

    def get_presigned_url(self, bucket, fnm, expires):
        f_path = f"{bucket}/{fnm}"
        for _ in range(10):
            try:
                return self.conn.get_presigned_url("GET", bucket, f_path, expires)
            except Exception:
                logging.exception(f"fail get {bucket}/{fnm}")
                self.__open__()
                time.sleep(1)
        return None
