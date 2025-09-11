from __future__ import annotations

import json
import logging
from typing import Union, Dict

import websockets

logger = logging.getLogger(__name__)


def connect(base_url: str, **kwargs):
    return WebSocket(base_url, **kwargs)


class WebSocket:

    def __init__(self, base_url: str, **kwargs) -> None:
        """
        Parameters
        ----------
        base_url: str
        """
        self.url = base_url
        # id for messages
        self.id = 0
        self._kwargs = kwargs

    async def __aenter__(self) -> WebSocket:
        await self._connect()
        return self

    async def __aexit__(self, *args, **kwargs) -> None:
        await self._conn.__aexit__(*args, **kwargs)
        logger.info(f"Closed connection with {self.url}")

    def __aiter__(self) -> WebSocket:
        return self

    async def __anext__(self) -> dict:
        return await self.recv()

    async def send(self, message: dict) -> None:
        """
        Send a message to WebSocket server.
        """
        self.id += 1
        await self._ws.send(json.dumps(message))

    async def recv(self) -> dict:
        """
        Await a message from WebSocket server.
        """
        message = await self._ws.recv()
        return self._decode(message)

    async def _connect(self) -> None:
        """
        Connect to WebSocket server.
        """
        self._conn = websockets.connect(self.url, **self._kwargs)
        self._ws = await self._conn.__aenter__()
        logger.info(f"Established connection with {self.url}")

    @staticmethod
    def _decode(message: Union[str, bytes, bytearray]) -> Dict:
        """
        Decode message from WebSocket server.
        """
        assert isinstance(message, str)
        return json.loads(message)
