"""WhatsApp channel implementation using Node.js bridge."""

import asyncio
import json
import time
from collections import deque
from typing import Any

from loguru import logger

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import WhatsAppConfig


class WhatsAppChannel(BaseChannel):
    """
    WhatsApp channel that connects to a Node.js bridge.
    
    The bridge uses @whiskeysockets/baileys to handle the WhatsApp Web protocol.
    Communication between Python and Node.js is via WebSocket.
    """
    
    name = "whatsapp"
    
    def __init__(self, config: WhatsAppConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: WhatsAppConfig = config
        self._ws = None
        self._connected = False
        self._seen_message_ids: deque[str] = deque()
        self._seen_message_ids_set: set[str] = set()
        self._seen_message_ids_limit = 200
    
    async def start(self) -> None:
        """Start the WhatsApp channel by connecting to the bridge."""
        import websockets
        
        bridge_url = self.config.bridge_url
        
        logger.info(f"Connecting to WhatsApp bridge at {bridge_url}...")
        
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(bridge_url) as ws:
                    self._ws = ws
                    self._connected = True
                    logger.info("Connected to WhatsApp bridge")
                    
                    # Listen for messages
                    async for message in ws:
                        try:
                            await self._handle_bridge_message(message)
                        except Exception as e:
                            logger.error(f"Error handling bridge message: {e}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                self._ws = None
                logger.warning(f"WhatsApp bridge connection error: {e}")
                
                if self._running:
                    logger.info("Reconnecting in 5 seconds...")
                    await asyncio.sleep(5)
    
    async def stop(self) -> None:
        """Stop the WhatsApp channel."""
        self._running = False
        self._connected = False
        
        if self._ws:
            await self._ws.close()
            self._ws = None
    
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through WhatsApp."""
        if not self._ws or not self._connected:
            logger.warning("WhatsApp bridge not connected")
            return
        
        try:
            presence = msg.metadata.get("presence") if msg.metadata else None
            if presence:
                payload = {
                    "type": "presence",
                    "to": msg.chat_id,
                    "presence": presence,
                }
                await self._ws.send(json.dumps(payload))
                ready_perf = (msg.metadata or {}).get("_nb_response_ready_perf")
                if isinstance(ready_perf, (int, float)):
                    send_lag_ms = int(round((time.perf_counter() - float(ready_perf)) * 1000))
                    logger.info(
                        "WhatsApp presence sent in {}ms after response-ready (chat_id={})",
                        send_lag_ms,
                        msg.chat_id,
                    )
                return

            if not msg.content:
                return

            payload = {
                "type": "send",
                "to": msg.chat_id,
                "text": msg.content
            }
            await self._ws.send(json.dumps(payload))
            ready_perf = (msg.metadata or {}).get("_nb_response_ready_perf")
            if isinstance(ready_perf, (int, float)):
                send_lag_ms = int(round((time.perf_counter() - float(ready_perf)) * 1000))
                logger.info(
                    "WhatsApp send in {}ms after response-ready (chat_id={}, chars={})",
                    send_lag_ms,
                    msg.chat_id,
                    len(msg.content or ""),
                )
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {e}")
    
    async def _handle_bridge_message(self, raw: str) -> None:
        """Handle a message from the bridge."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from bridge: {raw[:100]}")
            return
        
        msg_type = data.get("type")
        
        if msg_type == "message":
            # Incoming message from WhatsApp
            # Deprecated by whatsapp: old phone number style typically: <phone>@s.whatspp.net
            pn = data.get("pn", "")
            # New LID sytle typically: 
            sender = data.get("sender", "")
            content = data.get("content", "")
            media_path = data.get("mediaPath", "")
            message_id = data.get("id")

            if message_id:
                if message_id in self._seen_message_ids_set:
                    logger.debug(f"Duplicate WhatsApp message ignored: {message_id}")
                    return
                self._seen_message_ids.append(message_id)
                self._seen_message_ids_set.add(message_id)
                while len(self._seen_message_ids) > self._seen_message_ids_limit:
                    dropped = self._seen_message_ids.popleft()
                    self._seen_message_ids_set.discard(dropped)
            
            # Prefer phone-number JID (remoteJidAlt) for replies when available.
            user_id = pn if pn else sender
            sender_id = user_id.split("@")[0] if "@" in user_id else user_id
            logger.info(f"Sender {sender}")
            
            # Handle voice transcription if it's a voice message
            if content == "[Voice Message]":
                logger.info(f"Voice message received from {sender_id}, but direct download from bridge is not yet supported.")
                content = "[Voice Message: Transcription not available for WhatsApp yet]"

            media: list[str] = []
            if isinstance(media_path, str) and media_path.strip():
                media.append(media_path.strip())
            
            await self._handle_message(
                sender_id=sender_id,
                chat_id=user_id,
                content=content,
                media=media,
                metadata={
                    "message_id": message_id,
                    "timestamp": data.get("timestamp"),
                    "is_group": data.get("isGroup", False),
                    "media_type": data.get("mediaType"),
                }
            )
        
        elif msg_type == "status":
            # Connection status update
            status = data.get("status")
            logger.info(f"WhatsApp status: {status}")
            
            if status == "connected":
                self._connected = True
            elif status == "disconnected":
                self._connected = False
        
        elif msg_type == "qr":
            # QR code for authentication
            logger.info("Scan QR code in the bridge terminal to connect WhatsApp")
        
        elif msg_type == "error":
            logger.error(f"WhatsApp bridge error: {data.get('error')}")
