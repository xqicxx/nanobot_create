/**
 * WhatsApp client wrapper using Baileys.
 * Based on OpenClaw's working implementation.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
  downloadMediaMessage,
} from '@whiskeysockets/baileys';

import { Boom } from '@hapi/boom';
import { mkdir, writeFile } from 'fs/promises';
import { homedir } from 'os';
import { join } from 'path';
import qrcode from 'qrcode-terminal';
import pino from 'pino';

const VERSION = '0.1.0';

const WA_MARK_ONLINE = true;
const WA_AUTO_READ = true;
const WA_DEBUG = /^(1|true|yes)$/i.test(process.env.WA_DEBUG ?? '');

export interface InboundMessage {
  id: string;
  sender: string;
  pn: string;
  content: string;
  timestamp: number;
  isGroup: boolean;
  mediaPath?: string;
  mediaType?: 'image' | 'video' | 'document' | 'audio';
}

export interface WhatsAppClientOptions {
  authDir: string;
  onMessage: (msg: InboundMessage) => void;
  onQR: (qr: string) => void;
  onStatus: (status: string) => void;
}

export class WhatsAppClient {
  private sock: any = null;
  private options: WhatsAppClientOptions;
  private reconnecting = false;
  private presenceInterval: NodeJS.Timeout | null = null;
  private jidMap: Map<string, string> = new Map();
  private readonly mediaDir: string;
  private readonly mediaLogger = pino({ level: 'silent' });

  constructor(options: WhatsAppClientOptions) {
    this.options = options;
    this.mediaDir = process.env.WA_MEDIA_DIR || join(homedir(), '.nanobot', 'media', 'whatsapp');
  }

  private rememberJidPair(primary: unknown, alt: unknown): void {
    if (typeof primary !== 'string' || typeof alt !== 'string') return;
    if (!primary || !alt) return;
    this.jidMap.set(primary, alt);
    this.jidMap.set(alt, primary);
  }

  private altFor(jid: string): string | null {
    const alt = this.jidMap.get(jid);
    return alt && alt !== jid ? alt : null;
  }

  private async sendPresenceAvailable(jid?: string): Promise<void> {
    if (!WA_MARK_ONLINE) return;
    if (!this.sock?.sendPresenceUpdate) return;
    try {
      // Presence updates are often jid-scoped; providing a jid makes "online" more reliable.
      await this.sock.sendPresenceUpdate('available', jid);
    } catch (err) {
      if (WA_DEBUG) console.error('sendPresenceUpdate failed:', err);
    }
  }

  private async maybeAutoRead(msg: any): Promise<void> {
    if (!WA_AUTO_READ) return;
    if (!this.sock) return;

    const remoteJid: string | undefined = typeof msg?.key?.remoteJid === 'string' ? msg.key.remoteJid : undefined;
    const remoteJidAlt: string | undefined =
      typeof msg?.key?.remoteJidAlt === 'string' ? msg.key.remoteJidAlt : undefined;
    const receiptJid = remoteJidAlt || remoteJid;
    const ids = msg?.key?.id ? [msg.key.id] : [];

    try {
      if (typeof this.sock.readMessages === 'function') {
        await this.sock.readMessages([msg.key]);
      }
    } catch (err) {
      if (WA_DEBUG) console.error('readMessages failed:', err);
    }

    if (!receiptJid || ids.length === 0) return;

    try {
      if (typeof this.sock.sendReadReceipt === 'function') {
        await this.sock.sendReadReceipt(receiptJid, msg?.key?.participant, ids);
      }
    } catch (err) {
      if (WA_DEBUG) console.error('sendReadReceipt failed:', err);
    }

    // Some setups only accept the other jid format; try both when we have them.
    if (remoteJid && remoteJidAlt && remoteJid !== remoteJidAlt) {
      try {
        if (typeof this.sock.sendReadReceipt === 'function') {
          await this.sock.sendReadReceipt(remoteJid, msg?.key?.participant, ids);
        }
      } catch {
        // Ignore
      }
    }
  }

  async connect(): Promise<void> {
    const logger = pino({ level: 'silent' });
    const { state, saveCreds } = await useMultiFileAuthState(this.options.authDir);
    const { version } = await fetchLatestBaileysVersion();

    console.log(`Using Baileys version: ${version.join('.')}`);
    console.log(
      `Bridge options: WA_MARK_ONLINE=${WA_MARK_ONLINE ? '1' : '0'} WA_AUTO_READ=${WA_AUTO_READ ? '1' : '0'}`,
    );

    // Create socket following OpenClaw's pattern
    this.sock = makeWASocket({
      auth: {
        creds: state.creds,
        keys: makeCacheableSignalKeyStore(state.keys, logger),
      },
      version,
      logger,
      printQRInTerminal: false,
      browser: ['nanobot', 'cli', VERSION],
      syncFullHistory: false,
      markOnlineOnConnect: WA_MARK_ONLINE,
    });

    // Handle WebSocket errors
    if (this.sock.ws && typeof this.sock.ws.on === 'function') {
      this.sock.ws.on('error', (err: Error) => {
        console.error('WebSocket error:', err.message);
      });
    }

    // Handle connection updates
    this.sock.ev.on('connection.update', async (update: any) => {
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        // Display QR code in terminal
        console.log('\n📱 Scan this QR code with WhatsApp (Linked Devices):\n');
        qrcode.generate(qr, { small: true });
        this.options.onQR(qr);
      }

      if (connection === 'close') {
        const statusCode = (lastDisconnect?.error as Boom)?.output?.statusCode;
        const shouldReconnect = statusCode !== DisconnectReason.loggedOut;

        console.log(`Connection closed. Status: ${statusCode}, Will reconnect: ${shouldReconnect}`);
        this.options.onStatus('disconnected');
        if (this.presenceInterval) {
          clearInterval(this.presenceInterval);
          this.presenceInterval = null;
        }

        if (shouldReconnect && !this.reconnecting) {
          this.reconnecting = true;
          console.log('Reconnecting in 5 seconds...');
          setTimeout(() => {
            this.reconnecting = false;
            this.connect();
          }, 5000);
        }
      } else if (connection === 'open') {
        console.log('✅ Connected to WhatsApp');
        this.options.onStatus('connected');

        if (WA_MARK_ONLINE) {
          await this.sendPresenceAvailable();
          // Keep the "online" presence fresh. Without periodic updates, WhatsApp may not
          // reflect online/last-seen reliably for linked-device clients.
          this.presenceInterval = setInterval(() => {
            void this.sendPresenceAvailable();
          }, 45_000);
          this.presenceInterval.unref?.();
        }
      }
    });

    // Save credentials on update
    this.sock.ev.on('creds.update', saveCreds);

    // Handle incoming messages
    this.sock.ev.on('messages.upsert', async ({ messages, type }: { messages: any[]; type: string }) => {
      if (type !== 'notify') return;

      for (const msg of messages) {
        // Skip own messages
        if (msg.key.fromMe) continue;

        // Skip status updates
        if (msg.key.remoteJid === 'status@broadcast') continue;

        // Track mappings between LID JIDs and phone-number JIDs (remoteJidAlt) when present.
        this.rememberJidPair(msg?.key?.remoteJid, msg?.key?.remoteJidAlt);

        const presenceJid =
          typeof msg?.key?.remoteJidAlt === 'string' && msg.key.remoteJidAlt ? msg.key.remoteJidAlt : msg.key.remoteJid;
        await this.sendPresenceAvailable(presenceJid);
        await this.maybeAutoRead(msg);

        const parsed = await this.extractIncomingMessage(msg);
        if (!parsed.content) continue;

        const isGroup = msg.key.remoteJid?.endsWith('@g.us') || false;

        this.options.onMessage({
          id: msg.key.id || '',
          sender: msg.key.remoteJid || '',
          pn: msg.key.remoteJidAlt || '',
          content: parsed.content,
          timestamp: msg.messageTimestamp as number,
          isGroup,
          mediaPath: parsed.mediaPath,
          mediaType: parsed.mediaType,
        });
      }
    });
  }

  private async extractIncomingMessage(msg: any): Promise<{
    content: string | null;
    mediaPath?: string;
    mediaType?: 'image' | 'video' | 'document' | 'audio';
  }> {
    const message = this.unwrapMessageContainer(msg.message);
    if (!message) return { content: null };

    // Text message
    if (message.conversation) {
      return { content: message.conversation };
    }

    // Extended text (reply, link preview)
    if (message.extendedTextMessage?.text) {
      return { content: message.extendedTextMessage.text };
    }

    if (message.imageMessage) {
      const mediaPath = await this.saveMediaMessage(msg, 'image', message.imageMessage.mimetype);
      const caption = message.imageMessage.caption?.trim();
      return {
        content: this.formatMediaContent('Image', mediaPath, caption),
        mediaPath: mediaPath || undefined,
        mediaType: mediaPath ? 'image' : undefined,
      };
    }

    if (message.videoMessage) {
      const mediaPath = await this.saveMediaMessage(msg, 'video', message.videoMessage.mimetype);
      const caption = message.videoMessage.caption?.trim();
      return {
        content: this.formatMediaContent('Video', mediaPath, caption),
        mediaPath: mediaPath || undefined,
        mediaType: mediaPath ? 'video' : undefined,
      };
    }

    if (message.documentMessage) {
      const mediaPath = await this.saveMediaMessage(msg, 'document', message.documentMessage.mimetype);
      const caption = message.documentMessage.caption?.trim() || message.documentMessage.fileName;
      return {
        content: this.formatMediaContent('Document', mediaPath, caption),
        mediaPath: mediaPath || undefined,
        mediaType: mediaPath ? 'document' : undefined,
      };
    }

    if (message.audioMessage) {
      const mediaPath = await this.saveMediaMessage(msg, 'audio', message.audioMessage.mimetype);
      const label = message.audioMessage.ptt ? 'Voice Message' : 'Audio';
      return {
        content: this.formatMediaContent(label, mediaPath),
        mediaPath: mediaPath || undefined,
        mediaType: mediaPath ? 'audio' : undefined,
      };
    }

    return { content: null };
  }

  private unwrapMessageContainer(message: any): any {
    let current = message;
    const visited = new Set<any>();
    while (current && typeof current === 'object' && !visited.has(current)) {
      visited.add(current);
      if (current.ephemeralMessage?.message) {
        current = current.ephemeralMessage.message;
        continue;
      }
      if (current.viewOnceMessage?.message) {
        current = current.viewOnceMessage.message;
        continue;
      }
      if (current.viewOnceMessageV2?.message) {
        current = current.viewOnceMessageV2.message;
        continue;
      }
      if (current.viewOnceMessageV2Extension?.message) {
        current = current.viewOnceMessageV2Extension.message;
        continue;
      }
      if (current.deviceSentMessage?.message) {
        current = current.deviceSentMessage.message;
        continue;
      }
      if (current.editedMessage?.message) {
        current = current.editedMessage.message;
        continue;
      }
      if (current.documentWithCaptionMessage?.message) {
        current = current.documentWithCaptionMessage.message;
        continue;
      }
      break;
    }
    return current;
  }

  private formatMediaContent(kind: string, mediaPath: string | null, caption?: string): string {
    const prefix = mediaPath ? `[${kind}: ${mediaPath}]` : `[${kind}]`;
    return caption ? `${prefix} ${caption}` : prefix;
  }

  private async saveMediaMessage(
    msg: any,
    mediaType: 'image' | 'video' | 'document' | 'audio',
    mimeType?: string,
  ): Promise<string | null> {
    if (!this.sock) return null;
    try {
      const raw = await downloadMediaMessage(
        msg,
        'buffer',
        {},
        {
          logger: this.mediaLogger,
          reuploadRequest: this.sock.updateMediaMessage,
        },
      );
      if (!raw) return null;
      const buffer = Buffer.isBuffer(raw) ? raw : Buffer.from(raw as Uint8Array);
      const maxSizeBytes = 20 * 1024 * 1024;
      if (buffer.length > maxSizeBytes) {
        if (WA_DEBUG) console.error(`media too large (${buffer.length} bytes), skip save`);
        return null;
      }
      await mkdir(this.mediaDir, { recursive: true });
      const ext = this.extensionForMime(mimeType, mediaType);
      const rawId = String(msg?.key?.id || Date.now());
      const safeId = rawId.replace(/[^a-zA-Z0-9_-]/g, '_');
      const filePath = join(this.mediaDir, `${safeId}_${Date.now()}${ext}`);
      await writeFile(filePath, buffer);
      return filePath;
    } catch (err) {
      console.error('saveMediaMessage failed:', err instanceof Error ? err.message : String(err));
      return null;
    }
  }

  private extensionForMime(mimeType: string | undefined, mediaType: 'image' | 'video' | 'document' | 'audio'): string {
    const mime = (mimeType || '').toLowerCase();
    if (mime.includes('jpeg') || mime.includes('jpg')) return '.jpg';
    if (mime.includes('png')) return '.png';
    if (mime.includes('webp')) return '.webp';
    if (mime.includes('gif')) return '.gif';
    if (mime.includes('mp4')) return '.mp4';
    if (mime.includes('mpeg')) return '.mp3';
    if (mime.includes('ogg')) return '.ogg';
    if (mime.includes('wav')) return '.wav';
    if (mime.includes('pdf')) return '.pdf';
    if (mime.includes('zip')) return '.zip';
    if (mime.includes('json')) return '.json';

    if (mediaType === 'image') return '.jpg';
    if (mediaType === 'video') return '.mp4';
    if (mediaType === 'audio') return '.ogg';
    return '.bin';
  }

  async sendMessage(to: string, text: string): Promise<void> {
    if (!this.sock) {
      throw new Error('Not connected');
    }

    const normalized = this.normalizeJid(to);
    const alt = this.altFor(normalized);
    const preferLid = alt && alt.endsWith('@lid') ? alt : normalized;
    const fallback = alt && alt !== preferLid ? alt : normalized !== preferLid ? normalized : null;

    // Try to look "online" to the recipient before sending.
    await this.sendPresenceAvailable(preferLid);
    if (fallback && fallback !== preferLid) {
      await this.sendPresenceAvailable(fallback);
    }

    try {
      await this.sock.sendMessage(preferLid, { text });
    } catch (err) {
      if (!fallback || fallback === preferLid) throw err;
      if (WA_DEBUG) console.error('sendMessage failed, retrying with fallback jid:', err);
      await this.sock.sendMessage(fallback, { text });
    }
  }

  private normalizeJid(to: string): string {
    if (!to) return to;
    if (to.includes('@')) return to;
    if (/^\d+$/.test(to)) {
      return `${to}@s.whatsapp.net`;
    }
    return to;
  }

  async sendPresence(to: string, presence: 'composing' | 'paused' | 'available'): Promise<void> {
    if (!this.sock?.sendPresenceUpdate) return;
    const targets = [to];
    const alt = this.altFor(to);
    if (alt) targets.push(alt);
    for (const jid of targets) {
      try {
        await this.sock.sendPresenceUpdate(presence, jid);
      } catch (err) {
        if (WA_DEBUG) console.error('sendPresenceUpdate failed:', err);
      }
    }
  }

  async disconnect(): Promise<void> {
    if (this.sock) {
      this.sock.end(undefined);
      this.sock = null;
    }
  }
}
