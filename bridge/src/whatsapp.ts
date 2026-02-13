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
} from '@whiskeysockets/baileys';

import { Boom } from '@hapi/boom';
import qrcode from 'qrcode-terminal';
import pino from 'pino';

const VERSION = '0.1.0';

const WA_MARK_ONLINE =
  process.env.WA_MARK_ONLINE === undefined
    ? true
    : /^(1|true|yes)$/i.test(process.env.WA_MARK_ONLINE);
const WA_AUTO_READ =
  process.env.WA_AUTO_READ === undefined
    ? true
    : /^(1|true|yes)$/i.test(process.env.WA_AUTO_READ);
const WA_DEBUG = /^(1|true|yes)$/i.test(process.env.WA_DEBUG ?? '');

export interface InboundMessage {
  id: string;
  sender: string;
  pn: string;
  content: string;
  timestamp: number;
  isGroup: boolean;
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

  constructor(options: WhatsAppClientOptions) {
    this.options = options;
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

        const content = this.extractMessageContent(msg);
        if (!content) continue;

        const isGroup = msg.key.remoteJid?.endsWith('@g.us') || false;

        this.options.onMessage({
          id: msg.key.id || '',
          sender: msg.key.remoteJid || '',
          pn: msg.key.remoteJidAlt || '',
          content,
          timestamp: msg.messageTimestamp as number,
          isGroup,
        });
      }
    });
  }

  private extractMessageContent(msg: any): string | null {
    const message = msg.message;
    if (!message) return null;

    // Text message
    if (message.conversation) {
      return message.conversation;
    }

    // Extended text (reply, link preview)
    if (message.extendedTextMessage?.text) {
      return message.extendedTextMessage.text;
    }

    // Image with caption
    if (message.imageMessage?.caption) {
      return `[Image] ${message.imageMessage.caption}`;
    }

    // Video with caption
    if (message.videoMessage?.caption) {
      return `[Video] ${message.videoMessage.caption}`;
    }

    // Document with caption
    if (message.documentMessage?.caption) {
      return `[Document] ${message.documentMessage.caption}`;
    }

    // Voice/Audio message
    if (message.audioMessage) {
      return `[Voice Message]`;
    }

    return null;
  }

  async sendMessage(to: string, text: string): Promise<void> {
    if (!this.sock) {
      throw new Error('Not connected');
    }

    // Try to look "online" to the recipient before sending.
    await this.sendPresenceAvailable(to);
    const alt = this.altFor(to);
    if (alt) {
      await this.sendPresenceAvailable(alt);
    }

    await this.sock.sendMessage(to, { text });
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
