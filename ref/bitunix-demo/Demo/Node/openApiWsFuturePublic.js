const WebSocket = require('ws');
const EventEmitter = require('events');

/**
 * OpenAPI WebSocket futures public API client
 */
class OpenApiWsFuturePublic extends EventEmitter {
    /**
     * initialize the WebSocket public API client
     * @param {Object} config  config object
     */
    constructor(config) {
        super();
        this.config = config;
        this.baseUrl = config.websocket.public_uri;
        this.reconnectInterval = config.websocket.reconnect_interval * 1000; // convert to milliseconds
        this.heartbeatInterval = 3000; // 3 seconds
        this.ws = null;
        this.isConnected = false;
        this.stopPing = false;
        this.pingTimer = null;
        this.reconnectTimer = null;
    }

    /**
     * send ping message
     * @private
     */
    sendPing() {
        if (!this.stopPing && this.isConnected && this.ws) {
            try {
                const pingData = {
                    op: 'ping',
                    ping: Date.now()
                };
                console.log("send ping message:", JSON.stringify(pingData));
                this.ws.send(JSON.stringify(pingData));
            } catch (error) {
                console.error("send ping error:", error.message);
                this.isConnected = false;
            }
        }
    }

    /**
     * handle received messages
     * @param {string} message  message content
     * @private
     */
    handleMessage(message) {
        try {
            const data = JSON.parse(message);
            console.log("received original message:", message);
            
            if (data.op) {
                switch (data.op) {
                    case 'pong':
                        console.log("received pong message:", message);
                        return;
                    case 'connect':
                        console.log("received connection response:", message);
                        return;
                }
            }

            // define allowed public channels
            const allowedChannels = ['depth_book1', 'trade', 'ticker'];
            if (data.ch && allowedChannels.includes(data.ch)) {
                this.processMessage(data);
            } else {
                console.log("received unknown type message:", message);
            }
        } catch (error) {
            console.error("handle message error:", error.message);
        }
    }

    /**
     * handle business messages
     * @param {Object} message  message object
     * @private
     */
    processMessage(message) {
        try {
            switch (message.ch) {
                case 'trade':
                    console.log("received trade data:", JSON.stringify(message.data));
                    this.emit('trade', message.data);
                    break;
                case 'ticker':
                    console.log("received 24-hour ticker data:", JSON.stringify(message.data));
                    this.emit('ticker', message.data);
                    break;
                case 'depth_book1':
                    console.log("received order book depth data:", JSON.stringify(message.data));
                    this.emit('depth', message.data);
                    break;
            }
        } catch (error) {
            console.error("handle business message error:", error.message);
        }
    }

    /**
     * subscribe channels
     * @param {Array} channels  channels to subscribe
     */
    subscribe(channels) {
        if (!this.isConnected || !this.ws) {
            throw new Error("WebSocket is not connected");
        }

        const msg = JSON.stringify({
            op: 'subscribe',
            args: channels
        });
        this.ws.send(msg);
        console.log("public channel subscribed successfully");
    }

    /**
     * connect WebSocket
     * @private
     */
    connect() {
        try {
            this.ws = new WebSocket(this.baseUrl, {
                headers: {
                    'User-Agent': 'Node.js WebSocket Client'
                }
            });

            this.ws.on('open', () => {
                this.isConnected = true;
                console.log("WebSocket connected successfully - public channel");

                // start heartbeat
                this.startHeartbeat();

                // subscribe after connection
                this.subscribe([
                    {symbol: "BTCUSDT", ch: "trade"},
                    {symbol: "BTCUSDT", ch: "ticker"},
                    {symbol: "BTCUSDT", ch: "depth_book1"}
                ]);

                // trigger connected event
                this.emit('connected');
            });

            this.ws.on('message', (data) => {
                this.handleMessage(data.toString());
            });

            this.ws.on('close', (code, reason) => {
                console.log(`WebSocket connection closed: ${code} ${reason}`);
                this.isConnected = false;
                this.ws = null;
                
                // clear heartbeat timer
                if (this.pingTimer) {
                    clearInterval(this.pingTimer);
                    this.pingTimer = null;
                }

                // delay reconnect
                this.reconnectTimer = setTimeout(() => {
                    this.connect();
                }, this.reconnectInterval);

                // trigger disconnected event
                this.emit('disconnected', code, reason);
            });

            this.ws.on('error', (error) => {
                console.error("WebSocket error:", error.message);
                this.isConnected = false;
                this.ws = null;

                // trigger error event
                this.emit('error', error);
            });

        } catch (error) {
            console.error("connection failed:", error.message);
            this.isConnected = false;
            this.ws = null;
            
            // delay reconnect
            this.reconnectTimer = setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        }
    }

    /**
     * start heartbeat
     * @private
     */
    startHeartbeat() {
        this.stopPing = false;
        this.pingTimer = setInterval(() => {
            this.sendPing();
        }, this.heartbeatInterval);
    }

    /**
     * start WebSocket client
     */
    start() {
        this.connect();
    }

    /**
     * stop WebSocket client
     */
    stop() {
        this.stopPing = true;
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.isConnected = false;
    }
}

// example usage
async function main() {
    const config = require('./config.json');
    const client = new OpenApiWsFuturePublic(config);
    
    // listen events
    client.on('connected', () => {
        console.log('WebSocket connected');
    });

    client.on('disconnected', (code, reason) => {
        console.log('WebSocket disconnected:', code, reason);
    });

    client.on('error', (error) => {
        console.error('WebSocket error:', error.message);
    });

    client.on('trade', (data) => {
        console.log('received trade data:', data);
    });

    client.on('ticker', (data) => {
        console.log('received ticker data:', data);
    });

    client.on('depth', (data) => {
        console.log('received depth data:', data);
    });

    // start client
    client.start();
}

// if this file is directly run, then execute the main function
if (require.main === module) {
    main().catch(console.error);
}

module.exports = OpenApiWsFuturePublic; 