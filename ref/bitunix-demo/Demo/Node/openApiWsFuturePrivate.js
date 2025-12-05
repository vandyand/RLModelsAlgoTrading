const WebSocket = require('ws');
const EventEmitter = require('events');
const wsSign = require('./openApiWsSign');

/**
 * OpenAPI WebSocket futures private API client
 */
class OpenApiWsFuturePrivate extends EventEmitter {
    /**
     * initialize the WebSocket private API client
     * @param {Object} config  config object
     */
    constructor(config) {
        super();
        this.config = config;
        this.baseUrl = config.websocket.private_uri;
        this.reconnectInterval = config.websocket.reconnect_interval * 1000; // convert to milliseconds
        this.heartbeatInterval = 3000; // 3 seconds
        this.ws = null;
        this.isConnected = false;
        this.stopPing = false;
        this.pingTimer = null;
        this.reconnectTimer = null;
        this.apiKey = config.credentials.api_key;
        this.secretKey = config.credentials.secret_key;
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
                console.log("sending ping message:", JSON.stringify(pingData));
                this.ws.send(JSON.stringify(pingData));
            } catch (error) {
                console.error("sending ping error:", error.message);
                this.isConnected = false;
            }
        }
    }

    /**
     * authenticate
     * @private
     */
    authenticate() {
        if (!this.isConnected || !this.ws) {
            throw new Error("WebSocket is not connected");
        }

        try {
            const authData = wsSign.getAuthWsFuture(this.apiKey, this.secretKey);
            const msg = JSON.stringify({
                op: 'login',
                args: [authData]
            });
            this.ws.send(msg);
            console.log("WebSocket authentication successful");
        } catch (error) {
            console.error("authentication failed:", error.message);
            throw error;
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

            // define allowed private channels
            const allowedChannels = ['balance', 'position', 'order', 'tpsl'];
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
                case 'balance':
                    const balanceData = message.data;
                    console.log("\n=== balance update ===");
                    console.log("coin:", balanceData.coin ?? 'N/A');
                    console.log("available:", balanceData.available ?? 'N/A');
                    console.log("frozen:", balanceData.frozen ?? 'N/A');
                    console.log("isolationFrozen:", balanceData.isolationFrozen ?? 'N/A');
                    console.log("crossFrozen:", balanceData.crossFrozen ?? 'N/A');
                    console.log("margin:", balanceData.margin ?? 'N/A');
                    console.log("isolationMargin:", balanceData.isolationMargin ?? 'N/A');
                    console.log("crossMargin:", balanceData.crossMargin ?? 'N/A');
                    console.log("expMoney:", balanceData.expMoney ?? 'N/A');
                    console.log("-------------------");
                    this.emit('balance', balanceData);
                    break;

                case 'position':
                    const positionData = message.data;
                    console.log("\n=== position update ===");
                    console.log("event:", positionData.event ?? 'N/A');
                    console.log("positionId:", positionData.positionId ?? 'N/A');
                    console.log("marginMode:", positionData.marginMode ?? 'N/A');
                    console.log("positionMode:", positionData.positionMode ?? 'N/A');
                    console.log("side:", positionData.side ?? 'N/A');
                    console.log("leverage:", positionData.leverage ?? 'N/A');
                    console.log("margin:", positionData.margin ?? 'N/A');
                    console.log("ctime:", positionData.ctime ?? 'N/A');
                    console.log("qty:", positionData.qty ?? 'N/A');
                    console.log("entryValue:", positionData.entryValue ?? 'N/A');
                    console.log("symbol:", positionData.symbol ?? 'N/A');
                    console.log("realizedPNL:", positionData.realizedPNL ?? 'N/A');
                    console.log("unrealizedPNL:", positionData.unrealizedPNL ?? 'N/A');
                    console.log("funding:", positionData.funding ?? 'N/A');
                    console.log("fee:", positionData.fee ?? 'N/A');
                    console.log("-------------------");
                    this.emit('position', positionData);
                    break;

                case 'order':
                    const orderData = message.data;
                    console.log("\n=== order update ===");
                    console.log("orderId:", orderData.orderId ?? 'N/A');
                    console.log("symbol:", orderData.symbol ?? 'N/A');
                    console.log("type:", orderData.type ?? 'N/A');
                    console.log("price:", orderData.price ?? 'N/A');
                    console.log("qty:", orderData.qty ?? 'N/A');
                    console.log("-------------------");
                    this.emit('order', orderData);
                    break;

                case 'tpsl':
                    const tpslData = message.data;
                    console.log("\n=== tpsl update ===");
                    console.log("symbol:", tpslData.symbol ?? 'N/A');
                    console.log("orderId:", tpslData.orderId ?? 'N/A');
                    console.log("positionId:", tpslData.positionId ?? 'N/A');
                    console.log("leverage:", tpslData.leverage ?? 'N/A');
                    console.log("side:", tpslData.side ?? 'N/A');
                    console.log("positionMode:", tpslData.positionMode ?? 'N/A');
                    console.log("type:", tpslData.type ?? 'N/A');
                    console.log("slQty:", tpslData.slQty ?? 'N/A');
                    console.log("tpOrderType:", tpslData.tpOrderType ?? 'N/A');
                    console.log("slStopType:", tpslData.slStopType ?? 'N/A');
                    console.log("slPrice:", tpslData.slPrice ?? 'N/A');
                    console.log("slOrderPrice:", tpslData.slOrderPrice ?? 'N/A');
                    console.log("-------------------");
                    this.emit('tpsl', tpslData);
                    break;
            }
        } catch (error) {
            console.error("process business message error:", error.message);
        }
    }

    /**
     * subscribe channels
     * @param {Array} channels need to subscribe channels
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
        console.log("private channel subscribed successfully");
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
                console.log("WebSocket connected successfully - private channel");

                // authenticate
                this.authenticate();

                // start heartbeat
                this.startHeartbeat();

                // subscribe after connection
                this.subscribe([
                    {ch: "balance"},
                    {ch: "position"},
                    {ch: "order"},
                    {ch: "tpsl"}
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
    const client = new OpenApiWsFuturePrivate(config);
    
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

    client.on('balance', (data) => {
        console.log('balance update:', data);
    });

    client.on('position', (data) => {
        console.log('position update:', data);
    });

    client.on('order', (data) => {
        console.log('order update:', data);
    });

    client.on('tpsl', (data) => {
        console.log('tpsl update:', data);
    });

    // start client
    client.start();
}

// if this file is directly run, then execute the main function
if (require.main === module) {
    main().catch(console.error);
}

module.exports = OpenApiWsFuturePrivate; 