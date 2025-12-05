const axios = require('axios');
const OpenApiHttpSign = require('./openApiHttpSign');
const ErrorCode = require('./errorCodes');

/**
 * OpenAPI HTTP future private API client
 */
class OpenApiHttpFuturePrivate {
    /**
     * initialize OpenApiHttpFuturePrivate class
     * @param {Object} config  configuration object
     */
    constructor(config) {
        this.config = config;
        this.apiKey = config.credentials.api_key;
        this.secretKey = config.credentials.secret_key;
        this.baseUrl = config.http.uri_prefix;
        
        // set common request headers
        this.defaultHeaders = {
            "language": "en-US",
            "Content-Type": "application/json"
        };
    }

    /**
     * handle response
     * @param {Object} response  response data
     * @returns {Object} processed data
     * @throws {Error} throw an error when the response status code is not 200 or the business status code is not 0
     * @private
     */
    handleResponse(response) {
        const { code, msg, data } = response;
        if (code !== 0) {
            const error = ErrorCode.getByCode(code);
            if (error) {
                throw new Error(JSON.stringify(error));
            }
            throw new Error(`未知错误: ${code} - ${msg}`);
        }
        return data;
    }

    /**
     * send HTTP request
     * @param {string} method  request method
     * @param {string} url  request URL
     * @param {Object} params  query parameters
     * @param {Object} data  request body data
     * @returns {Promise<Object>} response data
     * @private
     */
    async sendRequest(method, url, params = {}, data = {}) {
        try {
            // prepare request headers
            const headers = { ...this.defaultHeaders };
            
            // prepare authentication headers
            let authHeaders;
            if (method === 'POST') {
                const jsonData = Object.keys(data).length > 0 ? JSON.stringify(data) : '';
                authHeaders = OpenApiHttpSign.getAuthHeaders(this.apiKey, this.secretKey, {}, jsonData);
            } else {
                authHeaders = OpenApiHttpSign.getAuthHeaders(this.apiKey, this.secretKey, params);
            }
            
            Object.assign(headers, authHeaders);

            // send request
            const response = await axios({
                method,
                url: `${this.baseUrl}${url}`,
                params: method === 'GET' ? params : undefined,
                data: method === 'POST' ? data : undefined,
                headers
            });

            return this.handleResponse(response.data);
        } catch (error) {
            if (error.response) {
                throw new Error(`HTTP错误: ${error.response.status}`);
            }
            throw error;
        }
    }

    /**
     * get account information
     * @param {string} marginCoin  margin coin, default USDT
     * @returns {Promise<Object>} account information
     */
    async getAccount(marginCoin = "USDT") {
        const url = "/api/v1/futures/account";
        const params = {
            marginCoin
        };
        
        return this.sendRequest('GET', url, params);
    }

    /**
     * place order
     * @param {Object} options  place order parameters
     * @param {string} options.symbol  trading pair
     * @param {string} options.side  direction, BUY or SELL
     * @param {string} options.orderType  order type, LIMIT or MARKET
     * @param {string} options.qty  quantity
     * @param {string} [options.price]  price, when orderType is LIMIT
     * @param {string} [options.positionId]  position id
     * @param {string} [options.tradeSide]  trade side, OPEN or CLOSE, default OPEN
     * @param {string} [options.effect]  order effective time, default GTC
     * @param {boolean} [options.reduceOnly]  reduce only, default false
     * @param {string} [options.clientId]  client id
     * @param {string} [options.tpPrice]  take profit price
     * @param {string} options.symbol  trading pair
     * @param {string} options.side  direction, BUY or SELL
     * @param {string} options.orderType  order type, LIMIT or MARKET
     * @param {string} options.qty  quantity
     * @param {string} [options.price]  price, when orderType is LIMIT
     * @param {string} [options.positionId]  position id
     * @param {string} [options.tradeSide]  trade side, OPEN or CLOSE, default OPEN
     * @param {string} [options.effect]  order effective time, default GTC
     * @param {boolean} [options.reduceOnly]  reduce only, default false
     * @param {string} [options.clientId]  client id
     * @param {string} [options.tpPrice]  take profit price
     * @param {string} [options.tpStopType]  take profit stop type, MARK or LAST, default MARK
     * @param {string} [options.tpOrderType]  take profit order type, LIMIT or MARKET
     * @param {string} [options.tpOrderPrice]  take profit order price
     * @returns {Promise<Object>} order information
     */
    async placeOrder({
        symbol,
        side,
        orderType,
        qty,
        price,
        positionId,
        tradeSide = "OPEN",
        effect = "GTC",
        reduceOnly = false,
        clientId,
        tpPrice,
        tpStopType,
        tpOrderType,
        tpOrderPrice
    }) {
        const url = "/api/v1/futures/trade/place_order";
        
        const data = {
            symbol,
            side,
            orderType,
            qty,
            tradeSide,
            effect,
            reduceOnly
        };
        
        if (price) data.price = price;
        if (positionId) data.positionId = positionId;
        if (clientId) data.clientId = clientId;
        if (tpPrice) data.tpPrice = tpPrice;
        if (tpStopType) data.tpStopType = tpStopType;
        if (tpOrderType) data.tpOrderType = tpOrderType;
        if (tpOrderPrice) data.tpOrderPrice = tpOrderPrice;
        
        return this.sendRequest('POST', url, {}, data);
    }

    /**
     * cancel order
     * @param {string} symbol  trading pair
     * @param {Array<Object>} orderList  order list, each order contains orderId or clientId
     * @returns {Promise<Object>} cancel order result
     */
    async cancelOrders(symbol, orderList) {
        const url = "/api/v1/futures/trade/cancel_orders";
        
        const data = {
            symbol,
            orderList
        };
        
        return this.sendRequest('POST', url, {}, data);
    }

    /**
     * get history orders
     * @param {string} [symbol]  trading pair, if not provided, query all trading pairs
     * @returns {Promise<Array>} history orders list
     */
    async getHistoryOrders(symbol = null) {
        const url = "/api/v1/futures/trade/get_history_orders";
        const params = {};
        if (symbol) {
            params.symbol = symbol;
        }
        
        return this.sendRequest('GET', url, params);
    }

    /**
     * get history positions
     * @param {string} [symbol]  trading pair, if not provided, query all trading pairs
     * @returns {Promise<Array>} history positions list
     */
    async getHistoryPositions(symbol = null) {
        const url = "/api/v1/futures/position/get_history_positions";
        const params = {};
        if (symbol) {
            params.symbol = symbol;
        }
        
        return this.sendRequest('GET', url, params);
    }

    /**
     * get current positions
     * @param {string} [symbol]  trading pair, if not provided, query all trading pairs
     * @returns {Promise<Array>} current positions list
     */
    async getCurrentPositions(symbol = null) {
        const url = "/api/v1/futures/position/get_pending_positions";
        const params = {};
        if (symbol) {
            params.symbol = symbol;
        }
        
        return this.sendRequest('GET', url, params);
    }

    /**
     * adjust margin
     * @param {string} symbol  trading pair
     * @param {string} positionId  position id
     * @param {string} amount  adjust amount
     * @param {string} type  adjust type, ADD or SUB
     * @returns {Promise<Object>} adjust result
     */
    async adjustMargin(symbol, positionId, amount, type) {
        const url = "/api/v1/futures/position/adjust_margin";
        
        const data = {
            symbol,
            positionId,
            amount,
            type
        };
        
        return this.sendRequest('POST', url, {}, data);
    }
}

// test main function
async function main() {
    try {
        // load config from config file
        const config = require('./config.json');
        
        // create http private api client instance
        const client = new OpenApiHttpFuturePrivate(config);
        
        console.log('testing http private api...');
        console.log('using api config:', {
            baseUrl: config.http.uri_prefix,
            apiKey: config.credentials.api_key.substring(0, 8) + '...'  // only show part of api key
        });

        // 1. test get account info
        try {
            console.log('\n1. get account info:');
            const accountInfo = await client.getAccount();
            console.log(JSON.stringify(accountInfo, null, 2));
        } catch (error) {
            console.error('get account info failed:', error.message);
        }

        // 2. test get current positions
        try {
            console.log('\n2. get current positions:');
            console.log('requesting url:', `${config.http.uri_prefix}/api/v1/futures/position/get_positions`);
            const positions = await client.getCurrentPositions();
            console.log(JSON.stringify(positions, null, 2));
        } catch (error) {
            console.error('get current positions failed:', error.message);
            if (error.response) {
                console.error('response status code:', error.response.status);
                console.error('response data:', error.response.data);
            }
        }

        // 3. test get history orders
        try {
            console.log('\n3. get history orders:');
            const historyOrders = await client.getHistoryOrders();
            console.log(JSON.stringify(historyOrders, null, 2));
        } catch (error) {
            console.error('get history orders failed:', error.message);
        }

        // 4. test place order (note: use small quantity to test)
        try {
            console.log('\n4. test place order:');
            const orderResult = await client.placeOrder({
                symbol: 'BTCUSDT',
                side: 'BUY',
                orderType: 'LIMIT',
                qty: '0.001',
                price: '25000',  // set a low price to avoid actual execution
                tradeSide: 'OPEN'
            });
            console.log(JSON.stringify(orderResult, null, 2));
        } catch (error) {
            console.error('place order failed:', error.message);
            if (error.response) {
                console.error('response status code:', error.response.status);
                console.error('response data:', error.response.data);
            }
        }

    } catch (error) {
        console.error('error occurred during testing:', error.message);
    }
}

// if this file is directly run, then execute the test
if (require.main === module) {
    main();
}

module.exports = OpenApiHttpFuturePrivate; 