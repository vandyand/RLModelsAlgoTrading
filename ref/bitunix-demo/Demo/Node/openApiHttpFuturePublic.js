const axios = require('axios');
const ErrorCode = require('./errorCodes');
const fs = require('fs');
const path = require('path');

class OpenApiHttpFuturePublic {
    /**
     * initialize the futures public api client
     * @param {string} configPath  config file path, default is 'config.json'
     */
    constructor(configPath = 'config.json') {
        this.loadConfig(configPath);
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'language': 'en-US'
        };
    }

    /**
     * load config file
     * @param {string} configPath  config file path
     * @private
     */
    loadConfig(configPath) {
        try {
            const configContent = fs.readFileSync(path.resolve(__dirname, configPath), 'utf8');
            const config = JSON.parse(configContent);
            this.baseUrl = config.http.uri_prefix;
        } catch (error) {
            throw new Error(`failed to load config file: ${error.message}`);
        }
    }

    /**
     * handle response data
     * @param {Object} response  response data
     * @returns {Object} processed response data
     * @private
     */
    handleResponse(response) {
        const { code, msg, data } = response;
        if (code !== 0) {
            const error = ErrorCode.getByCode(code);
            if (error) {
                throw new Error(ErrorCode.getMessage(code));
            }
            throw new Error(`未知错误: ${code} - ${msg}`);
        }
        return data;
    }

    /**
     * send HTTP request
     * @param {string} method  request method
     * @param {string} url  request URL
     * @param {Object} params  request parameters
     * @returns {Promise<Object>} response data
     * @private
     */
    async sendRequest(method, url, params = {}) {
        try {
            const config = {
                method,
                url: `${this.baseUrl}${url}`,
                headers: this.defaultHeaders
            };

            if (method === 'GET') {
                config.params = params;
            } else {
                config.data = params;
            }

            const response = await axios(config);
            return this.handleResponse(response.data);
        } catch (error) {
            if (error.response) {
                throw new Error(`HTTP错误: ${error.response.status}`);
            }
            throw error;
        }
    }

    /**
     * get futures trading pair market data
     * @param {string} [symbols]  futures trading pair, multiple separated by commas, e.g. BTCUSDT,ETHUSDT
     * @returns {Promise<Object>} market data
     */
    async getTickers(symbols = null) {
        const params = {};
        if (symbols) {
            params.symbols = symbols;
        }
        return this.sendRequest('GET', '/api/v1/futures/market/tickers', params);
    }

    /**
     * get depth data
     * @param {string} symbol  futures trading pair
     * @param {number} [limit=100]  depth quantity
     * @returns {Promise<Object>} depth data
     */
    async getDepth(symbol, limit = 100) {
        const params = {
            symbol,
            limit
        };
        return this.sendRequest('GET', '/api/v1/futures/market/depth', params);
    }

    /**
     * get K-line data
     * @param {string} symbol  futures trading pair
     * @param {string} interval K-line interval, e.g. 1m, 5m, 15m, 30m, 1h, 4h, 1d
     * @param {number} [limit=100]  data quantity, default 100, max 200
     * @param {number} [startTime]  start time (Unix timestamp, millisecond format)
     * @param {number} [endTime]  end time (Unix timestamp, millisecond format)
     * @param {string} [type='LAST_PRICE'] K-line type, optional values: LAST_PRICE (latest price), MARK_PRICE (mark price)
     * @returns {Promise<Object>} K-line data
     */
    async getKline(symbol, interval, limit = 100, startTime = null, endTime = null, type = 'LAST_PRICE') {
        const params = {
            symbol,
            interval,
            limit,
            type
        };

        if (startTime) {
            params.startTime = startTime;
        }
        if (endTime) {
            params.endTime = endTime;
        }

        return this.sendRequest('GET', '/api/v1/futures/market/kline', params);
    }

    /**
     * get batch funding rate
     * @returns {Promise<Object>} funding rate data
     */
    async getBatchFundingRate() {
        const params = {};

        return this.sendRequest('GET', '/api/v1/futures/market/funding_rate/batch', params);
    }
}

// example usage
async function main() {
    try {
        // create client instance
        const client = new OpenApiHttpFuturePublic();

        // get market data
        const tickers = await client.getTickers("BTCUSDT,ETHUSDT");
        console.log("tickers data:", JSON.stringify(tickers, null, 2));

        // get depth data
        const depth = await client.getDepth("BTCUSDT", 5);
        console.log("depth data:", JSON.stringify(depth, null, 2));

        // get K-line data
        const currentTime = Date.now();
        const oneHourAgo = currentTime - (60 * 60 * 1000);
        const klines = await client.getKline(
            "BTCUSDT",
            "1m",
            5,
            oneHourAgo,
            currentTime,
            "LAST_PRICE"
        );
        console.log("K-line data:", JSON.stringify(klines, null, 2));

        // get batch funding rate
        const fundingRates = await client.getBatchFundingRate();
        console.log("funding rate data:", JSON.stringify(fundingRates, null, 2));

    } catch (error) {
        console.error("error:", error.message);
    }
}

// if this file is directly run, then execute the main function
if (require.main === module) {
    main();
}

module.exports = OpenApiHttpFuturePublic; 