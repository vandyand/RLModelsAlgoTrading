const crypto = require('crypto');

/**
 * OpenAPI HTTP signature tool class
 */
class OpenApiHttpSign {
    /**
     * generate a random string as nonce
     * @returns {string} 32-bit random string
     */
    static getNonce() {
        return crypto.randomBytes(16).toString('hex');
    }

    /**
     * get current timestamp (millisecond)
     * @returns {string} millisecond timestamp
     */
    static getTimestamp() {
        return Date.now().toString();
    }

    /**
     * generate signature
     * @param {string} apiKey API key
     * @param {string} secretKey secret key
     * @param {string} nonce random string
     * @param {string} timestamp timestamp
     * @param {string} queryParams sorted query params string (no spaces)
     * @param {string} body original JSON string (no spaces)
     * @returns {string} signature
     */
    static generateSignature(
        apiKey,
        secretKey,
        nonce,
        timestamp,
        queryParams = "",
        body = ""
    ) {
        const digestInput = nonce + timestamp + apiKey + queryParams + body;
        const digest = crypto.createHash('sha256').update(digestInput).digest('hex');
        const signInput = digest + secretKey;
        return crypto.createHash('sha256').update(signInput).digest('hex');
    }

    /**
     * get authentication header information
     * @param {string} apiKey API key
     * @param {string} secretKey secret key
     * @param {Object} queryParams query parameters
     * @param {string} body request body
     * @returns {Object} authentication header information
     */
    static getAuthHeaders(
        apiKey,
        secretKey,
        queryParams = {},
        body = ""
    ) {
        const nonce = this.getNonce();
        const timestamp = this.getTimestamp();
        const queryParamsStr = this.sortParams(queryParams);

        const sign = this.generateSignature(
            apiKey,
            secretKey,
            nonce,
            timestamp,
            queryParamsStr,
            body
        );
        
        return {
            "api-key": apiKey,
            "sign": sign,
            "nonce": nonce,
            "timestamp": timestamp
        };
    }

    /**
     * sort parameters and concatenate
     * @param {Object} params parameter dictionary
     * @returns {string} sorted parameter string
     */
    static sortParams(params) {
        if (!params || Object.keys(params).length === 0) {
            return "";
        }
        
        // sort by key and concatenate directly
        return Object.keys(params)
            .sort()
            .map(key => key + params[key])
            .join('');
    }
}

module.exports = OpenApiHttpSign; 