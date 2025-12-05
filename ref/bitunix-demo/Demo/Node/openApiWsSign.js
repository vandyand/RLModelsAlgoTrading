const crypto = require('crypto');

/**
 * generate a random string as nonce
 * @returns {string} 32-bit random string
 */
function generateNonce() {
    const characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let nonce = '';
    for (let i = 0; i < 32; i++) {
        nonce += characters[Math.floor(Math.random() * characters.length)];
    }
    return nonce;
}

/**
 * generate current timestamp
 * @returns {string} timestamp
 */
function generateTimestamp() {
    return Math.floor(Date.now() / 1000).toString();
}

/**
 * calculate the SHA256 hash of the input string
 * @param {string} inputString  input string
 * @returns {string} SHA256 hash value
 */
function sha256Hex(inputString) {
    return crypto.createHash('sha256').update(inputString).digest('hex');
}

/**
 * generate the authentication signature
 * @param {string} nonce  random string
 * @param {string} timestamp  timestamp
 * @param {string} apiKey API key
 * @param {string} secretKey secret key
 * @returns {string} signature
 */
function generateSign(nonce, timestamp, apiKey, secretKey) {
    const digestInput = nonce + timestamp + apiKey;
    const digest = sha256Hex(digestInput);
    const signInput = digest + secretKey;
    return sha256Hex(signInput);
}

/**
 * generate WebSocket authentication data
 * @param {string} apiKey API key
 * @param {string} secretKey secret key
 * @returns {Object} WebSocket authentication data
 */
function getAuthWsFuture(apiKey, secretKey) {
    const nonce = generateNonce();
    const timestamp = generateTimestamp();
    const sign = generateSign(nonce, timestamp, apiKey, secretKey);
    
    return {
        apiKey,
        timestamp: parseInt(timestamp),
        nonce,
        sign
    };
}

module.exports = {
    generateNonce,
    generateTimestamp,
    sha256Hex,
    generateSign,
    getAuthWsFuture
}; 