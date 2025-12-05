class ErrorCode {
    // general error code (10000-10099)
    static SUCCESS = [0, "Success"];
    static NETWORK_ERROR = [10001, "Network Error"];
    static PARAMETER_ERROR = [10002, "Parameter Error"];
    static API_KEY_EMPTY = [10003, "api-key can't be empty"];
    static IP_NOT_IN_WHITELIST = [10004, "The current ip is not in the apikey ip whitelist"];
    static TOO_MANY_REQUESTS = [10005, "Too many requests, please try again later"];
    static REQUEST_TOO_FREQUENTLY = [10006, "Request too frequently"];
    static SIGN_SIGNATURE_ERROR = [10007, "Sign signature error"];
    static VALUE_NOT_COMPLY = [10008, "{value} does not comply with the rule, optional [correctValue]"];

    // market related error code (20000-20099)
    static MARKET_NOT_EXISTS = [20001, "Market not exists"];
    static POSITION_EXCEED_LIMIT = [20002, "The current positions amount has exceeded the maximum open limit, please adjust the risk limit"];
    static INSUFFICIENT_BALANCE = [20003, "Insufficient balance"];
    static INSUFFICIENT_TRADER = [20004, "Insufficient Trader"];
    static INVALID_LEVERAGE = [20005, "Invalid leverage"];
    static CANNOT_CHANGE_LEVERAGE = [20006, "You can't change leverage or margin mode as there are open orders"];
    static ORDER_NOT_FOUND = [20007, "Order not found, please try it later"];
    static INSUFFICIENT_AMOUNT = [20008, "Insufficient amount"];
    static POSITION_MODE_UPDATE_FAILED = [20009, "Position exists, so positions mode cannot be updated"];

    // transaction related error code (30000-30099)
    static ORDER_FAILED_LIQUIDATION = [30001, "Failed to order. Please adjust the order price or the leverage as the order price dealt may immediately liquidate."];
    static PRICE_BELOW_LIQUIDATED = [30002, "Price below liquidated price"];
    static PRICE_ABOVE_LIQUIDATED = [30003, "Price above liquidated price"];
    static POSITION_NOT_EXIST = [30004, "Position not exist"];
    static TRIGGER_PRICE_TOO_CLOSE = [30005, "The trigger price is closer to the current price and may be triggered immediately"];

    /**
     * get error message by error code
     * @param {number} code  error code
     * @returns {array|null} return the array contains error code and error message, if not found, return null
     */
    static getByCode(code) {
        for (const [key, value] of Object.entries(this)) {
            if (Array.isArray(value) && value[0] === code) {
                return value;
            }
        }
        return null;
    }

    /**
     * get error message by error code
     * @param {number} code  error code
     * @returns {string} error message
     */
    static getMessage(code) {
        const error = this.getByCode(code);
        if (!error) {
            return `Unknown error code: ${code}`;
        }
        return `Error ${error[0]}: ${error[1]}`;
    }
}

module.exports = ErrorCode; 