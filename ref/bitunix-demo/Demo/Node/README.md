# Bitunix Futures API Node.js Demo

This is a Node.js example code for the Bitunix Futures API, demonstrating how to interact with the Bitunix futures trading platform using HTTP and WebSocket APIs.

## Directory Structure

```
.
├── README.md                     # This document
├── config.json                   # Configuration file
├── errorCodes.js                # Error code definitions
├── openApiHttpFuturePrivate.js  # HTTP private API implementation
├── openApiHttpFuturePublic.js   # HTTP public API implementation
├── openApiHttpSign.js           # API signature implementation
├── openApiWsFuturePrivate.js    # WebSocket private API implementation
└── openApiWsFuturePublic.js     # WebSocket public API implementation
```

## Requirements

- Node.js >= 14.0.0
- npm >= 6.0.0

## Installation

```bash
npm install
```

## Configuration

Before using, please configure your API keys in the `config.json` file:

```json
{
    "credentials": {
        "api_key": "Your API Key",
        "secret_key": "Your Secret Key"
    },
    "http": {
        "uri_prefix": "https://api.bitunix.com"
    },
    "ws": {
        "uri_prefix": "wss://api.bitunix.com"
    }
}
```

## Running Examples

```bash
cd Demo/Node
```

### HTTP Public API Example

```bash
npm run start:http:public
```

This will run the HTTP public API example, including:
- Get trading pair information
- Get order book depth
- Get latest trades
- Get K-line data
- Get 24-hour ticker

### HTTP Private API Example

```bash
npm run start:http:private
```

This will run the HTTP private API example, including:
- Get account information
- Place orders
- Cancel orders
- Get current positions
- Get historical positions
- Get historical orders

### WebSocket Public API Example

```bash
npm run start:ws:public
```

This will run the WebSocket public API example, including:
- Subscribe to order book depth
- Subscribe to latest trades
- Subscribe to K-line data
- Subscribe to 24-hour ticker

### WebSocket Private API Example

```bash
npm run start:ws:private
```

This will run the WebSocket private API example, including:
- Subscribe to account updates
- Subscribe to position updates
- Subscribe to order updates

## API Documentation

For more API details, please refer to the [Bitunix API Documentation](https://api.bitunix.com/docs).

## Notes

1. Before running private API examples, please ensure your API keys are properly configured.
2. The order placement functionality in the example code is commented out by default. If you want to test it, please read the code carefully and understand the risks.
3. It is recommended to test in a test environment first to avoid any financial losses.
4. WebSocket connections automatically handle reconnection, but it's recommended to add more error handling in production environments.

## Error Handling

All example code includes basic error handling, with error codes defined in the `errorCodes.js` file. In actual use, it's recommended to add more comprehensive error handling mechanisms based on your needs.

## Contributing

If you find any issues or have suggestions for improvements, please feel free to submit an Issue or Pull Request.

## License

MIT License
