package com.bitunix.openapi.response;

import com.fasterxml.jackson.annotation.JsonProperty;

public class BatchFundingRate {
    /**
     * trading pair
     */
    private String symbol;

    /**
     * mark price
     */
    @JsonProperty("markPrice")
    private Double markPrice;

    /**
     * last price
     */
    @JsonProperty("lastPrice")
    private Double lastPrice;

    /**
     * funding rate
     */
    @JsonProperty("fundingRate")
    private Double fundingRate;

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public Double getMarkPrice() {
        return markPrice;
    }

    public void setMarkPrice(Double markPrice) {
        this.markPrice = markPrice;
    }

    public Double getLastPrice() {
        return lastPrice;
    }

    public void setLastPrice(Double lastPrice) {
        this.lastPrice = lastPrice;
    }

    public Double getFundingRate() {
        return fundingRate;
    }

    public void setFundingRate(Double fundingRate) {
        this.fundingRate = fundingRate;
    }
} 