# syntax=docker/dockerfile:1

FROM golang:1.24 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /app/bin/broker ./cmd/broker

FROM gcr.io/distroless/base-debian12
WORKDIR /srv/app
ENV BROKER_LISTEN_ADDR=:8080 \
    BROKER_DATA_DIR=/data
COPY --from=builder /app/bin/broker /usr/local/bin/broker
VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/broker"]
