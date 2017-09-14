#!/bin/bash

function finish {
	kill $pid
}

trap 'finish' SIGTERM

cd /app
mvn clean install fizzed-watcher:run &
pid=$!

wait
