#!/bin/bash

function finish {
	if [ $PID1 -ge 0 ]; then
		kill $PID1
	fi
	kill $PID2
}

trap 'finish' SIGTERM

PID1=-1
while true; do
	if [ ! -f ${JAR_NAME} ]; then
		echo -e "\n[INFO] No Jar file found. Waiting for jar before restarting JVM\n"
	fi
	while [ ! -f ${JAR_NAME} ]; do sleep 2; done
	if [ $PID1 -ge 0 ]; then
		echo -e "\n[INFO] Restarting JVM\n"
	  kill $PID1
		PID1=-1
	fi
	java -jar ${JAR_NAME} &
  PID1=$!
  inotifywait -e modify -e delete -e create -e attrib ${JAR_NAME}
done
PID2=$!

wait
