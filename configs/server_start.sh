#!/usr/bin/env bash
cd "$(dirname "$0")"
echo "eula=true" > eula.txt
java -Xmx4G -jar fabric-server-launch.jar nogui
