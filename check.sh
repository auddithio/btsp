#!/bin/bash

# Define the file path in a variable
FILE="/vision/group/ego4d/v1/clips/e9559fa8-9678-42be-a658-4c8d06b4a7b5.mp4"

# Check if the file exists and is a regular file
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
fi
