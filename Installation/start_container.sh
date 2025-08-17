#!/bin/bash
# This script launches a Docker container with the necessary configurations for GUI applications.
# Usage: ./start_container.sh <host_OS> <optional_container_name> <optional_image_name>

# Check if the user has provided a host OS, otherwise quit and display usage
if [ -z "$1" ]; then
    echo "Usage: ./start_container.sh <host_OS> <optional_container_name> <optional_image_name>"
    echo "Example: ./start_container.sh linux"
    exit 1
fi
# Check if the provided host OS is supported
if [ "$1" != "linux" ] && [ "$1" != "macos" ]; then
    echo "Unsupported host OS: $1. Supported OS are: linux and macos."
    exit 1
fi
# Set the host OS variable
HOST_OS="$1"


# Check if the user has provided a container name, otherwise use a default name
if [ -z "$2" ]; then
    CONTAINER_NAME="ece4078_2025_lab"
else
    CONTAINER_NAME="$2"
fi

# Check if the user has provided an image name, otherwise use a default image name
if [ -z "$3" ]; then
    IMAGE_NAME="adedayoakinade/ece4078:noetic"
else
    IMAGE_NAME="$3"
fi

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "Container '$CONTAINER_NAME' is already running."
    exit 0
fi


# If the host OS is Linux, allow local root access to the X server and start the container
if [ "$HOST_OS" == "linux" ]; then
    # Allow local root access to the X server
    xhost +local:root

    # Check if the container exists
    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "Starting existing container '$CONTAINER_NAME'..."
        docker start $CONTAINER_NAME
    else
        echo "Container '$CONTAINER_NAME' does not exist. Starting Docker container for host OS: $HOST_OS..."
        # echo "Starting Docker container for host OS: $HOST_OS"
        docker run -it   --env="DISPLAY=$DISPLAY"   --env="QT_X11_NO_MITSHM=1" --env="LIBGL_ALWAYS_SOFTWARE=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"   --network=host   --privileged   --name $CONTAINER_NAME   $IMAGE_NAME
    fi

elif [ "$HOST_OS" == "macos" ]; then
    # Get XQuartz running
    if ! pgrep XQuartz >/dev/null; then
        echo "Launching XQuartz..."
        open -a XQuartz
        sleep 2
    fi

    # extract ip from pc
    PC_IP=$(/usr/sbin/ipconfig getifaddr en0)
    xhost + "$PC_IP"

    if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        # If container exists but not running
        echo "Starting existing container '$CONTAINER_NAME'..."
        docker start $CONTAINER_NAME
        docker attach $CONTAINER_NAME
    else
        echo "Starting new container '$CONTAINER_NAME' on macOS..."
        docker run -it -e DISPLAY="${PC_IP}:0" \
          -e SDL_VIDEODRIVER=x11 \
          -e LIBGL_ALWAYS_INDIRECT=1 \
          -e LIBGL_ALWAYS_SOFTWARE=1 \
          --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
          --name $CONTAINER_NAME \
          $IMAGE_NAME
        #   --volume="../../ECE4078_Lab_2025/:/home/ece4078/ECE4078_Lab/" \
    fi
    exit 0

# elif [ "$HOST_OS" == "windows" ]; then
#     # For Windows, we assume XLaunch (VcXsrv) is running with access control disabled.
#     DISPLAY_VAR="host.docker.internal:0.0"

#     if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
#         echo "Starting existing container '$CONTAINER_NAME'..."
#         docker start -ai $CONTAINER_NAME
#     else
#         echo "Container '$CONTAINER_NAME' does not exist. Starting Docker container for host OS: $HOST_OS..."
#         docker run -it \
#             -e DISPLAY=$DISPLAY_VAR \
#             --privileged \
#             --name $CONTAINER_NAME \
#             $IMAGE_NAME
#     fi
#     exit 0

else
    echo "Unsupported host OS: $HOST_OS"
    exit 1
fi
