nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/yabebal:/xhome \
-v /data/alireza/:/data \
--name ttx aoflagger/kern:latest \
/bin/bash
