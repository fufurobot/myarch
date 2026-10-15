please give me an arch linux dockerfile jinja2 template that:

1. create users and groups with proper sudo:
administrator-master: can sudo everything
pikaur-manager: can only sudo `/usr/bin/pikaur -Syu [!-]* --pikaur-config=/home/pikaur-manager/.config/pikaur.conf --user-id=1005 --group-id=1005 --home-dir=/home/pikaur-manager --xdg-cache-home=/home/pikaur-manager/.cache --xdg-data-home=/home/pikaur-manager/.local/share`
developer: can only sudo on `/usr/bin/pacman`. 
system-manager: can only sudo on `fdisk`, `mount`, and `ls`. 
service-manager: can only sudo on `systemctl`.
user: just user, can sudo nothing. 
then we install sshd, and set default password (and ssh login able) for:

user: `iamsafe`
pikaur-manager: `iknowtheremotecodeisdangerous`
developer: `iwillnotinstallpackagesidonotownsourcecode`
system-manager: `iwillnotstealotherusersdata`
service-manager: `iwillnotstartuntrustedservices`

and use `chage -d 0 <username>` to force users to change to their password after first login. 

2. install newest pikaur by makepkg; 
3. install the properity driver if nvidia/amd gpu detected on host (templated); if driver too old, use pikaur to find proper drivers. after this step, *-smi should be available. 
4. install kde and vnc server, and configure properly with one 1920*1080 virtual screen;
5. install uv and use uv to install the python version match the python script running env (conditional templated, if too old use the oldest one that uv supports).
6. install pytorch by uv with proper config (nvidia/amd/cpu).
7. install vllm if gpu is found and use it as the backend; if no, install llama.cpp and accept only gguf files on cpu.
8. install coder server (READ https://github.com/coder/code-server-aur to understand how to install the latest stable version because the aur package is not actively maintained) for remote development, with proper plugin configured to the vllm/llama.cpp instance. 
9. install aichat and configure it with vllm/llama.cpp instance. 

and also, a python script to detect host and build proper image. user can specify an gguf file or a huggingface model folder to pack the docker image.

