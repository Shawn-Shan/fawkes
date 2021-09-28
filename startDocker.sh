#Map Parrent directory to workspace folder under home directory
docker run -it -v "$(pwd)/..:/home/testuser/workspace" fawkes bash
