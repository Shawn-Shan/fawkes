#Pass in user id's for mirroring host user
docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -t fawkes .
