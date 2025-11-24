# Publish a new release

1. Update the semantic version and (if applicable) the version of the learning loop node library in the file `pyproject.toml` in both the `trainer` and `detector` folders.

2. Build and upload the new docker images by running the following commands in the `training` and `detection` folders:

```bash
# build docker image
./docker.sh b

# publish docker image
./docker.sh p
```

3. Tag the new release in git on the main branch (after merging the changes)
