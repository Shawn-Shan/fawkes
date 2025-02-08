#!/bin/bash

# Publish the package
poetry publish --build

# Get the current version
VERSION=$(poetry version -s)

# Create and push a Git tag
git tag v$VERSION
git push origin v$VERSION