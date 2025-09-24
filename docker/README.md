
## Setting up Docker
1. Set the DOCKER_ARCH env variable:

If you are using x86 CPU (Intel, AMD)

```
make clone_dependencies # Make sure you're added to the private fastfusion repo!
make build_docker
make run_docker
```

If there's update to the dependencies, run:

```
make pull_dependencies
```
