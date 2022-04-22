# Build Basic Docker Image

```
git clone git@github.com:Joy1112/vanilla-docker-images.git
docker build -t ubuntu18.04-anaconda3/cuda-11.0 vanilla-docker-images/ubuntu-anaconda
```

# Build PYMARL2 Docker Image

```
git clone --recursive git@github.com:Joy1112/pymarl-football.git
docker build -t pymarl/football pymarl-football
```

# Run Exp
## Start a New Container
If folder `pymarl_results` does not exist, create it first and the results of the exps will be in it.
```
mkdir <what_path>/pymarl_results
# start a container
docker run --name <what_name> -it --gpus "devices=0" -u $(id -u):$(id -g) -v <what_path>/pymarl_results:/home/docker/pymarl2/results pymarl/football
```
## Run Exp
```
# run in docker
python src/main.py --config=wurl_gfootball --env-config=gfootball
# then ctrl+P+Q to left the docker container.
```
## Check the outputs
The outputs can be viewd by
```
# view the last 100 lines of the outputs
docker logs <what_name> --tail=100
```

# Visualization
`<path>` should be absolute path or relative path like `results/models/wurl_qmix_football__2022-04-21_19-38-03`.
Remember to copy the config.json from the `sacred/` to `models/`
```
python src/main.py --exp-config=<path> with vis_process=True --env-config=gfootball_vis
```
